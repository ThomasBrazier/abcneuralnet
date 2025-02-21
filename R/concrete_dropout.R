library(torch)

nn_concrete_dropout = torch::nn_module(
  classname = "ConcreteDropout",
  # the initialize function tuns whenever we instantiate the model
  initialize = function(weight_regularizer=1e-6,
                        dropout_regularizer=1e-5,
                        init_min=0.1,
                        init_max=0.1) {

    # just for you to see when this function is called
    # cat("Calling initialize Concrete dropout!\n")
    self$weight_regularizer = weight_regularizer
    self$dropout_regularizer = dropout_regularizer

    init_min = log(init_min) - log(1 - init_min)
    init_max = log(init_max) - log(1 - init_max)

    self$p_logit = nn_parameter(nn_init_uniform_(torch_empty(1), init_min, init_max))
  },

  # this function is called whenever we call our model on input.
  forward = function(x, layer) {
    # cat("Calling forward Concrete dropout!\n")
    p = torch_sigmoid(self$p_logit)

    output = layer(self$concrete_dropout(x))

    sum_of_square = 0
    for (param in layer$parameters) {
      sum_of_square = sum_of_square + torch_sum(torch_pow(param, 2))
    }

    weights_regularizer = self$weight_regularizer * sum_of_square / (1 - p)

    dropout_regularizer = p * torch_log(p)
    dropout_regularizer = dropout_regularizer + (1 - p) * torch_log(1 - p)

    input_dimensionality = x[1]$numel() # Number of elements of first item in batch
    dropout_regularizer = dropout_regularizer * self$dropout_regularizer * input_dimensionality

    self$regularization = weights_regularizer + dropout_regularizer

    return(output)
  },

  concrete_dropout = function(x) {
    # Computes the Concrete Dropout
    eps = 1e-7
    temp = 0.1

    self$p = torch_sigmoid(self$p_logit)
    u_noise = torch_rand_like(x)

    drop_prob = (torch_log(self$p + eps) -
                   torch_log(1 - self$p + eps) +
                   torch_log(u_noise + eps) -
                   torch_log(1 - u_noise + eps))

    drop_prob = torch_sigmoid(drop_prob / temp)

    random_tensor = 1 - drop_prob
    retain_prob = 1 - self$p

    x = torch_mul(x, random_tensor) / retain_prob

    return(x)
  }
)


nn_concrete_linear = nn_module(
  "ConcreteLinearLayer",
  initialize = function(n_dim_in,
                        n_dim_out,
                        weight_regularizer=1e-6,
                        dropout_regularizer=1e-5) {
    self$linear = nn_linear(n_dim_in, n_dim_out)
    self$conc_drop = nn_concrete_dropout(weight_regularizer=weight_regularizer,
                                         dropout_regularizer=dropout_regularizer)
    self$relu = nn_leaky_relu()
  },
  forward = function(x) {
    x = self$conc_drop(x, nn_sequential(self$linear, self$relu))
    self$regularization = self$conc_drop$regularization
    return(x)
  }
)


concrete_model = nn_module(
  classname = "ConcreteModel",
  # the initialize function tuns whenever we instantiate the model
  initialize = function(num_input_dim = 1,
                        num_hidden_dim = 1024,
                        num_output_dim = 1,
                        num_hidden_layers = 3,
                        weight_regularizer = 1e-6,
                        dropout_regularizer = 1e-5,
                        clamp = c(-1e25, 1e25)) {

    self$num_hidden_layers = num_hidden_layers

    self$concrete_dropout = nn_sequential()

    self$concrete_dropout$add_module("conc_drop1", nn_concrete_linear(num_input_dim,
                                                                      num_hidden_dim,
                                                                      weight_regularizer=weight_regularizer,
                                                                      dropout_regularizer=dropout_regularizer))

    for (i in 2:num_hidden_layers) {
      self$concrete_dropout$add_module(paste0("conc_drop", i), nn_concrete_linear(num_hidden_dim,
                                                                                  num_hidden_dim,
                                                                                  weight_regularizer=weight_regularizer,
                                                                                  dropout_regularizer=dropout_regularizer))
    }

    self$linear_mu = nn_linear(num_hidden_dim, num_output_dim)
    self$linear_logvar = nn_linear(num_hidden_dim, num_output_dim)
    self$conc_drop_mu = nn_concrete_dropout(weight_regularizer=weight_regularizer,
                                            dropout_regularizer=dropout_regularizer)
    self$conc_drop_logvar = nn_concrete_dropout(weight_regularizer=weight_regularizer,
                                                dropout_regularizer=dropout_regularizer)

    self$clamp = clamp
    # TODO
    self$p = NA
    # self$p = data.frame() # Monitor dropout rates at each iteration

  },

  # this function is called whenever we call our model on input.
  forward = function(x) {

    # Apply concrete dropout on hidden layers and two output layers (mu + log var)
    x1 = self$concrete_dropout(x)
    mean = self$conc_drop_mu(x1, self$linear_mu)
    log_var = self$conc_drop_logvar(x1, self$linear_logvar)
    # ensure that the variance does not become too small, which can lead to numerical instability
    log_var = torch_clamp(log_var,
                          min = sign(-1e25) * log(abs(self$clamp[1])),
                          max = log(self$clamp[2]))

    # Regularization terms
    conc_drop_layers = paste0("concrete_dropout.conc_drop", c(1:self$num_hidden_layers))
    regularization = torch_empty(length(conc_drop_layers) + 2, device=x$device)

    for (i in 1:length(conc_drop_layers)) {
      y = which(grepl(conc_drop_layers[i], self$modules))
      regularization[i] = self$modules[conc_drop_layers[i]][[1]]$regularization
    }

    regularization[length(conc_drop_layers) + 1] = self$conc_drop_mu$regularization
    regularization[length(conc_drop_layers) + 2] = self$conc_drop_logvar$regularization

    self$regularization = torch_sum(regularization)

    # TODO Monitor dropout rates
    params = self$named_parameters()
    p_logit = params[grepl("p_logit", names(params))]
    p = lapply(p_logit, function(x) torch_sigmoid(x))
    p = unlist(lapply(p, function(x) as.numeric(x)))
    self$p = p

    # return a concatenated tensor
    torch_stack(list(mean, log_var), dim = 1)
  },

  # Heteroscedastic loss function
  loss = function(preds, target) {
    # preds = ctx$model(input)

    mu = preds[1,,]
    log_var = preds[2,,]

    # add a small constant to the variance to prevent it from being zero
    precision = torch_exp(-log_var) + 1e-6
    # Logsumexp for numerical stability
    # precision = torch_logsumexp(-log_var, 1, keepdim = TRUE)

    # Must return a scalar - Do two times the sum when more than one parameter (sum of losses)
    # heteroscedastic_loss = torch_sum(torch_mean(torch_sum(precision * (target - mu)^2 + log_var, 1), 1))
    heteroscedastic_loss = torch_mean(torch_sum(precision * (target - mu)^2 + log_var, 1), 1)

    return(heteroscedastic_loss)
  }
)

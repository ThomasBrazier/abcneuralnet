# The base model for Deep Ensemble, a stack of linear layers
single_model = nn_module(
  "Model",
  initialize = function(num_input_dim = 1,
                        num_output_dim = 1,
                        num_hidden_layers = 3,
                        num_hidden_dim = 512,
                        clamp = c(1e-6, 1e6)) {

    self$num_hidden_layers = num_hidden_layers
    self$num_hidden_dim = num_hidden_dim

    self$mlp = nn_sequential(nn_linear(num_input_dim, num_hidden_dim),
                             nn_leaky_relu())

    for (i in 2:num_hidden_layers) {
      self$mlp$add_module(paste0("hidden_layer", i), nn_linear(num_hidden_dim,
                                                                    num_hidden_dim))
      # self$mlp$add_module(paste0("batch_norm", i), nn_batch_norm1d(num_hidden_dim))
      self$mlp$add_module(paste0("relu", i), nn_leaky_relu())
    }

    self$mu = nn_linear(num_hidden_dim, num_output_dim)
    self$sigma = nn_linear(num_hidden_dim, num_output_dim)

    self$clamp = clamp

  },

  forward = function(x) {
    x1 = self$mlp(x)
    mu = self$mu(x1)

    sig = self$sigma(x1)
    sig = torch_clamp(sig, min = self$clamp[1], max = self$clamp[2])  # For numerical stability

    # return a concatenated tensor
    torch_stack(list(mu, sig), dim = 1)
  }
)


# Custom dataset retunring an array of n = num_models input and target
ensemble_dataset = dataset(

  name = "ensemble_dataset",

  initialize = function(x, y, num_models, randomize = TRUE) {
    # x and y are inputs and targets to shuffle independently
    x_array = torch_empty(num_models, x$shape[1], x$shape[2])
    y_array = torch_empty(num_models, y$shape[1], y$shape[2])

    for (i in 1:num_models) {
      if (randomize) {
        idx = sample(1:x$shape[1], replace = FALSE)
        x_rand = x[idx,]$view(x$size())
        y_rand = y[idx,]$view(y$size())
      } else {
        x_rand = x
        y_rand = y
      }

      # print(x_rand)
      # print(y_rand)

      x_array[i,,] = x_rand$detach()
      y_array[i,,] = y_rand$detach()
    }

    # Input data x
    # self$x = torch_stack(x_list,
    # dim = 1)
    # print(x_array)
    self$x = x_array

    # Target data y
    # self$y = torch_stack(y_list,
    #                      dim = 1)
    # print(y_array)
    self$y = y_array

  },

  .getitem = function(i) {
    list(x = self$x[,i,], y = self$y[,i,])
  },

  .length = function() {
    self$y$size()[[2]]
  }

)




nn_ensemble = nn_module(
  classname = "DeepEnsemble",

  # Initialize function called whenever we instantiate the model
  initialize = function(model,
                        num_models = 5,
                        learning_rate = 0.001,
                        weight_decay = 1e-5,
                        num_input_dim = 1,
                        num_output_dim = 1,
                        num_hidden_dim = 128,
                        num_hidden_layers = 3,
                        epsilon = NULL,
                        clamp = c(1e-6, 1e6)) {
    # print("Init")
    self$num_models = num_models
    self$learning_rate = learning_rate
    self$weight_decay = weight_decay
    self$epsilon = epsilon

    # Initialize multiple models
    model_list = lapply(1:num_models, function(x) model(num_input_dim = num_input_dim,
                                                        num_output_dim = num_output_dim,
                                                        num_hidden_dim = num_hidden_dim,
                                                        num_hidden_layers = num_hidden_layers,
                                                        clamp = clamp))
    names(model_list) = seq(1, self$num_models)
    self$model_list = nn_module_list(model_list)
    # self$model_list = model()

    # Initialize optimizers
    opt_list = lapply(self$model_list, function(m) optim_adam(m$parameters,
                                                              lr = self$learning_rate,
                                                              weight_decay = self$weight_decay))
    names(opt_list) = seq(1, self$num_models)
    # opt_list = optim_adam(self$model_list$parameters, lr = self$learning_rate)
    self$optimizers = opt_list
  },

  # Set optimizers method
  set_optimizers = function() {
    # Return the stored optimizers
    self$optimizers
  },

  # Forward pass: Collect predictions from all models
  forward = function(input) {
    # print("forward")
    # Collect predictions from each model
    predictions = lapply(fitted$model$model_list, function(model) model(x_sample))
    predictions = torch_stack(predictions, dim = 4)  # Stack predictions along a new dimension

    # Compute the mean and variance of predictions
    # dim (2 i.e. mu + var, num samples, num parameters, num networks)
    mu = predictions[1,,,]
    sigma = predictions[2,,,]
    # mean_prediction = torch_mean(predictions[1,,,], dim = 3)  # Mean of means
    # variance_prediction = torch_mean(predictions[2,,,], dim = 3)  # Mean of variances
    mean_prediction = torch_mean(mu, dim = 3)  # Mean of means across networks
    variance_prediction = torch_sqrt(torch_mean(sigma, dim = 3) +
                                       torch_mean(torch_square(mu), dim = 3) -
                                       torch_square(mean_prediction))


    # TODO Correct variance using ensemble variance formula
    # variance_prediction = torch_sqrt(variance_prediction +
    #                                    torch_mean(torch_square(predictions[, 1, ]), dim = 3) -
    #                                    torch_square(mean_prediction))
    # variance_prediction = torch_log(1 + torch_exp(variance_prediction)) + 1e-6

    # TODO Check and compare
    # Get final test result
    # out_mu_final = torch_mean(out_mu, dim = 1)
    # out_sig_final = torch_sqrt(torch_mean(out_sig, dim = 1) + torch_mean(torch_square(out_mu), dim = 1) - torch_square(out_mu_final))

    return(list(mean = mean_prediction, variance = variance_prediction))
  },

  # Training step
  step = function() {
    # print("step")
    # Store loss for each model
    # print("Init loss")
    ctx$loss = list()

    # Iterate over each model and optimizer
    # print("Iterate")
    for (i in seq_along(self$model_list)) {

      if (ctx$training) {
        input = ctx$input[,i,]
        target = ctx$target[,i,]
      } else {
        # Test/eval all networks on the same batch
        input = ctx$input
        target = ctx$target
      }

      # print(input)
      # print(input$shape)

      opt_name = names(self$optimizers)[i]
      model = self$model_list[[i]]
      optimizer = self$optimizers[[opt_name]]

      # Forward pass and loss - Combine two losses if adversarial
      if (!is.null(self$epsilon) & ctx$training) {
        # Adversarial
        # Loss without perturbation to get the gradient sign
        input$requires_grad = TRUE
        preds = model(input)
        mean = preds[1]
        var = preds[2]
        # print("Loss for gradient sign")
        # print(mean$shape)
        # print(mean)
        # print(preds$shape)

        loss_for_adv = self$adversarial_nll_loss(mean, target, var)
        # print(loss_for_adv)

        # Gradient for Gaussian NLL loss
        grad = autograd_grad(loss_for_adv, input, retain_graph = FALSE)[[1]]

        batch_x = input$detach()
        batch_y = target$detach()
        # print(batch_x)
        # print(batch_y)

        mean = mean$detach()
        var = var$detach()
        loss_for_adv$detach_()
        # print(mean)
        # print(var)


        # print("Perturb input data")
        perturbed_data = self$fgsm_attack(batch_x, self$epsilon, grad)
        out_adv = model(perturbed_data)
        mean_adv = out_adv[1]
        var_adv = out_adv[2]
        # print(mean_adv)
        # print(var_adv)

        # print("Adversarial loss")
        loss = self$adversarial_nll_loss(mean, batch_y, var) + self$adversarial_nll_loss(mean_adv, batch_y, var_adv)
        # print(loss)

      } else {
        # Forward pass
        preds = model(input)
        # Compute loss
        loss = self$nll_loss(preds, target)
      }

      if (ctx$training) {
        # Zero gradients
        optimizer$zero_grad()
        # Backpropagation
        loss$backward()
        # Gradient clipping to avoid exploding gradients
        # nn_utils_clip_grad_norm_(model$parameters, max_norm = 1, norm_type = 2)
        # Update parameters
        optimizer$step()
      }

      # Detach loss to store it
      ctx$loss[[opt_name]] = loss$detach()
    }
  },

  # Negative Log-Likelihood loss for Gaussian predictions
  nll_loss = function(preds, target) {
    mu_train = preds[1]
    sig_train = preds[2]

    # !!! When sig_train is high (high variance in data), torch_exp returns Inf
    # sig_train_pos = torch_log(1 + torch_exp(sig_train)) + 1e-6
    # Returns the log of summed exponentials of each row of the input tensor in the given dimension dim.
    # The computation is numerically stabilized.
    # sig_train_pos = torch_logsumexp(sig_train, 1, keepdim = TRUE) + 1e-6
    sig_train_pos = log1pexp(sig_train) + 1e-6

    loss = torch_mean(0.5 * torch_log(sig_train_pos) + 0.5 * (torch_square(target - mu_train)/sig_train_pos)) + 1

    if (is.nan(loss$item())) {
      print(mu_train)
      print(target)
      # print(sig_train)
      # print(sig_train_pos)
      stop("Loss computation returned NaN. Check inputs!")
    }
    return(loss)
  },

  # Adversarial function
  fgsm_attack = function(input, epsilon, data_grad) {
    sign_data_grad = data_grad$sign()
    perturbed_input = input + epsilon * sign_data_grad
    return(perturbed_input)
  },

  # Adversarial loss
  adversarial_nll_loss = function(input, target, var) {
    # var_pos = torch_log(1 + torch_exp(var)) + 1e-6
    # var_pos = torch_logsumexp(var, 1, keepdim = TRUE) + 1e-6
    var_pos = log1pexp(var) + 1e-6

    loss = torch_mean(0.5 * torch_log(var_pos) + 0.5 * (torch_square(target - input)/var_pos)) + 1

    if (is.nan(loss$item())) {
      print(input)
      print(target)
      # print(var)
      # print(var_pos)
      stop("Loss computation returned NaN. Check inputs!")
    }

    return(loss)
  }
)

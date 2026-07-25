# TODO
# Heteroscedastic loss in mc dropout as in https://shrmtmt.medium.com/beyond-average-predictions-embracing-variability-with-heteroscedastic-loss-in-deep-learning-f098244cad6f

# Heteroscedastic loss = ((mean prediction - true value)^2 / 2*variance of prediction) + log(variance of prediction)

# Custom MC dropout layer is the same as nn_mc_dropout

# Model with Dense layers and MC dropout + heteroscedastic loss
gaussian_mc_model = torch::nn_module(
  "GaussianMCDropout",
  initialize = function(num_input_dim = 1,
                        num_hidden_dim = 1024,
                        num_output_dim = 1,
                        num_hidden_layers = 3,
                        dropout_hidden = 0.5,
                        clamp = c(-1e25, 1e25)) {

    self$num_hidden_layers = num_hidden_layers

    # Set a minimal model with a single layer and dropout on inputs (facultative)
    self$gaussian_mc_dropout = torch::nn_sequential(
      torch::nn_linear(num_input_dim, num_hidden_dim),
      nn_mc_dropout(p = dropout_hidden),
      torch::nn_leaky_relu())

    # `seq_len()` so that a single hidden layer gives an empty loop.
    # `2:num_hidden_layers` would count down to c(2, 1) and add a second layer.
    for (i in seq_len(num_hidden_layers - 1) + 1) {
      self$gaussian_mc_dropout$add_module(paste0("linear_", i), torch::nn_linear(num_hidden_dim, num_hidden_dim))
      self$gaussian_mc_dropout$add_module(paste0("dropout_", i), nn_mc_dropout(p = dropout_hidden))
      self$gaussian_mc_dropout$add_module(paste0("relu_", i), torch::nn_leaky_relu())
    }

    # Add output layers
    self$linear_mu = torch::nn_linear(num_hidden_dim, num_output_dim)
    self$linear_logvar = torch::nn_linear(num_hidden_dim, num_output_dim)
    
    self$clamp = clamp
  },
  
  # this function is called whenever we call our model on input.
  forward = function(x) {
    x1 = self$gaussian_mc_dropout(x)
    
    # Two output layers (mu + log var)
    mean = self$linear_mu(x1)
    log_var = self$linear_logvar(x1)
    
    # Ensure that the variance does not become too small, which can lead to numerical instability
    # Signed log of the lower bound, so that a bound given on either side of
    # zero maps to the log variance scale. `sign()` must be taken on the bound
    # itself: a hardcoded negative sign turns any positive lower bound into
    # `min = max`, which pins the log variance to a constant.
    log_var = torch::torch_clamp(log_var,
                                 min = sign(self$clamp[1]) * log(abs(self$clamp[1])),
                                 max = log(self$clamp[2]))
    
    # return a concatenated tensor
    torch::torch_stack(list(mean, log_var), dim = 1)
  },
  
  # Heteroscedastic loss function
  loss = function(preds, target) {
    mu = preds[1,,]
    log_var = preds[2,,]
    
    # Add a small constant to the variance to prevent it from being zero
    precision = torch::torch_exp(-log_var) + 1e-6

    # Must return a scalar - Do two times the sum when more than one parameter (sum of losses)
    heteroscedastic_loss = torch::torch_mean(torch::torch_sum(precision * (target - mu)^2 + log_var, 1), 1)
    
    return(heteroscedastic_loss)
  }
)



build_gaussian_mcdropout_model = function(optimizer = optim_adam,
                                 loss = nn_mse_loss(),
                                 input_dim = 1,
                                 num_hidden_dim = 1024,
                                 output_dim = 1,
                                 num_hidden_layers = 3,
                                 dropout = 0.5,
                                 learning_rate = 0.001,
                                 L2_weigth_decay = 1e-5) {
  model = gaussian_mc_model %>%
    luz::setup(optimizer = optimizer,
          loss = loss) %>%
    set_hparams(num_input_dim = input_dim,
                num_hidden_dim = num_hidden_dim,
                num_output_dim = output_dim,
                num_hidden_layers = num_hidden_layers,
                dropout_hidden = dropout) %>%
    set_opt_hparams(lr = learning_rate, weight_decay = L2_weigth_decay)

  return(model)
}

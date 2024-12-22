
library(torch)
library(luz)

# Custom MC dropout layer
nn_mc_dropout = nn_module(
  "nn_mc_dropout",
  initialize = function(p = 0.5, inplace = FALSE) {
    if (p < 0 || p > 1) {
      value_error("dropout probability has to be between 0 and 1 but got {p}")
    }
    self$mc_dropout = nn_dropout(p = p, inplace = inplace)
    self$p = p
    self$inplace = inplace
  },
  forward = function(x) {
    self$mc_dropout$train()
    self$mc_dropout(x)
  }
)


# Model with Dense layers and MC dropout
mc_dropout_model = nn_module(
  "MCDropout",
  initialize = function(num_input_dim = 1,
                        num_hidden_dim = 1024,
                        num_output_dim = 1,
                        num_hidden_layers = 3,
                        dropout_hidden = 0.5,
                        dropout_input = 0.2) {
    # Set a minimal model with a single layer and dropout on inputs (facultative)
    self$mc_dropout = nn_sequential(
      nn_dropout(p = dropout_input),
      nn_linear(num_input_dim, num_hidden_dim),
      nn_mc_dropout(p = dropout_hidden),
      nn_relu())

    for (i in 2:num_hidden_layers) {
      self$mc_dropout$add_module(paste0("linear_", i), nn_linear(num_hidden_dim, num_hidden_dim))
      self$mc_dropout$add_module(paste0("dropout_", i), nn_mc_dropout(p = dropout_hidden))
      self$mc_dropout$add_module(paste0("relu_", i), nn_relu())
    }

    # Add output layer
    self$mc_dropout$add_module(paste0("output"), nn_linear(num_hidden_dim, num_output_dim))
  },
  forward = function(x) {
    self$mc_dropout(x)
  }
)



build_mcdropout_model = function(optimizer = optim_adam,
                                 loss = nn_mse_loss(),
                                 input_dim = 1,
                                 num_hidden_dim = 1024,
                                 output_dim = 1,
                                 num_hidden_layers = 3,
                                 dropout = 0.5,
                                 dropout_input = 0,
                                 learning_rate = 0.001,
                                 L2_weigth_decay = 1e-5) {
  model = mc_dropout_model %>%
    setup(optimizer = optimizer,
          loss = loss) %>%
    set_hparams(num_input_dim = input_dim,
                num_hidden_dim = num_hidden_dim,
                num_output_dim = output_dim,
                num_hidden_layers = num_hidden_layers,
                dropout_hidden = dropout,
                dropout_input = dropout_input) %>%
    set_opt_hparams(lr = learning_rate, weight_decay = L2_weigth_decay)

  return(model)
}

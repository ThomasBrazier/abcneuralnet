test_that("save_abcnn() and load_abcnn() functions work", {

  # Load test data
  df = readRDS("inst/extdata/test_data.Rds")

  theta = df$train_y
  sumstats = df$train_x
  observed = df$observed_y

  methods = c("monte carlo dropout",
              "concrete dropout",
              "deep ensemble")

  for (m in methods) {
    # Init an abcnn object with inputs and targets
    abc = abcnn$new(theta,
                    sumstats,
                    observed,
                    method = m,
                    scale_input = "none",
                    scale_target = "none",
                    num_hidden_layers = 3,
                    num_hidden_dim = 128,
                    epochs = 10,
                    batch_size = 32)

    abc$fit()
    abc$predict()

    save_abcnn(abc, prefix = "tests/data/abc_test")

    # Is it possible to fit and predict again?
    abc = load_abcnn(prefix = "tests/data/abc_test")
    abc$fit()

    abc = load_abcnn(prefix = "tests/data/abc_test")
    abc$predict()

    # Other methods work?
    abc = load_abcnn(prefix = "tests/data/abc_test")
    abc$plot_training()
    abc$plot_prediction()
    abc$plot_posterior()
    abc$summary()
  }

})

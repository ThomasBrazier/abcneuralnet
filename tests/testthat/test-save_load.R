test_that("save_abcnn() and load_abcnn() functions work", {
  # Load test data
  # TODO Make tests data ad hoc
  df = readRDS("../../tests/data/test_data.Rds")

  theta = df$train_y
  sumstats = df$train_x
  observed = df$observed_y

  methods = c("monte carlo dropout",
              "concrete dropout",
              "deep ensemble")
  # TODO tabnet method

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
                    epochs = 3,
                    batch_size = 32)

    abc$fit()
    abc$predict()

    save_abcnn(abc, prefix = "../../tests/data/abc_test")

    # Is it possible to fit and predict again?
    abc = load_abcnn(prefix = "../../tests/data/abc_test")
    abc$fit()

    abc = load_abcnn(prefix = "../../tests/data/abc_test")
    abc$predict()

    # Other methods work?
    abc = load_abcnn(prefix = "../../tests/data/abc_test")
    abc$plot_training()
    abc$plot_prediction()
    abc$plot_posterior()
    abc$summary()
  }

})


# TODO
# test_that("save_abcnn handles TabNet-ABC method specifically", {
#   set.seed(123)
#   n_samples = 100  # TabNet needs more data
#   theta_training = data.frame(param1 = runif(n_samples, 0, 10))
#   sumstats_training = data.frame(
#     stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5),
#     stat2 = rnorm(n_samples, 0, 1)
#   )
#   sumstats_observed = data.frame(stat1 = 5.2, stat2 = 0.1)
#   
#   abc = abcnn$new(
#     theta_training, sumstats_training, sumstats_observed,
#     method = "tabnet-abc", epochs = 2, verbose = FALSE
#   )
#   abc$fit()
#   
#   # Save TabNet model
#   temp_prefix = tempfile("tabnet_test_")
#   expect_no_error(save_abcnn(abc, prefix = temp_prefix))
#   
#   # Check specific TabNet files
#   expect_true(file.exists(paste0(temp_prefix, "_abcnn.Rds")))
#   expect_true(file.exists(paste0(temp_prefix, "_model.Rds")))
#   expect_true(file.exists(paste0(temp_prefix, "_fitted.Rds")))
#   expect_true(file.exists(paste0(temp_prefix, "_torch.Rds")))
#   
#   # Load TabNet model
#   abc_loaded = load_abcnn(prefix = temp_prefix)
#   expect_s3_class(abc_loaded, "abcnn")
#   expect_equal(abc_loaded$method, "tabnet-abc")
#   
#   # Clean up
#   unlink(paste0(temp_prefix, "*"))
# })


# TODO
# test_that("save_abcnn and load_abcnn handle errors gracefully", {
#   set.seed(123)
#   n_samples = 50
#   theta_training = data.frame(param1 = runif(n_samples, 0, 10))
#   sumstats_training = data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5))
#   sumstats_observed = data.frame(stat1 = 5.2)
#   
#   abc = abcnn$new(
#     theta_training, sumstats_training, sumstats_observed,
#     method = "monte carlo dropout", epochs = 1, verbose = FALSE
#   )
#   
#   # Test saving without fitted model
#   expect_error(
#     save_abcnn(abc, prefix = tempfile()),
#     "must have a fitted model"
#   )
#   
#   # Test loading non-existent files
#   expect_error(
#     load_abcnn(prefix = "non_existent_prefix"),
#     "No such file or directory"
#   )
#   
#   # Test loading incomplete files
#   temp_prefix = tempfile("incomplete_test_")
#   saveRDS(abc, paste0(temp_prefix, "_abcnn.Rds"))
#   # Missing other files
#   
#   expect_error(
#     load_abcnn(prefix = temp_prefix),
#     "Missing required files"
#   )
#   
#   # Clean up
#   unlink(paste0(temp_prefix, "*"))
# })

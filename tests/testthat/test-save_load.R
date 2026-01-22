test_that("save_abcnn() and load_abcnn() functions work", {
  # Make test data
  set.seed(123)
  n_samples = 10000
  theta_training = data.frame(param1 = runif(n_samples, 0, 1))
  sumstats_training = data.frame(
    stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.05),
    stat2 = theta_training$param1^2 + rnorm(n_samples, 0, 0.1)
  )
  sumstats_observed = data.frame(stat1 = c(0.4, 0.6), stat2 = c(0.2, 0.7))
  
  num_posterior_samples = 50 

  methods = c("monte carlo dropout",
              "concrete dropout",
              "deep ensemble",
              "tabnet-abc")
  
  for (m in methods) {
    # Init an abcnn object with inputs and targets
    abc = abcnn$new(theta_training,
                    sumstats_training,
                    sumstats_observed,
                    method = m,
                    scale_input = "none",
                    scale_target = "none",
                    num_hidden_layers = 3,
                    num_hidden_dim = 128,
                    epochs = 3,
                    batch_size = 32,
                    tol = 0.1,
                    num_posterior_samples = num_posterior_samples,
                    abc_method = "loclinear")

    abc$fit()
    abc$predict()

    save_abcnn(abc, prefix = "../../tests/data/abc_test")

    # Is it possible to fit and predict again?
    abc = load_abcnn(prefix = "../../tests/data/abc_test")
    expect_no_error(abc$fit())

    abc = load_abcnn(prefix = "../../tests/data/abc_test")
    expect_no_error(abc$predict())

    # Other methods work?
    abc = load_abcnn(prefix = "../../tests/data/abc_test")
    expect_no_error(abc$plot_training())
    expect_no_error(abc$plot_prediction())
    expect_no_error(abc$plot_posterior())
    expect_no_error(abc$summary())
  }

})



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

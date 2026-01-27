# Test abcnn class initialization and basic functionality
test_that("abcnn class initializes correctly", {
  # Create simple test data
  n_samples = 10000
  theta_training = data.frame(param1 = runif(n_samples, 0, 1))
  sumstats_training = data.frame(
    stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.05),
    stat2 = theta_training$param1^2 + rnorm(n_samples, 0, 0.1)
  )
  sumstats_observed = data.frame(stat1 = 0.4, stat2 = 0.2)
  
  # Test initialization with different methods
  methods = c("monte carlo dropout", "concrete dropout", "deep ensemble", "tabnet-abc")
  
  for (method in methods) {
    expect_no_error({
      abc = abcnn$new(
        theta_training,
        sumstats_training, 
        sumstats_observed,
        method = method,
        epochs = 1,
        verbose = FALSE,
        tol = 0.1
      )
    }, message = paste("Failed to initialize with method:", method))
    
    # Check that object is created correctly
    abc = abcnn$new(
      theta_training,
      sumstats_training,
      sumstats_observed,
      method = method,
      epochs = 1,
      verbose = FALSE
    )
    
    expect_r6_class(abc, "abcnn")
    expect_equal(abc$method, method)
    expect_equal(dim(abc$theta), c(n_samples, 1))
    expect_equal(dim(abc$sumstat), c(n_samples, 2))
    expect_equal(dim(abc$observed), c(1, 2))
  }
})

test_that("abcnn parameter validation works", {
  n_samples = 10000
  theta_training = data.frame(param1 = runif(n_samples, 0, 1))
  sumstats_training = data.frame(
    stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.05),
    stat2 = theta_training$param1^2 + rnorm(n_samples, 0, 0.1)
  )
  sumstats_observed = data.frame(stat1 = 0.4, stat2 = 0.2)
  methods = c("monte carlo dropout", "concrete dropout", "deep ensemble", "tabnet-abc")
  
  # Test invalid method
  expect_error(
    abcnn$new(theta_training, sumstats_training, sumstats_observed, method = "invalid"),
    paste("Method must be one of:", paste(methods, collapse = ","))
  )
})



test_that("abcnn plotting methods work", {
  set.seed(123)
  n_samples = 10000
  theta_training = data.frame(param1 = runif(n_samples, 0, 1))
  sumstats_training = data.frame(
    stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.05),
    stat2 = theta_training$param1^2 + rnorm(n_samples, 0, 0.1)
  )
  sumstats_observed = data.frame(stat1 = 0.4, stat2 = 0.2)
  
  methods = c("monte carlo dropout", "concrete dropout", "deep ensemble", "tabnet-abc")
  
  for (method in methods) {
    abc = abcnn$new(
      theta_training,
      sumstats_training,
      sumstats_observed,
      method = method,
      epochs = 3,
      verbose = FALSE,
      tol = 0.1
    )
    
    abc$fit()
    abc$predict()
    
    # Test plotting methods (they should not error)
    expect_no_error(abc$plot_training())
    expect_no_error(abc$plot_prediction())
    if (method != "tabnet-abc") expect_no_error(abc$plot_posterior())
  }
})



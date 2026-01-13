# Test abcnn class initialization and basic functionality
test_that("abcnn class initializes correctly", {
  # Create simple test data
  n_samples <- 100
  theta_training <- data.frame(param1 = rnorm(n_samples))
  sumstats_training <- data.frame(stat1 = rnorm(n_samples), stat2 = rnorm(n_samples))
  sumstats_observed <- data.frame(stat1 = 0.5, stat2 = -0.3)
  
  # Test initialization with different methods
  methods <- c("monte carlo dropout", "concrete dropout", "deep ensemble", "tabnet-abc")
  
  for (method in methods) {
    expect_no_error({
      abc <- abcnn$new(
        theta_training,
        sumstats_training, 
        sumstats_observed,
        method = method,
        epochs = 1,
        verbose = FALSE
      )
    }, msg = paste("Failed to initialize with method:", method))
    
    # Check that object is created correctly
    abc <- abcnn$new(
      theta_training,
      sumstats_training,
      sumstats_observed,
      method = method,
      epochs = 1,
      verbose = FALSE
    )
    
    expect_s3_class(abc, "abcnn")
    expect_equal(abc$method, method)
    expect_equal(dim(abc$theta), c(n_samples, 1))
    expect_equal(dim(abc$sumstat), c(n_samples, 2))
    expect_equal(dim(abc$observed), c(1, 2))
  }
})

test_that("abcnn parameter validation works", {
  n_samples <- 100
  theta_training <- data.frame(param1 = rnorm(n_samples))
  sumstats_training <- data.frame(stat1 = rnorm(n_samples))
  sumstats_observed <- data.frame(stat1 = 0.5)
  
  # Test invalid method
  expect_error(
    abcnn$new(theta_training, sumstats_training, sumstats_observed, method = "invalid"),
    "method must be one of"
  )
  
  # Test mismatched dimensions
  theta_wrong <- data.frame(param1 = rnorm(n_samples + 10))
  expect_error(
    abcnn$new(theta_wrong, sumstats_training, sumstats_observed),
    "must have the same number of rows"
  )
  
  # Test invalid dropout rate
  expect_error(
    abcnn$new(
      theta_training, sumstats_training, sumstats_observed,
      method = "monte carlo dropout", dropout = 0.8
    ),
    "dropout must be between"
  )
  
  # Test invalid scaling method
  expect_error(
    abcnn$new(
      theta_training, sumstats_training, sumstats_observed,
      scale_input = "invalid"
    ),
    "scale_input must be one of"
  )
})

test_that("abcnn fit and predict workflow works", {
  # Create simple test data
  set.seed(123)
  n_samples <- 50
  theta_training <- data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(
    stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5),
    stat2 = theta_training$param1^2 + rnorm(n_samples, 0, 1)
  )
  sumstats_observed <- data.frame(stat1 = 5.2, stat2 = 28.5)
  
  # Test with monte carlo dropout (fastest for testing)
  abc <- abcnn$new(
    theta_training,
    sumstats_training,
    sumstats_observed,
    method = "monte carlo dropout",
    epochs = 2,
    batch_size = 16,
    verbose = FALSE
  )
  
  # Test fitting
  expect_no_error(abc$fit())
  expect_true(abc$trained)
  
  # Test prediction
  expect_no_error(abc$predict())
  expect_equal(dim(abc$predictive_mean), c(1, 1))
  expect_equal(dim(abc$predictive_variance), c(1, 1))
  
  # Test that predictions are reasonable
  expect_true(is.finite(abc$predictive_mean[1, 1]))
  expect_true(is.finite(abc$predictive_variance[1, 1]))
  expect_true(abc$predictive_variance[1, 1] >= 0)
})

test_that("abcnn posterior sampling works", {
  set.seed(123)
  n_samples <- 50
  theta_training <- data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(
    stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5)
  )
  sumstats_observed <- data.frame(stat1 = 5.2)
  
  # Test with concrete dropout
  abc <- abcnn$new(
    theta_training,
    sumstats_training,
    sumstats_observed,
    method = "concrete dropout",
    epochs = 2,
    num_posterior_samples = 100,
    verbose = FALSE
  )
  
  abc$fit()
  abc$predict()
  
  # Test posterior sampling
  expect_no_error(abc$posterior())
  expect_equal(dim(abc$posterior_samples), c(100, 1))
  expect_true(all(is.finite(abc$posterior_samples)))
  
  # Test credible intervals
  expect_no_error(abc$credible_interval())
  expect_equal(length(abc$credible_interval), 2)
  expect_true(abc$credible_interval[1] <= abc$credible_interval[2])
})

test_that("abcnn plotting methods work", {
  set.seed(123)
  n_samples <- 50
  theta_training <- data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(
    stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5)
  )
  sumstats_observed <- data.frame(stat1 = 5.2)
  
  abc <- abcnn$new(
    theta_training,
    sumstats_training,
    sumstats_observed,
    method = "monte carlo dropout",
    epochs = 2,
    verbose = FALSE
  )
  
  abc$fit()
  abc$predict()
  
  # Test plotting methods (they should not error)
  expect_no_error(abc$plot_training())
  expect_no_error(abc$plot_prediction())
  expect_no_error(abc$plot_posterior())
  
  # Test summary
  expect_no_error(abc$summary())
  expect_s3_class(abc$summary(), "data.frame")
})

test_that("abcnn data scaling works correctly", {
  set.seed(123)
  n_samples <- 50
  theta_training <- data.frame(param1 = runif(n_samples, 10, 100))
  sumstats_training <- data.frame(stat1 = runif(n_samples, -50, 50))
  sumstats_observed <- data.frame(stat1 = 0)
  
  # Test different scaling methods
  scaling_methods <- c("none", "minmax", "robustscaler")
  
  for (scale_method in scaling_methods) {
    abc <- abcnn$new(
      theta_training,
      sumstats_training,
      sumstats_observed,
      method = "monte carlo dropout",
      scale_input = scale_method,
      scale_target = scale_method,
      epochs = 1,
      verbose = FALSE
    )
    
    abc$fit()
    abc$predict()
    
    # Check that scaling was applied
    expect_true(is.finite(abc$predictive_mean[1, 1]))
  }
})

test_that("abcnn handles edge cases", {
  # Test with minimal data
  theta_small <- data.frame(param1 = c(1, 2, 3))
  sumstats_small <- data.frame(stat1 = c(1, 2, 3))
  observed_small <- data.frame(stat1 = 2)
  
  expect_warning(
    {
      abc <- abcnn$new(
        theta_small, sumstats_small, observed_small,
        method = "monte carlo dropout",
        epochs = 1,
        verbose = FALSE
      )
      abc$fit()
    },
    "small dataset"
  )
  
  # Test with single observation
  theta_single <- data.frame(param1 = 5)
  sumstats_single <- data.frame(stat1 = 5)
  observed_single <- data.frame(stat1 = 5)
  
    expect_error(
    abcnn$new(theta_single, sumstats_single, observed_single),
    "require at least"
  )
}
)
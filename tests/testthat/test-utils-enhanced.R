# Enhanced tests for utility functions
test_that("scaler function handles all methods correctly", {
  # Test data
  x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  
  # Summary statistics
  summary_stats <- list(
    min = min(x),
    max = max(x),
    mean = mean(x),
    sd = sd(x),
    quantile_25 = quantile(x, 0.25),
    quantile_75 = quantile(x, 0.75)
  )
  
  # Test each scaling method
  methods <- c("none", "minmax", "robustscaler", "normalization")
  
  for (method in methods) {
    # Forward transformation
    scaled <- scaler(x, summary_stats, method = method, type = "forward")
    expect_true(is.numeric(scaled$x))
    expect_true(all(is.finite(scaled$x)))
    
    # Backward transformation should recover original
    if (method != "none") {
      original <- scaler(scaled, summary_stats, method = method, type = "backward")
      expect_true(all(abs(original$x - x) < 1e-10))
    }
  }
})


test_that("abcnn data scaling works correctly", {
  set.seed(123)
  n_samples = 10000
  theta_training = data.frame(param1 = runif(n_samples, 0, 1))
  sumstats_training = data.frame(
    stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.05),
    stat2 = theta_training$param1^2 + rnorm(n_samples, 0, 0.1)
  )
  sumstats_observed = data.frame(stat1 = 0.4, stat2 = 0.2)
  
  # Test different scaling methods
  scaling_methods = c("none", "minmax", "robustscaler")
  
  for (scale_method in scaling_methods) {
    abc = abcnn$new(
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
    
    # Check that proper scaling was applied
    # TODO
  }
})


test_that("data_summary function works correctly", {
  # Test with vector
  x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  summary <- data_summary(x)
  
  expect_true(is.list(summary))
  expect_true(all(c("min", "max", "mean", "sd", "quantile_25", "quantile_75") %in% names(summary)))
  expect_equal(summary$min, 1)
  expect_equal(summary$max, 10)
  expect_equal(summary$mean, 5.5)
  
  # Test with data frame
  df <- data.frame(a = 1:5, b = 6:10)
  summary_df <- data_summary(df)
  
  expect_true(is.list(summary_df))
  expect_equal(length(summary_df), 6)  # 6 statistics
  expect_equal(length(summary_df$min), 2)  # 2 columns
})

test_that("save_abcnn and load_abcnn work for all methods", {
  set.seed(123)
  n_samples <- 50
  theta_training <- data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5))
  sumstats_observed <- data.frame(stat1 = 5.2)
  
  methods <- c("monte carlo dropout", "concrete dropout", "deep ensemble")
  
  for (method in methods) {
    # Create and train model
    abc <- abcnn$new(
      theta_training, sumstats_training, sumstats_observed,
      method = method, epochs = 2, verbose = FALSE
    )
    abc$fit()
    abc$predict()
    
    # Save model
    temp_prefix <- tempfile("abc_test_")
    expect_no_error(save_abcnn(abc, prefix = temp_prefix))
    
    # Check that files were created
    expect_true(file.exists(paste0(temp_prefix, "_abcnn.Rds")))
    expect_true(file.exists(paste0(temp_prefix, "_model.Rds")))
    
    if (method != "tabnet-abc") {
      expect_true(file.exists(paste0(temp_prefix, "_luz.Rds")))
    } else {
      expect_true(file.exists(paste0(temp_prefix, "_fitted.Rds")))
    }
    
    # Load model
    abc_loaded <- load_abcnn(prefix = temp_prefix)
    
    # Check that loaded model has same properties
    expect_s3_class(abc_loaded, "abcnn")
    expect_equal(abc_loaded$method, method)
    expect_equal(dim(abc_loaded$theta), dim(abc$theta))
    expect_equal(dim(abc_loaded$sumstat), dim(abc$sumstat))
    
    # Test that loaded model can predict
    expect_no_error(abc_loaded$predict())
    expect_equal(dim(abc_loaded$predictive_mean), dim(abc$predictive_mean))
    
    # Test that loaded model can be fitted again
    expect_no_error(abc_loaded$fit())
    
    # Clean up
    unlink(paste0(temp_prefix, "*"))
  }
})



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

test_that("scaler handles edge cases", {
  # Test with constant data
  x_constant <- rep(5, 10)
  summary_constant <- list(
    min = 5, max = 5, mean = 5, sd = 0,
    quantile_25 = 5, quantile_75 = 5
  )
  
  # Should handle gracefully
  expect_no_error({
    scaled <- scaler(x_constant, summary_constant, method = "minmax", type = "forward")
  })
  
  # Test with NA values
  x_na <- c(1, 2, NA, 4, 5)
  summary_na <- list(
    min = 1, max = 5, mean = 3, sd = 1.58,
    quantile_25 = 1.75, quantile_75 = 4.25
  )
  
  expect_warning(
    scaler(x_na, summary_na, method = "minmax", type = "forward"),
    "NA values"
  )
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

test_that("save_abcnn handles TabNet-ABC method specifically", {
  set.seed(123)
  n_samples <- 100  # TabNet needs more data
  theta_training <- data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(
    stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5),
    stat2 = rnorm(n_samples, 0, 1)
  )
  sumstats_observed <- data.frame(stat1 = 5.2, stat2 = 0.1)
  
  abc <- abcnn$new(
    theta_training, sumstats_training, sumstats_observed,
    method = "tabnet-abc", epochs = 2, verbose = FALSE
  )
  abc$fit()
  
  # Save TabNet model
  temp_prefix <- tempfile("tabnet_test_")
  expect_no_error(save_abcnn(abc, prefix = temp_prefix))
  
  # Check specific TabNet files
  expect_true(file.exists(paste0(temp_prefix, "_abcnn.Rds")))
  expect_true(file.exists(paste0(temp_prefix, "_model.Rds")))
  expect_true(file.exists(paste0(temp_prefix, "_fitted.Rds")))
  expect_true(file.exists(paste0(temp_prefix, "_torch.Rds")))
  
  # Load TabNet model
  abc_loaded <- load_abcnn(prefix = temp_prefix)
  expect_s3_class(abc_loaded, "abcnn")
  expect_equal(abc_loaded$method, "tabnet-abc")
  
  # Clean up
  unlink(paste0(temp_prefix, "*"))
})

test_that("save_abcnn and load_abcnn handle errors gracefully", {
  set.seed(123)
  n_samples <- 50
  theta_training <- data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5))
  sumstats_observed <- data.frame(stat1 = 5.2)
  
  abc <- abcnn$new(
    theta_training, sumstats_training, sumstats_observed,
    method = "monte carlo dropout", epochs = 1, verbose = FALSE
  )
  
  # Test saving without fitted model
  expect_error(
    save_abcnn(abc, prefix = tempfile()),
    "must have a fitted model"
  )
  
  # Test loading non-existent files
  expect_error(
    load_abcnn(prefix = "non_existent_prefix"),
    "No such file or directory"
  )
  
  # Test loading incomplete files
  temp_prefix <- tempfile("incomplete_test_")
  saveRDS(abc, paste0(temp_prefix, "_abcnn.Rds"))
  # Missing other files
  
  expect_error(
    load_abcnn(prefix = temp_prefix),
    "Missing required files"
  )
  
  # Clean up
  unlink(paste0(temp_prefix, "*"))
})

test_that("utility functions handle different data types", {
  # Test with matrix
  x_matrix <- matrix(1:10, nrow = 5, ncol = 2)
  summary_matrix <- data_summary(x_matrix)
  
  expect_true(is.list(summary_matrix))
  expect_equal(length(summary_matrix$min), 2)
  
  # Test with tibble
  x_tibble <- tibble::tibble(a = 1:5, b = 6:10)
  summary_tibble <- data_summary(x_tibble)
  
  expect_true(is.list(summary_tibble))
  expect_equal(length(summary_tibble$min), 2)
  
  # Test scaling with different data types
  for (x_data in list(x_matrix, x_tibble)) {
    summary <- data_summary(x_data)
    expect_no_error({
      scaled <- scaler(x_data, summary, method = "minmax", type = "forward")
    })
  }
})

test_that("conformal prediction utilities work", {
  set.seed(123)
  n_samples <- 100
  theta_training <- data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5))
  sumstats_observed <- data.frame(stat1 = 5.2)
  
  # Test with methods that support conformal prediction
  methods_conformal <- c("concrete dropout", "deep ensemble")
  
  for (method in methods_conformal) {
    abc <- abcnn$new(
      theta_training, sumstats_training, sumstats_observed,
      method = method, epochs = 2, num_conformal = 50, verbose = FALSE
    )
    abc$fit()
    abc$predict()
    
    # Test conformal prediction
    expect_no_error(abc$conformal_prediction())
    expect_true("conformal_bounds" %in% names(abc))
    expect_equal(length(abc$conformal_bounds), 2)
    expect_true(abc$conformal_bounds[1] <= abc$conformal_bounds[2])
    
    # Test that conformal bounds are reasonable
    expect_true(all(is.finite(abc$conformal_bounds)))
  }
})

test_that("ABC sampling utilities work", {
  set.seed(123)
  n_samples <- 200
  theta_training <- data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(
    stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5),
    stat2 = rnorm(n_samples, 0, 1)
  )
  sumstats_observed <- data.frame(stat1 = 5.2, stat2 = 0.1)
  
  abc <- abcnn$new(
    theta_training, sumstats_training, sumstats_observed,
    method = "tabnet-abc", epochs = 2, verbose = FALSE
  )
  abc$fit()
  
  # Test different ABC methods
  abc_methods <- c("rejection", "loclinear", "neuralnet", "ridge")
  
  for (abc_method in abc_methods) {
    abc$abc_method <- abc_method
    
    expect_no_error(abc$abc_inference())
    expect_true(abc$abc_performed)
    expect_true("abc_posterior" %in% names(abc))
    
    # Test that posterior samples are reasonable
    expect_true(all(is.finite(abc$abc_posterior)))
    expect_true(nrow(abc$abc_posterior) > 0)
  }
  
  # Test different sampling methods
  sampling_methods <- c("rejection", "importance")
  
  for (sampling_method in sampling_methods) {
    abc$sampling <- sampling_method
    
    expect_no_error(abc$abc_inference())
    expect_true(abc$abc_performed)
  }
  
  # Test kernel functions
  kernels <- c("rbf", "epanechnikov")
  
  for (kernel in kernels) {
    abc$kernel <- kernel
    
    expect_no_error(abc$abc_inference())
    expect_true(abc$abc_performed)
  }
}
)
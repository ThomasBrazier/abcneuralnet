# Comprehensive error handling and edge case tests
test_that("abcnn handles invalid inputs gracefully", {
  # Test with NULL inputs
  expect_error(
    abcnn$new(NULL, NULL, NULL),
    "must be data frames"
  )
  
  # Test with empty data frames
  expect_error(
    abcnn$new(data.frame(), data.frame(), data.frame()),
    "require at least"
  )
  
  # Test with mismatched row counts
  theta_mismatch <- data.frame(param1 = 1:10)
  sumstats_mismatch <- data.frame(stat1 = 1:5)  # Different number of rows
  observed_mismatch <- data.frame(stat1 = 2.5)
  
  expect_error(
    abcnn$new(theta_mismatch, sumstats_mismatch, observed_mismatch),
    "must have the same number of rows"
  )
  
  # Test with non-numeric data
  theta_char <- data.frame(param1 = letters[1:10])
  sumstats_numeric <- data.frame(stat1 = 1:10)
  observed_numeric <- data.frame(stat1 = 5)
  
  expect_error(
    abcnn$new(theta_char, sumstats_numeric, observed_numeric),
    "must contain only numeric"
  )
  
  # Test with NA values in critical positions
  theta_na <- data.frame(param1 = c(1, 2, NA, 4, 5))
  sumstats_clean <- data.frame(stat1 = 1:5)
  observed_clean <- data.frame(stat1 = 3)
  
  expect_warning(
    abcnn$new(theta_na, sumstats_clean, observed_clean),
    "contains NA values"
  )
})


test_that("abcnn handles edge cases", {
  # Test with minimal data
  theta_small = data.frame(param1 = c(1, 2, 3))
  sumstats_small = data.frame(stat1 = c(1, 2, 3))
  observed_small = data.frame(stat1 = 2)
  
  # TODO
  
  # Test with single observation
  # TODO
}
)


test_that("abcnn handles extreme parameter values", {
  n_samples <- 50
  theta_training <- data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5))
  sumstats_observed <- data.frame(stat1 = 5.2)
  
  # Test extreme dropout rates
  expect_error(
    abcnn$new(
      theta_training, sumstats_training, sumstats_observed,
      method = "monte carlo dropout", dropout = 1.0
    ),
    "dropout must be between"
  )
  
  expect_error(
    abcnn$new(
      theta_training, sumstats_training, sumstats_observed,
      method = "monte carlo dropout", dropout = 0.0
    ),
    "dropout must be between"
  )
  
  # Test extreme learning rates
  expect_warning(
    abcnn$new(
      theta_training, sumstats_training, sumstats_observed,
      learning_rate = 10.0
    ),
    "very high learning rate"
  )
  
  expect_warning(
    abcnn$new(
      theta_training, sumstats_training, sumstats_observed,
      learning_rate = 1e-10
    ),
    "very low learning rate"
  )
  
  # Test extreme regularization values
  expect_error(
    abcnn$new(
      theta_training, sumstats_training, sumstats_observed,
      method = "concrete dropout", weight_regularizer = -1
    ),
    "must be positive"
  )
  
  # Test extreme network sizes
  expect_warning(
    abcnn$new(
      theta_training, sumstats_training, sumstats_observed,
      num_hidden_layers = 20
    ),
    "large number of layers"
  )
  
  expect_warning(
    abcnn$new(
      theta_training, sumstats_training, sumstats_observed,
      num_hidden_dim = 10000
    ),
    "large hidden dimension"
  )
})

test_that("abcnn handles memory and computational constraints", {
  # Test with very large dataset
  n_large <- 10000
  theta_large <- data.frame(param1 = runif(n_large, 0, 10))
  sumstats_large <- data.frame(stat1 = theta_large$param1 + rnorm(n_large, 0, 0.5))
  observed_large <- data.frame(stat1 = 5.2)
  
  expect_warning(
    {
      abc_large <- abcnn$new(
        theta_large, sumstats_large, observed_large,
        method = "monte carlo dropout", epochs = 1, verbose = FALSE
      )
      abc_large$fit()
    },
    "large dataset"
  )
  
  # Test with very small dataset
  n_small <- 10
  theta_small <- data.frame(param1 = runif(n_small, 0, 10))
  sumstats_small <- data.frame(stat1 = theta_small$param1 + rnorm(n_small, 0, 0.5))
  observed_small <- data.frame(stat1 = 5.2)
  
  expect_warning(
    {
      abc_small <- abcnn$new(
        theta_small, sumstats_small, observed_small,
        method = "monte carlo dropout", epochs = 1, verbose = FALSE
      )
      abc_small$fit()
    },
    "small dataset"
  )
  
  # Test with extreme batch sizes
  n_normal <- 100
  theta_normal <- data.frame(param1 = runif(n_normal, 0, 10))
  sumstats_normal <- data.frame(stat1 = theta_normal$param1 + rnorm(n_normal, 0, 0.5))
  observed_normal <- data.frame(stat1 = 5.2)
  
  # Batch size larger than dataset
  expect_warning(
    abcnn$new(
      theta_normal, sumstats_normal, observed_normal,
      batch_size = 200
    ),
    "batch size larger"
  )
  
  # Batch size of 1
  expect_warning(
    abcnn$new(
      theta_normal, sumstats_normal, observed_normal,
      batch_size = 1
    ),
    "batch size of 1"
  )
})

test_that("abcnn handles training failures gracefully", {
  set.seed(123)
  n_samples <- 50
  theta_training <- data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5))
  sumstats_observed <- data.frame(stat1 = 5.2)
  
  abc <- abcnn$new(
    theta_training, sumstats_training, sumstats_observed,
    method = "monte carlo dropout", epochs = 1, verbose = FALSE
  )
  
  # Test prediction without fitting
  expect_error(
    abc$predict(),
    "must be fitted before prediction"
  )
  
  # Test posterior sampling without prediction
  expect_error(
    abc$posterior(),
    "must predict before sampling"
  )
  
  # Test plotting without fitting
  expect_error(
    abc$plot_training(),
    "must be fitted before plotting"
  )
  
  # Test summary without fitting
  expect_error(
    abc$summary(),
    "must be fitted before summary"
  )
  
  # Test with NaN loss (simulated by problematic data)
  theta_nan <- data.frame(param1 = c(1, 2, Inf, 4, 5))
  sumstats_nan <- data.frame(stat1 = c(1, 2, NaN, 4, 5))
  observed_nan <- data.frame(stat1 = 3)
  
  expect_warning(
    {
      abc_nan <- abcnn$new(
        theta_nan, sumstats_nan, observed_nan,
        method = "monte carlo dropout", epochs = 1, verbose = FALSE
      )
      abc_nan$fit()
    },
    "NaN or Inf values"
  )
})

test_that("abcnn handles device and torch issues", {
  set.seed(123)
  n_samples <- 50
  theta_training <- data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5))
  sumstats_observed <- data.frame(stat1 = 5.2)
  
  # Test with manual device specification (if supported)
  abc <- abcnn$new(
    theta_training, sumstats_training, sumstats_observed,
    method = "monte carlo dropout", epochs = 1, verbose = FALSE
  )
  
  # Should handle device detection gracefully
  expect_no_error(abc$fit())
  
  # Test memory cleanup
  expect_no_error(abc$cleanup())
})

test_that("abcnn handles data preprocessing edge cases", {
  # Test with data having very different scales
  n_samples <- 50
  theta_mixed <- data.frame(
    param1 = runif(n_samples, 0, 1),  # Small scale
    param2 = runif(n_samples, 0, 1000000)  # Large scale
  )
  sumstats_mixed <- data.frame(
    stat1 = rnorm(n_samples, 0, 0.001),  # Small variance
    stat2 = rnorm(n_samples, 0, 1000)  # Large variance
  )
  observed_mixed <- data.frame(stat1 = 0.0005, stat2 = 500)
  
  expect_no_error({
    abc_mixed <- abcnn$new(
      theta_mixed, sumstats_mixed, observed_mixed,
      method = "monte carlo dropout", epochs = 1, verbose = FALSE
    )
    abc_mixed$fit()
  })
  
  # Test with collinear data
  theta_collinear <- data.frame(param1 = 1:50)
  sumstats_collinear <- data.frame(
    stat1 = theta_collinear$param1,
    stat2 = 2 * theta_collinear$param1,  # Perfect correlation
    stat3 = theta_collinear$param1 + rnorm(50, 0, 0.001)  # Near correlation
  )
  observed_collinear <- data.frame(stat1 = 25, stat2 = 50, stat3 = 25.1)
  
  expect_warning(
    {
      abc_collinear <- abcnn$new(
        theta_collinear, sumstats_collinear, observed_collinear,
        method = "monte carlo dropout", epochs = 1, verbose = FALSE
      )
      abc_collinear$fit()
    },
    "collinear"
  )
  
  # Test with zero variance data
  theta_zero <- data.frame(param1 = rep(5, 50))
  sumstats_zero <- data.frame(
    stat1 = rep(10, 50),
    stat2 = rnorm(50, 0, 1)  # One column has variance
  )
  observed_zero <- data.frame(stat1 = 10, stat2 = 0)
  
  expect_warning(
    {
      abc_zero <- abcnn$new(
        theta_zero, sumstats_zero, observed_zero,
        method = "monte carlo dropout", epochs = 1, verbose = FALSE
      )
      abc_zero$fit()
    },
    "zero variance"
  )
})

test_that("abcnn handles method-specific edge cases", {
  set.seed(123)
  n_samples <- 50
  theta_training <- data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5))
  sumstats_observed <- data.frame(stat1 = 5.2)
  
  # Test Concrete Dropout with extreme regularization
  expect_warning(
    abcnn$new(
      theta_training, sumstats_training, sumstats_observed,
      method = "concrete dropout",
      weight_regularizer = 1e10,
      dropout_regularizer = 1e10,
      epochs = 1, verbose = FALSE
    ),
    "extreme regularization"
  )
  
  # Test Deep Ensemble with insufficient data for multiple models
  expect_warning(
    abcnn$new(
      theta_training, sumstats_training, sumstats_observed,
      method = "deep ensemble",
      epochs = 1, verbose = FALSE
    ),
    "insufficient data"
  )
  
  # Test TabNet-ABC with insufficient features
  theta_tabnet <- data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_tabnet <- data.frame(stat1 = theta_tabnet$param1 + rnorm(n_samples, 0, 0.5))
  observed_tabnet <- data.frame(stat1 = 5.2)
  
  expect_warning(
    abcnn$new(
      theta_tabnet, sumstats_tabnet, observed_tabnet,
      method = "tabnet-abc",
      epochs = 1, verbose = FALSE
    ),
    "limited features"
  )
  
  # Test TabNet-ABC with extreme tolerance
  expect_error(
    abcnn$new(
      theta_tabnet, sumstats_tabnet, observed_tabnet,
      method = "tabnet-abc",
      tol = 2.0  # Invalid tolerance > 1
    ),
    "tol must be between"
  )
})

test_that("abcnn handles concurrent operations safely", {
  set.seed(123)
  n_samples <- 50
  theta_training <- data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5))
  sumstats_observed <- data.frame(stat1 = 5.2)
  
  abc <- abcnn$new(
    theta_training, sumstats_training, sumstats_observed,
    method = "monte carlo dropout", epochs = 1, verbose = FALSE
  )
  
  # Test multiple operations in sequence
  abc$fit()
  abc$predict()
  abc$posterior()
  abc$credible_interval()
  
  # Test re-fitting
  expect_no_error(abc$fit())
  expect_no_error(abc$predict())
  
  # Test state consistency
  expect_true(abc$trained)
  expect_true(is.finite(abc$predictive_mean[1, 1]))
})

test_that("abcnn handles file system edge cases", {
  set.seed(123)
  n_samples <- 50
  theta_training <- data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5))
  sumstats_observed <- data.frame(stat1 = 5.2)
  
  abc <- abcnn$new(
    theta_training, sumstats_training, sumstats_observed,
    method = "monte carlo dropout", epochs = 1, verbose = FALSE
  )
  abc$fit()
  
  # Test saving to non-existent directory
  non_existent_dir <- "/tmp/non_existent_abc_test_dir"
  expect_error(
    save_abcnn(abc, prefix = file.path(non_existent_dir, "test")),
    "No such file or directory"
  )
  
  # Test saving with invalid prefix
  expect_error(
    save_abcnn(abc, prefix = ""),
    "prefix cannot be empty"
  )
  
  # Test loading corrupted files
  temp_prefix <- tempfile("corrupted_test_")
  saveRDS(list(corrupted = "data"), paste0(temp_prefix, "_abcnn.Rds"))
  
  expect_error(
    load_abcnn(prefix = temp_prefix),
    "corrupted or incomplete"
  )
  
  # Clean up
  unlink(paste0(temp_prefix, "*"))
}
)
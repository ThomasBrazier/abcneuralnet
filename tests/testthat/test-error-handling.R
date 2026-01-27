# Comprehensive error handling and edge case tests
test_that("abcnn handles invalid inputs gracefully", {
  set.seed(123)
  n_samples = 10000
  theta_training = data.frame(param1 = runif(n_samples, 0, 1))
  sumstats_training = data.frame(
    stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.05),
    stat2 = theta_training$param1^2 + rnorm(n_samples, 0, 0.1)
  )
  sumstats_observed = data.frame(stat1 = 0.4, stat2 = 0.2)
  
  # Test with NULL inputs
  expect_error(
    abcnn$new(NULL, sumstats_training, sumstats_observed),
    "'theta' has to be a data.frame with column names."
  )
  expect_error(
    abcnn$new(theta_training, NULL, sumstats_observed),
    "'sumstat' has to be a data.frame with column names."
  )
  expect_error(
    abcnn$new(theta_training, sumstats_training, NULL),
    "'observed' has to be a data.frame with column names."
  )
  
  # Test with empty data frames
  expect_error(
    abcnn$new(data.frame(), sumstats_training, sumstats_observed),
    "'theta' is empty."
  )
  expect_error(
    abcnn$new(theta_training, data.frame(), sumstats_observed),
    "'sumstat' is empty."
  )
  expect_error(
    abcnn$new(theta_training, sumstats_training, data.frame()),
    "'observed' is empty."
  )
  
  # Test with mismatched row counts
  expect_error(
    abcnn$new(theta_training[1:100,, drop = F], sumstats_training, sumstats_observed),
    "Mismatch row count between training theta and summary statistics."
  )
  
  expect_error(
    abcnn$new(theta_training, sumstats_training[1:100,, drop = F], sumstats_observed),
    "Mismatch row count between training theta and summary statistics."
  )
  
  # Test with non-numeric data
  theta_char = theta_training
  theta_char$param1 = as.character(theta_char$param1)
  sumstats_char = sumstats_training
  sumstats_char$stat1 = as.character(sumstats_char$stat1)
  observed_char = sumstats_observed
  observed_char$stat1 = as.character(observed_char$stat1)
  
  expect_error(
    abcnn$new(theta_char, sumstats_training, sumstats_observed),
    "'theta' must be numeric."
  )
  expect_error(
    abcnn$new(theta_training, sumstats_char, sumstats_observed),
    "'sumstat' must be numeric."
  )
  expect_error(
    abcnn$new(theta_training, sumstats_training, observed_char),
    "'observed' must be numeric."
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
  n_samples = 50
  theta_training = data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training = data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5))
  sumstats_observed = data.frame(stat1 = 5.2)
  
  # TODO Tests
})


test_that("abcnn handles training failures gracefully", {
  set.seed(123)
  n_samples = 50
  theta_training = data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training = data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5))
  sumstats_observed = data.frame(stat1 = 5.2)
  
  abc = abcnn$new(
    theta_training, sumstats_training, sumstats_observed,
    method = "monte carlo dropout", epochs = 1, verbose = FALSE
  )
  
  # TODO Test prediction without fitting
  # expect_error(
  #   abc$predict(),
  #   "must be fitted before prediction"
  # )
  
  # TODO Test posterior sampling without prediction
  # expect_error(
  #   abc$posterior(),
  #   "must predict before sampling"
  # )

  # TODO Test plotting without fitting
  # expect_error(
  #   abc$plot_training(),
  #   "must be fitted before plotting"
  # )
  
  
  # TODO Test with NaN loss (simulated by problematic data)
  # theta_nan = data.frame(param1 = c(1, 2, Inf, 4, 5))
  # sumstats_nan = data.frame(stat1 = c(1, 2, NaN, 4, 5))
  # observed_nan = data.frame(stat1 = 3)
  # 
  # expect_warning(
  #   {
  #     abc_nan = abcnn$new(
  #       theta_nan, sumstats_nan, observed_nan,
  #       method = "monte carlo dropout", epochs = 1, verbose = FALSE
  #     )
  #     abc_nan$fit()
  #   },
  #   "NaN or Inf values"
  # )
})

test_that("abcnn handles device and torch issues", {
  set.seed(123)
  n_samples = 50
  theta_training = data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training = data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5))
  sumstats_observed = data.frame(stat1 = 5.2)
  
  # TODO
})


test_that("abcnn handles data preprocessing edge cases", {
  # Test with data having very different scales
  n_samples = 50
  theta_mixed = data.frame(
    param1 = runif(n_samples, 0, 1),  # Small scale
    param2 = runif(n_samples, 0, 1000000)  # Large scale
  )
  sumstats_mixed = data.frame(
    stat1 = rnorm(n_samples, 0, 0.001),  # Small variance
    stat2 = rnorm(n_samples, 0, 1000)  # Large variance
  )
  observed_mixed = data.frame(stat1 = 0.0005, stat2 = 500)
  
  # expect_no_error({
  #   abc_mixed = abcnn$new(
  #     theta_mixed, sumstats_mixed, observed_mixed,
  #     method = "monte carlo dropout", epochs = 1, verbose = FALSE
  #   )
  #   abc_mixed$fit()
  # })
  
  # TODO Test with collinear data

  # TODO Test with zero variance data

})



test_that("abcnn handles concurrent operations safely", {
  set.seed(123)
  n_samples = 50
  theta_training = data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training = data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5))
  sumstats_observed = data.frame(stat1 = 5.2)
  
  # abc = abcnn$new(
  #   theta_training, sumstats_training, sumstats_observed,
  #   method = "monte carlo dropout", epochs = 1, verbose = FALSE
  # )
  
  # Test multiple operations in sequence
  # abc$fit()
  # abc$predict()
  # abc$posterior()
  # abc$credible_interval()
  
  # TODO Test re-fitting
  # expect_no_error(abc$fit())
  # expect_no_error(abc$predict())
  
  # TODO Test re-predicting
  
})



test_that("abcnn handles file system edge cases", {
  set.seed(123)
  n_samples = 50
  theta_training = data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_training = data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.5))
  sumstats_observed = data.frame(stat1 = 5.2)
  
  # abc = abcnn$new(
  #   theta_training, sumstats_training, sumstats_observed,
  #   method = "monte carlo dropout", epochs = 1, verbose = FALSE
  # )
  # abc$fit()
  
  # TODO Test saving to non-existent directory
  # non_existent_dir = "/tmp/non_existent_abc_test_dir"
  # expect_error(
  #   save_abcnn(abc, prefix = file.path(non_existent_dir, "test")),
  #   "No such file or directory"
  # )
  
  # TODO Test saving with invalid prefix
  # expect_error(
  #   save_abcnn(abc, prefix = ""),
  #   "prefix cannot be empty"
  # )
  
  # TODO Test loading files not existing
  # temp_prefix = tempfile("corrupted_test_")
  # saveRDS(list(corrupted = "data"), paste0(temp_prefix, "_abcnn.Rds"))
  # 
  # expect_error(
  #   load_abcnn(prefix = temp_prefix),
  #   "corrupted or incomplete"
  # )

}
)
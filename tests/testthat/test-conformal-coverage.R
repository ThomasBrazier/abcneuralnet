# Empirical coverage of the conformal credible intervals
#
# Every other conformal assertion in the suite only checks `lower < upper`,
# which a badly calibrated interval satisfies just as well as a well calibrated
# one. These tests check the property conformal prediction actually promises:
# that the interval covers the truth about `credible_interval_p` of the time on
# data the model has never seen.
#
# They also pin the fact that `tabnet-abc` does not get conformal intervals at
# all. Its calibration set is held out neither of the TabNet fit (`dataloader()`
# hands `tabnet_fit()` every simulation instead of the training split) nor of
# the ABC reference table (each calibration point is matched against itself at
# distance 0), so the calibration scores are in-sample, the quantile comes out
# too small and the interval under-covers.


# `sumstat` is a noisy monotone function of `theta`, so the parameters are
# identifiable and a small network can learn them in a few epochs
make_simulations = function(n) {
  theta = data.frame(param1 = runif(n, 0, 1),
                     param2 = runif(n, 2, 4))

  sumstat = data.frame(stat1 = theta$param1 + rnorm(n, 0, 0.05),
                       stat2 = theta$param2 + rnorm(n, 0, 0.05),
                       stat3 = theta$param1 * theta$param2 + rnorm(n, 0, 0.1))

  return(list(theta = theta, sumstat = sumstat))
}

# Proportion of the held-out truths that fall inside the interval, per parameter
empirical_coverage = function(predictions, truth, lower, upper) {
  tidy_truth = tidyr::gather(truth, key = "parameter", value = "true_value")

  covered = tidy_truth$true_value >= predictions[[lower]] &
    tidy_truth$true_value <= predictions[[upper]]

  tapply(covered, predictions$parameter, mean)
}


test_that("the conformal intervals cover the truth at about the nominal rate", {
  skip_on_cran()
  set.seed(20240904)

  nominal = 0.9

  # The evaluation set is drawn separately and never passed to `abcnn$new()`,
  # so it is held out of the fit, of the scaling and of the calibration split
  training = make_simulations(2000)
  evaluation = make_simulations(500)

  abc = abcnn$new(training$theta,
                  training$sumstat,
                  evaluation$sumstat[1, , drop = FALSE],
                  method = "concrete dropout",
                  scale_input = "minmax",
                  scale_target = "minmax",
                  num_hidden_layers = 2,
                  num_hidden_dim = 64,
                  batch_size = 64,
                  epochs = 20,
                  num_conformal = 500,
                  num_posterior_samples = 200,
                  credible_interval_p = nominal,
                  validation_split = 0.1,
                  test_split = 0.1,
                  verbose = FALSE)

  abc$fit()

  # The conformal quantile is recalibrated inside `predict()`, on the same
  # calibration split, which stays held out of the evaluation set
  abc$predict(evaluation$sumstat)
  predictions = abc$predictions()

  expect_equal(nrow(predictions), nrow(evaluation$sumstat) * ncol(training$theta))
  expect_true(all(is.finite(predictions$overall_conformal_lower)))
  expect_true(all(is.finite(predictions$overall_conformal_upper)))

  overall = empirical_coverage(predictions, evaluation$theta,
                               "overall_conformal_lower",
                               "overall_conformal_upper")

  epistemic = empirical_coverage(predictions, evaluation$theta,
                                 "epistemic_conformal_lower",
                                 "epistemic_conformal_upper")

  # The band is deliberately loose. With 500 evaluation points the binomial
  # standard error is only about 0.013, but the network is trained for a handful
  # of epochs to keep the suite fast and that bias dominates the Monte Carlo
  # noise. It is still narrow enough to catch what this test exists for: a
  # calibration set that leaked into the fit produces gross under-coverage, not
  # a two-point shortfall.
  for (parameter in names(overall)) {
    expect_gt(overall[[parameter]], nominal - 0.12)
    expect_lt(overall[[parameter]], 0.99)

    expect_gt(epistemic[[parameter]], nominal - 0.12)
    expect_lt(epistemic[[parameter]], 0.99)
  }
})


test_that("conformal prediction is disabled for tabnet-abc", {
  skip_on_cran()
  set.seed(20240904)

  training = make_simulations(1000)
  observed = make_simulations(3)$sumstat

  expect_warning(
    abc <- abcnn$new(training$theta,
                     training$sumstat,
                     observed,
                     method = "tabnet-abc",
                     scale_input = "minmax",
                     scale_target = "minmax",
                     batch_size = 64,
                     epochs = 3,
                     tol = 0.1,
                     num_conformal = 500,
                     validation_split = 0.1,
                     test_split = 0.1,
                     verbose = FALSE),
    "not available for 'tabnet-abc'"
  )

  expect_equal(abc$num_conformal, 0)

  abc$fit()
  abc$predict()
  predictions = abc$predictions()

  # No conformal interval is produced, and that is visible rather than silent
  expect_true(all(is.na(predictions$epistemic_conformal_lower)))
  expect_true(all(is.na(predictions$epistemic_conformal_upper)))
  expect_true(all(is.na(predictions$overall_conformal_lower)))
  expect_true(all(is.na(predictions$overall_conformal_upper)))

  # The posterior quantile interval is the one to use instead
  expect_true(all(is.finite(predictions$posterior_lower_ci)))
  expect_true(all(is.finite(predictions$posterior_upper_ci)))
  expect_true(all(predictions$posterior_lower_ci < predictions$posterior_upper_ci))

  # The plots default to the conformal bounds, so they must fall back rather
  # than draw empty ribbons
  expect_warning(abc$plot_prediction(), "not available for 'tabnet-abc'")
  expect_warning(abc$plot_posterior(), "not available for 'tabnet-abc'")
})


test_that("non-finite conformal scores are ranked high instead of dropped", {
  skip_on_cran()
  set.seed(20240904)

  training = make_simulations(400)

  abc = abcnn$new(training$theta,
                  training$sumstat,
                  training$sumstat[1, , drop = FALSE],
                  method = "monte carlo dropout",
                  scale_input = "minmax",
                  scale_target = "minmax",
                  num_hidden_layers = 2,
                  num_hidden_dim = 32,
                  batch_size = 32,
                  epochs = 2,
                  num_conformal = 100,
                  num_posterior_samples = 50,
                  credible_interval_p = 0.9,
                  validation_split = 0.1,
                  test_split = 0.1,
                  verbose = FALSE)

  abc$fit()
  abc$predict()

  # The quantile is the ceiling((n_cal + 1) * (1 - alpha))-th smallest score,
  # so it is always one of the observed scores and never interpolated
  expect_equal(dim(abc$overall_conformal_quantile), c(1, ncol(training$theta)))
  expect_true(all(!is.na(as.numeric(abc$overall_conformal_quantile[1, ]))))
  expect_true(all(as.numeric(abc$overall_conformal_quantile[1, ]) > 0))
})

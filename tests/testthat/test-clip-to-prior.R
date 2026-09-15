# `predictions()`'s `clip_to_prior` argument.
#
# A parameter lies almost surely within its prior support, so intersecting a
# credible interval with that support never reduces its coverage - it only
# removes impossible values (e.g. a negative lower bound for a `minmax`-scaled
# rate). `predictions()` clips its six interval-endpoint columns to
# `[prior_lower, prior_upper]` by default, while always keeping the unclipped
# values available under a `_raw` suffix for diagnostics. Point estimates
# (`predictive_mean`, `posterior_median`) are never clipped: the coverage
# argument applies to intervals, not point estimates.


# An `abcnn` object with the target summary statistics filled in, but no fit.
# Mirrors the fixture in `test-backtransform.R`.
make_clip_test_abcnn = function(scale_target, theta_min = 0, theta_max = 1, n = 200) {
  set.seed(42)

  theta = data.frame(param1 = seq(theta_min, theta_max, length.out = n))
  sumstat = data.frame(stat1 = theta$param1 + rnorm(n, 0, 0.01))
  observed = data.frame(stat1 = theta_min + c(0.3, 0.5, 0.7) * (theta_max - theta_min))

  abc = abcnn$new(theta,
                  sumstat,
                  observed,
                  method = "deep ensemble",
                  scale_input = "none",
                  scale_target = scale_target,
                  num_hidden_layers = 2,
                  num_hidden_dim = 8,
                  epochs = 1,
                  num_conformal = 0,
                  verbose = FALSE)

  # The summary statistics `dataloader()` would learn on the training set
  abc$target_summary = list(min = min(theta$param1),
                            max = max(theta$param1),
                            mean = mean(theta$param1),
                            sd = sd(theta$param1),
                            quantile_25 = quantile(theta$param1, 0.25),
                            quantile_75 = quantile(theta$param1, 0.75),
                            n_training = n)
  abc$n_obs = nrow(observed)

  return(abc)
}

# Fill the slots `predictions()` reads, all in the scaled space
set_clip_test_predictions = function(abc, mu, sd, q_hat) {
  abc$predictive_mean = data.frame(param1 = mu)
  abc$epistemic_uncertainty = data.frame(param1 = sd)
  abc$aleatoric_uncertainty = data.frame(param1 = sd)
  abc$overall_uncertainty = data.frame(param1 = sd)
  abc$epistemic_conformal_quantile = data.frame(param1 = q_hat)
  abc$overall_conformal_quantile = data.frame(param1 = q_hat)

  return(abc)
}


test_that("predictions() clips interval endpoints to the prior support by default", {
  # theta ranges over [0, 1] (minmax back-transform is the identity here), and
  # mu +/- q_hat * sd overshoots both ends in the scaled space, so the raw
  # back-transformed bounds land outside [0, 1] - the parameter's prior support.
  abc = set_clip_test_predictions(make_clip_test_abcnn("minmax", theta_min = 0, theta_max = 1),
                        mu = c(0.1, 0.5, 0.9),
                        sd = c(0.2, 0.1, 0.2),
                        q_hat = 2)

  pred = abc$predictions()

  # Confirm the scenario actually produces out-of-support raw values,
  # otherwise this test would not exercise the fix
  expect_true(any(pred$overall_conformal_lower_raw < 0))
  expect_true(any(pred$overall_conformal_upper_raw > 1))
  expect_true(any(pred$epistemic_conformal_lower_raw < 0))
  expect_true(any(pred$epistemic_conformal_upper_raw > 1))

  # The clipped (default) columns never fall outside the prior support
  expect_true(all(pred$overall_conformal_lower >= 0))
  expect_true(all(pred$overall_conformal_upper <= 1))
  expect_true(all(pred$epistemic_conformal_lower >= 0))
  expect_true(all(pred$epistemic_conformal_upper <= 1))
})


test_that("the `_raw` columns always expose the unclipped endpoints", {
  abc = set_clip_test_predictions(make_clip_test_abcnn("minmax", theta_min = 0, theta_max = 1),
                        mu = c(0.1, 0.5, 0.9),
                        sd = c(0.2, 0.1, 0.2),
                        q_hat = 2)

  pred_clipped = abc$predictions(clip_to_prior = TRUE)
  pred_unclipped = abc$predictions(clip_to_prior = FALSE)

  bound_cols = c("epistemic_conformal_lower", "epistemic_conformal_upper",
                "overall_conformal_lower", "overall_conformal_upper",
                "posterior_lower_ci", "posterior_upper_ci")

  for (col in bound_cols) {
    raw_col = paste0(col, "_raw")

    # `_raw` is identical regardless of `clip_to_prior`
    expect_equal(pred_clipped[[raw_col]], pred_unclipped[[raw_col]], info = col)

    # `clip_to_prior = FALSE` returns the main column exactly equal to `_raw`
    expect_equal(pred_unclipped[[col]], pred_unclipped[[raw_col]], info = col)
  }
})


test_that("point estimates are never clipped", {
  # mu = -0.5 and mu = 1.5 are nonsensical predictions outside [0, 1],
  # deliberately chosen to check that predictive_mean is left alone: the
  # coverage argument for clipping applies to intervals, not point estimates.
  abc = set_clip_test_predictions(make_clip_test_abcnn("minmax", theta_min = 0, theta_max = 1),
                        mu = c(-0.5, 0.5, 1.5),
                        sd = c(0.05, 0.05, 0.05),
                        q_hat = 1)

  pred = abc$predictions()

  expect_equal(pred$predictive_mean, c(-0.5, 0.5, 1.5))
})


test_that("plot_prediction() and plot_posterior() run with clip_to_prior TRUE and FALSE", {
  set.seed(123)
  n_samples = 500
  theta_training = data.frame(param1 = runif(n_samples, 0, 1))
  sumstats_training = data.frame(stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.05))
  sumstats_observed = data.frame(stat1 = c(0.2, 0.5, 0.8))

  abc = abcnn$new(theta_training,
                  sumstats_training,
                  sumstats_observed,
                  method = "deep ensemble",
                  scale_input = "none",
                  scale_target = "minmax",
                  num_hidden_layers = 2,
                  num_hidden_dim = 16,
                  epochs = 2,
                  batch_size = 64,
                  num_conformal = 0,
                  verbose = FALSE)

  abc$fit()
  abc$predict()

  for (clip in c(TRUE, FALSE)) {
    expect_no_error(abc$plot_prediction(uncertainty_type = "conformal", clip_to_prior = clip))
    expect_no_error(abc$plot_prediction(uncertainty_type = "uncertainty", clip_to_prior = clip))
    expect_no_error(abc$plot_posterior(uncertainty_type = "conformal", clip_to_prior = clip))
    expect_no_error(abc$plot_posterior(uncertainty_type = "uncertainty", clip_to_prior = clip))
  }
})

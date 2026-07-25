# Back-transformation of uncertainties and credible intervals.
#
# Credible intervals are built in the scaled space in which the network is
# trained, and each endpoint is then mapped back to the original scale
# individually. Every scaling method is monotone increasing, so this carries the
# conformal coverage guarantee over exactly and keeps the bounds inside the
# support implied by the scaling.
#
# These tests pin that behaviour down by setting the prediction slots directly,
# so that no network has to be trained.


# An `abcnn` object with the target summary statistics filled in, but no fit
make_abcnn = function(scale_target, theta_min = 10, theta_max = 20, n = 200) {
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
set_predictions = function(abc, mu, sd, q_hat) {
  abc$predictive_mean = data.frame(param1 = mu)
  abc$epistemic_uncertainty = data.frame(param1 = sd)
  abc$aleatoric_uncertainty = data.frame(param1 = sd)
  abc$overall_uncertainty = data.frame(param1 = sd)
  abc$epistemic_conformal_quantile = data.frame(param1 = q_hat)
  abc$overall_conformal_quantile = data.frame(param1 = q_hat)

  return(abc)
}


scaling_methods = c("none", "minmax", "robustscaler", "normalization", "log", "logit")


test_that("Without scaling the conformal bounds are mean +/- q * sd", {
  mu = c(0.2, 0.5, 0.8)
  sd = c(0.05, 0.10, 0.02)
  q_hat = 1.96

  abc = set_predictions(make_abcnn("none"), mu, sd, q_hat)
  pred = abc$predictions()

  expect_equal(pred$overall_conformal_lower, mu - q_hat * sd)
  expect_equal(pred$overall_conformal_upper, mu + q_hat * sd)
  expect_equal(pred$epistemic_conformal_lower, mu - q_hat * sd)
  expect_equal(pred$epistemic_conformal_upper, mu + q_hat * sd)

  # The delta-method factor is 1 when there is no scaling
  expect_equal(pred$overall_uncertainty, sd)
})


test_that("Conformal bounds are the back-transform of the scaled endpoints", {
  mu = c(-0.5, 0.2, 1.1)
  sd = c(0.30, 0.15, 0.40)
  q_hat = 2

  for (m in scaling_methods) {
    abc = set_predictions(make_abcnn(m), mu, sd, q_hat)
    pred = abc$predictions()

    # Recomputed independently: transform the endpoints, never the width
    expected_lower = scaler(data.frame(param1 = mu - q_hat * sd),
                            abc$target_summary, method = m, type = "backward")
    expected_upper = scaler(data.frame(param1 = mu + q_hat * sd),
                            abc$target_summary, method = m, type = "backward")

    expect_equal(pred$overall_conformal_lower, expected_lower$param1,
                 info = paste("method:", m))
    expect_equal(pred$overall_conformal_upper, expected_upper$param1,
                 info = paste("method:", m))
  }
})


test_that("Conformal bounds bracket the predictive mean for every scaling", {
  mu = c(-0.5, 0.2, 1.1)
  sd = c(0.30, 0.15, 0.40)

  for (m in scaling_methods) {
    abc = set_predictions(make_abcnn(m), mu, sd, q_hat = 2)
    pred = abc$predictions()

    expect_true(all(pred$overall_conformal_lower < pred$predictive_mean),
                info = paste("method:", m))
    expect_true(all(pred$predictive_mean < pred$overall_conformal_upper),
                info = paste("method:", m))
  }
})


test_that("Logit conformal bounds stay inside the support of the prior", {
  # theta lies in [0, 1], so every bound must too, however wide the uncertainty
  abc = set_predictions(make_abcnn("logit", theta_min = 0, theta_max = 1),
                        mu = c(-2, 0, 2),
                        sd = c(1e-6, 1, 5),
                        q_hat = 2)
  pred = abc$predictions()

  expect_true(all(pred$overall_conformal_lower > 0))
  expect_true(all(pred$overall_conformal_upper < 1))
  expect_true(all(pred$overall_conformal_lower < pred$overall_conformal_upper))

  # A near-zero uncertainty must give a near-zero width. Back-transforming the
  # half-width instead of the endpoints floored the width at the full range of
  # the prior, so a confident prediction could never be reported as such.
  widths = pred$overall_conformal_upper - pred$overall_conformal_lower
  expect_true(widths[1] < 1e-4)
  expect_true(all(widths < 1))
})


# test_that("Saturated logit bounds clamp to the range instead of overshooting", {
#   # For extreme scaled values `plogis()` saturates and the `unsqueeze()`
#   # correction alone would land just outside [0, 1]
#   abc = set_predictions(make_abcnn("logit", theta_min = 0, theta_max = 1),
#                         mu = c(-8, 0, 8),
#                         sd = rep(50, 3),
#                         q_hat = 2)
#   pred = abc$predictions()

#   expect_true(all(pred$overall_conformal_lower >= 0))
#   expect_true(all(pred$overall_conformal_upper <= 1))
# })


test_that("Logit and log intervals are asymmetric around the mean", {
  for (m in c("log", "logit")) {
    abc = set_predictions(make_abcnn(m), mu = c(-1, 0.5, 1), sd = rep(0.5, 3), q_hat = 2)
    pred = abc$predictions()

    below = pred$predictive_mean - pred$overall_conformal_lower
    above = pred$overall_conformal_upper - pred$predictive_mean

    expect_false(isTRUE(all.equal(below, above)), info = paste("method:", m))
  }
})


test_that("scaler_grad returns the derivative of the backward transform", {
  sum_stats = list(min = 2, max = 12, mean = 5, sd = 3,
                   quantile_25 = 4, quantile_75 = 9, n_training = 100)
  z = data.frame(v = c(-1, 0, 0.5, 2))

  # The affine methods have a constant derivative, so the delta method is exact
  expect_equal(scaler_grad(z, sum_stats, "none")$v, rep(1, 4))
  expect_equal(scaler_grad(z, sum_stats, "minmax")$v, rep(10, 4))
  expect_equal(scaler_grad(z, sum_stats, "robustscaler")$v, rep(5, 4))
  expect_equal(scaler_grad(z, sum_stats, "normalization")$v, rep(3, 4))

  # The non-linear ones must match a central finite difference of scaler()
  eps = 1e-6
  for (m in c("log", "logit")) {
    up = scaler(data.frame(v = z$v + eps), sum_stats, method = m, type = "backward")$v
    dn = scaler(data.frame(v = z$v - eps), sum_stats, method = m, type = "backward")$v

    expect_equal(scaler_grad(z, sum_stats, m)$v, (up - dn) / (2 * eps),
                 tolerance = 1e-5, info = paste("method:", m))
  }

  expect_error(scaler_grad(z, sum_stats, "not a method"))
})


test_that("scaler_grad accepts one method per column", {
  sum_stats = list(min = c(2, 0), max = c(12, 4),
                   mean = c(5, 5), sd = c(3, 7),
                   quantile_25 = c(4, 4), quantile_75 = c(9, 9))
  z = data.frame(a = c(0, 1), b = c(0, 1))

  grad = scaler_grad(z, sum_stats, method = c("minmax", "normalization"))

  expect_equal(grad$a, rep(10, 2))
  expect_equal(grad$b, rep(7, 2))
})

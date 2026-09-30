# Rescaling of explainn attributions to a common scale across parameters.
#
# The network predicts the scaled target z, and theta = g(z). By the chain rule,
# attributions on the prior-range scale are the raw ones times |g'(z)| / (max - min).
# `rescale_result()` only reads `scale_target`, `target_summary` and `model` from
# the explainn object, so these tests fit a single small network and swap the
# target scaling on the explainn object to probe each case.

make_explainn = function(n = 2000) {
  set.seed(1)
  torch::torch_manual_seed(1)

  theta = data.frame(a = runif(n, 0, 1), b = runif(n, 100, 500))
  sumstat = data.frame(s1 = theta$a + rnorm(n, 0, 0.05),
                       s2 = theta$b / 100 + rnorm(n, 0, 0.05),
                       s3 = rnorm(n))
  observed = sumstat[1:10, ]

  abc = abcnn$new(theta,
                  sumstat,
                  observed,
                  method = "deep ensemble",
                  scale_input = "minmax",
                  scale_target = "minmax",
                  num_hidden_layers = 2,
                  num_hidden_dim = 8,
                  epochs = 2,
                  num_conformal = 0,
                  verbose = FALSE)
  abc$fit()

  exp = explainn$new(abc, method = "grad")
  list(abc = abc, exp = exp, data = observed)
}


test_that("minmax attributions are unchanged by prior_range rescaling", {
  skip_if_not_installed("innsight")
  m = make_explainn()

  m$exp$run(m$data, output_scale = "none")
  raw = m$exp$get_result()
  m$exp$run(m$data, output_scale = "prior_range")
  rescaled = m$exp$get_result()

  expect_equal(rescaled, raw, tolerance = 1e-5)
  expect_equal(m$exp$output_scale, "prior_range")
})


test_that("Affine target scalings are rescaled by a constant factor per parameter", {
  skip_if_not_installed("innsight")
  m = make_explainn()
  ts = m$exp$target_summary
  prior_range = ts$max - ts$min

  m$exp$run(m$data, output_scale = "none")
  raw = m$exp$get_result()

  expected_factor = list(none = 1 / prior_range,
                         normalization = ts$sd / prior_range,
                         robustscaler = (ts$quantile_75 - ts$quantile_25) / prior_range)

  for (method in names(expected_factor)) {
    m$exp$scale_target = method
    m$exp$run(m$data, output_scale = "prior_range")
    rescaled = m$exp$get_result()
    for (j in seq_along(prior_range)) {
      expect_equal(rescaled[, , j], raw[, , j] * expected_factor[[method]][j],
                   tolerance = 1e-5, ignore_attr = TRUE)
    }
  }
})


test_that("Mixed minmax/logit scalings use the per-sample logit gradient", {
  skip_if_not_installed("innsight")
  m = make_explainn()
  ts = m$exp$target_summary
  m$exp$scale_target = c("minmax", "logit")

  m$exp$run(m$data, output_scale = "none")
  raw = m$exp$get_result()
  m$exp$run(m$data, output_scale = "prior_range")
  rescaled = m$exp$get_result()

  scaled_data = scaler(m$data, m$exp$input_summary, method = m$exp$scale_input, type = "forward")
  z = torch::as_array(torch::with_no_grad(
    m$exp$model(torch::torch_tensor(as.matrix(scaled_data), dtype = torch::torch_float()))
  ))
  factor_logit = inv_logit_grad(z[, 2], ts$min[2], ts$max[2], 100000) / (ts$max[2] - ts$min[2])

  expect_equal(rescaled[, , 1], raw[, , 1], tolerance = 1e-5, ignore_attr = TRUE)
  expect_equal(rescaled[, , 2], raw[, , 2] * factor_logit, tolerance = 1e-5, ignore_attr = TRUE)
})


test_that("Relative attributions are shares that do not depend on scale_target", {
  skip_if_not_installed("innsight")
  m = make_explainn()

  m$exp$run(m$data, output_scale = "relative")
  rel_minmax = m$exp$get_result()
  expect_equal(apply(abs(rel_minmax), c(1, 3), sum),
               matrix(1, nrow = nrow(m$data), ncol = 2),
               tolerance = 1e-5, ignore_attr = TRUE)

  m$exp$scale_target = "logit"
  m$exp$run(m$data, output_scale = "relative")
  expect_equal(m$exp$get_result(), rel_minmax, tolerance = 1e-5)
})


test_that("cw is rescaled, and warns under logit scaling", {
  skip_if_not_installed("innsight")
  m = make_explainn()

  m$exp$method = "cw"
  expect_no_warning(m$exp$run(m$data))

  m$exp$scale_target = "logit"
  expect_warning(m$exp$run(m$data), "median of the scaled training targets")
  expect_true(all(is.finite(m$exp$get_result())))
})

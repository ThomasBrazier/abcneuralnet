# Test scaling summary statistics in
test_that("Scaling summary statistics works", {
  
  make_test_data = function() {
    t1 = seq(1, 10, length.out = 10)
    theta_training_y1 = t1
    sumstats_training_x1 = t1
    sumstats_observed_x1 = t1

    t2 = seq(20, 60, length.out = 10)
    theta_training_y2 = t2
    sumstats_training_x2 = t2
    sumstats_observed_x2 = t2

    return(list(t1 = t1,
                t2 = t2,
                theta_training_y1 = theta_training_y1,
                theta_training_y2 = theta_training_y2,
                sumstats_training_x1 = sumstats_training_x1,
                sumstats_training_x2 = sumstats_training_x2,
                sumstats_observed_x1 = sumstats_observed_x1,
                sumstats_observed_x2 = sumstats_observed_x2))
  }

  test_data = make_test_data()
  method = c("minmax", "robustscaler", "normalization", "none")
  
  input_summary = list(min = min(test_data$sumstats_training_x1),
                        max = max(test_data$sumstats_training_x1),
                        mean = mean(test_data$sumstats_training_x1),
                        sd = sd(test_data$sumstats_training_x1),
                       quantile_25 = quantile(test_data$sumstats_training_x1, 0.25),
                       quantile_75 = quantile(test_data$sumstats_training_x1, 0.75))
  
  for (m in method) {
    cat("Method:", m, "\n")
    cat("Forward\n")
    sc = scaler(test_data$sumstats_training_x1,
           input_summary,
           method = m,
           type = "forward")
    print(sc)
    
    cat("Backward\n")
    sc = scaler(sc,
                input_summary,
                method = m,
                type = "backward")
    print(sc)
    assertthat::assert_that(all(round(sc$x, digits = 0) == c(1:10)))
  }
    
  target_summary = list(min = min(test_data$sumstats_observed_x1),
                       max = max(test_data$sumstats_observed_x1),
                       mean = mean(test_data$sumstats_observed_x1),
                       sd = sd(test_data$sumstats_observed_x1),
                       quantile_25 = quantile(test_data$sumstats_observed_x1, 0.25),
                       quantile_75 = quantile(test_data$sumstats_observed_x1, 0.75))
  
  for (m in method) {
    cat("Method:", m, "\n")
    cat("Forward\n")
    sc = scaler(test_data$sumstats_observed_x1,
                target_summary,
                method = m,
                type = "forward")
    print(sc)
    
    cat("Backward\n")
    sc = scaler(sc,
                target_summary,
                method = m,
                type = "backward")
    print(sc)
    assertthat::assert_that(all(round(sc$x, digits = 0) == c(1:10)))
  }

})


# TODO Enhanced tests for utility functions
# test_that("scaler function handles all methods correctly", {
#   # Test data
#   x = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
#   
#   # Summary statistics
#   summary_stats = list(
#     min = min(x),
#     max = max(x),
#     mean = mean(x),
#     sd = sd(x),
#     quantile_25 = quantile(x, 0.25),
#     quantile_75 = quantile(x, 0.75)
#   )
#   
#   # Test each scaling method
#   methods = c("none", "minmax", "robustscaler", "normalization")
#   
#   for (method in methods) {
#     # Forward transformation
#     scaled = scaler(x, summary_stats, method = method, type = "forward")
#     expect_true(is.numeric(scaled$x))
#     expect_true(all(is.finite(scaled$x)))
#     
#     # Backward transformation should recover original
#     if (method != "none") {
#       original = scaler(scaled, summary_stats, method = method, type = "backward")
#       expect_true(all(abs(original$x - x) < 1e-10))
#     }
#   }
# })


# TODO
# test_that("abcnn data scaling works correctly", {
#   set.seed(123)
#   n_samples = 10000
#   theta_training = data.frame(param1 = runif(n_samples, 0, 1))
#   sumstats_training = data.frame(
#     stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.05),
#     stat2 = theta_training$param1^2 + rnorm(n_samples, 0, 0.1)
#   )
#   sumstats_observed = data.frame(stat1 = 0.4, stat2 = 0.2)
#   
#   # Test different scaling methods
#   scaling_methods = c("none", "minmax", "robustscaler")
#   
#   for (scale_method in scaling_methods) {
#     abc = abcnn$new(
#       theta_training,
#       sumstats_training,
#       sumstats_observed,
#       method = "monte carlo dropout",
#       scale_input = scale_method,
#       scale_target = scale_method,
#       epochs = 1,
#       verbose = FALSE
#     )
#     
#     abc$fit()
#     abc$predict()
#
#     # Check that proper scaling was applied
#     # TODO
#   }
# })


# Regression test: `log1pexp()` used to be implemented with
# `torch_where(x < threshold, log1p(exp(x)), x)`. `torch_where` evaluates both
# branches, so for `x >> threshold`, `exp(x)` overflows to `Inf`; the forward
# value is correctly discarded, but the backward pass still multiplies that
# branch's NaN gradient by a zero mask (0 * NaN = NaN), silently poisoning
# every parameter gradient. The forward value stays finite throughout, so a
# `is.nan(loss$item())` check (as used in `nn_ensemble$nll_loss()`) can never
# catch this - only checking the gradient does.
test_that("log1pexp() has no NaN forward value or gradient for large inputs", {
  x = torch::torch_tensor(c(-50, -1, 0, 1, 9, 10, 11, 50, 1e6, 1e15), requires_grad = TRUE)
  y = log1pexp(x)

  expect_false(any(as.array(torch::torch_isnan(y))))

  loss = torch::torch_sum(y)
  loss$backward()

  expect_false(any(as.array(torch::torch_isnan(x$grad))))

  # Well above `threshold`, softplus is linear (slope 1), matching the old `x` branch
  expect_equal(as.numeric(x$grad[8:10]), c(1, 1, 1), tolerance = 1e-5)
})


test_that("clamp_variance() floors negative input to zero and leaves positive input alone", {
  x = torch::torch_tensor(c(-1e-5, -1, 0, 3, 1e5))
  clamped = clamp_variance(x)

  expect_equal(as.numeric(clamped), c(0, 0, 0, 3, 1e5))
})


# Illustrative reproduction of the actual failure mode: Deep Ensemble's
# epistemic variance is `mean(mu^2) - mean(mu)^2` (`R/deep_ensemble.R`,
# `R/abcnn.R`). Mathematically this can never be negative, but for several
# *identical*, large-magnitude float32 values, `mean(mu^2)` and `mean(mu)^2`
# are each computed via a different arithmetic path and can round to values
# that are not bit-identical, so the subtraction can undershoot zero and
# `torch_sqrt()` of that is `NaN`. This is exactly what happens when Deep
# Ensemble's members closely agree - a confident, not a broken, ensemble.
test_that("clamp_variance() prevents NaN in an E[X^2] - E[X]^2 variance for agreeing float32 members", {
  mu = torch::torch_tensor(rep(100000, 5), dtype = torch::torch_float32())

  raw_variance = torch::torch_mean(torch::torch_square(mu)) - torch::torch_square(torch::torch_mean(mu))
  sd = torch::torch_sqrt(clamp_variance(raw_variance))

  expect_false(as.logical(torch::torch_isnan(sd)))
  expect_true(as.numeric(sd) >= 0)
})




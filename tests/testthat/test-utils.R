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






# Test scaling sunmmary statistics in
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

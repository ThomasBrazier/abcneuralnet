# Test if ABC-NN handle correctly input and output
# with one or more dimensions
test_that("ABC-NN handle correctly input and output with one or more dimensions", {

  # Make a 2 dimension input and output data frame
  make_test_data = function() {
    # Parameters of simulated input x
    data_range = 7
    data_step = 0.001

    # Boundaries of the gap in the data range
    bound1 = -2
    bound2 = 2

    # Random noise applied on y
    data_sigma1a = 0.1
    data_sigma2a = 0.5

    data_sigma1b = 0.2
    data_sigma2b = 0.1

    # Number of simulated data points
    # num_data = 10000

    # Simulate x1
    data_x1a = seq(-data_range, bound1 + data_step, by = data_step)
    data_x1b = seq(bound2, data_range + data_step, by = data_step)
    # Simulate targets y
    data_y1a = sin(data_x1a) + rnorm(length(data_x1a), 0, data_sigma1a)
    data_y1b = sin(data_x1b) + rnorm(length(data_x1b), 0, data_sigma2a)

    # Shift X1 to get X2
    data_x2a = data_x1a + 7
    data_x2b = data_x1b + 7
    # Simulate targets y
    data_y2a = cos(data_x2a) + rnorm(length(data_x2a), 0, data_sigma1b)
    data_y2b = cos(data_x2b) + rnorm(length(data_x2b), 0, data_sigma2b)

    df = data.frame(x1 = c(data_x1a, data_x1b),
                    x2 = c(data_x2a, data_x2b),
                    y1 = c(data_y1a, data_y1b),
                    y2 = c(data_y2a, data_y2b))

    # Shuffle data
    shuffle_idx = sample(1:(nrow(df)), nrow(df), replace = FALSE)
    df_train = df[shuffle_idx,]

    # Train/Test datasets
    train_x = df_train[, c("x1", "x2")]
    train_y = df_train[, c("y1", "y2")]

    # Make a pseudo-obseerved dataset with out of distribution data points
    # Simulate x1
    data_x1 = seq(-data_range, data_range, length.out = 100)
    # Simulate true targets y
    data_y1 = sin(data_x1)

    # Shift X1 to get X2
    data_x2 = data_x1 + 7
    # Simulate targets y
    data_y2 = cos(data_x2)

    df_observed = data.frame(x1 = data_x1,
                             x2 = data_x2,
                             y1 = data_y1,
                             y2 = data_y2)


    observed_x  = df_observed[, c("x1", "x2")]
    observed_y  = df_observed[, c("y1", "y2")]

    return(list(train_x = train_x,
                train_y = train_y,
                observed_x = observed_x,
                observed_y = observed_y))
  }

  test_data = make_test_data()

  # Test each method sequentially
  for (met in c("monte carlo dropout", "concrete dropout", "deep ensemble")) {
    # Test 1D
    theta_training = data.frame(y = test_data$train_y$y1)
    sumstats_training = data.frame(x = test_data$train_x$x1)
    sumstats_observed = data.frame(x = test_data$observed_x$x1)

    test_1d = abcnn$new(theta_training,
                        sumstats_training,
                        sumstats_observed,
                        method = met,
                        epochs = 3,
                        scale = TRUE)

    test_1d$fit()

    test_1d$predict()

    assertthat::assert_that(dim(test_1d$observed_adj)[1] == 100)
    assertthat::assert_that(dim(test_1d$sumstat_adj)[1] == 10004)
    assertthat::assert_that(dim(test_1d$theta)[1] == 10004)

    assertthat::assert_that(dim(test_1d$predictive_mean)[1] == 100)
    assertthat::assert_that(dim(test_1d$predictive_mean)[2] == 1)

    # Test 2D
    theta_training = data.frame(y = test_data$train_y)
    sumstats_training = data.frame(x = test_data$train_x)
    sumstats_observed = data.frame(x = test_data$observed_x)

    test_2d = abcnn$new(theta_training,
                        sumstats_training,
                        sumstats_observed,
                        method = met,
                        epochs = 3,
                        scale = TRUE)

    test_2d$fit()

    test_2d$predict()

    assertthat::assert_that(dim(test_2d$observed_adj)[1] == 100)
    assertthat::assert_that(dim(test_2d$sumstat_adj)[1] == 10004)
    assertthat::assert_that(dim(test_2d$theta)[1] == 10004)

    assertthat::assert_that(dim(test_2d$observed_adj)[2] == 2)
    assertthat::assert_that(dim(test_2d$sumstat_adj)[2] == 2)
    assertthat::assert_that(dim(test_2d$theta)[2] == 2)

    assertthat::assert_that(dim(test_2d$predictive_mean)[1] == 100)
    assertthat::assert_that(dim(test_2d$predictive_mean)[2] == 2)
  }

})





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

  theta_training = data.frame(y1 = test_data$theta_training_y1,
                              y2 = test_data$theta_training_y2)
  sumstats_training = data.frame(x1 = test_data$sumstats_training_x1,
                                 x2 = test_data$sumstats_training_x2)
  sumstats_observed = data.frame(x1 = test_data$sumstats_observed_x1,
                                 x2 = test_data$sumstats_observed_x2)

  test = abcnn$new(theta_training,
                    sumstats_training,
                    sumstats_observed,
                    scale = TRUE)

  t1 = test_data$t1
  t2 = test_data$t2

  assertthat::are_equal(as.numeric(test$sumstat_mean[1]), as.numeric(mean(t1)))
  assertthat::are_equal(as.numeric(test$sumstat_sd[1]), as.numeric(sd(t1)))

  assertthat::are_equal(as.numeric(test$sumstat_adj$x1), as.numeric((t1 - mean(t1)) / sd(t1)))
  assertthat::are_equal(as.numeric(mean(test$sumstat_adj$x1)), 0)
  assertthat::are_equal(as.numeric(sd(test$sumstat_adj$x1)), 1)

  assertthat::are_equal(as.numeric(test$observed_adj$x1), as.numeric((t1 - mean(t1)) / sd(t1)))
  assertthat::are_equal(as.numeric(mean(test$observed_adj$x1)), 0)
  assertthat::are_equal(as.numeric(sd(test$observed_adj$x1)), 1)
})







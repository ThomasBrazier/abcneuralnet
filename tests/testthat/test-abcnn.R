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
                        scale_input = "minmax")

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
                        scale_input = "minmax")

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



# Test random seed when initializing a torch model
test_that("Test random seed when initializing a torch model always return the same model", {
  # Make test data
  set.seed(123)
  n_samples = 10000
  theta_training = data.frame(param1 = runif(n_samples, 0, 1))
  sumstats_training = data.frame(
    stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.05),
    stat2 = theta_training$param1^2 + rnorm(n_samples, 0, 0.1)
  )
  sumstats_observed = data.frame(stat1 = c(0.4, 0.6), stat2 = c(0.2, 0.7))

  num_posterior_samples = 50

  methods = c("monte carlo dropout",
              "concrete dropout",
              "deep ensemble",
              "tabnet-abc")

  for (m in methods) {
    # Init an abcnn object with inputs and targets
    abc = abcnn$new(theta_training,
                    sumstats_training,
                    sumstats_observed,
                    method = m,
                    scale_input = "none",
                    scale_target = "none",
                    num_hidden_layers = 3,
                    num_hidden_dim = 128,
                    epochs = 3,
                    batch_size = 32,
                    tol = 0.1,
                    num_posterior_samples = num_posterior_samples,
                    abc_method = "loclinear",
                    seed = 42)

    abc$fit()
    abc$predict()

    pred_1 = abc$predictions()

    # Expect the same results when re-done with same seed
    abc = abcnn$new(theta_training,
                    sumstats_training,
                    sumstats_observed,
                    method = m,
                    scale_input = "none",
                    scale_target = "none",
                    num_hidden_layers = 3,
                    num_hidden_dim = 128,
                    epochs = 3,
                    batch_size = 32,
                    tol = 0.1,
                    num_posterior_samples = num_posterior_samples,
                    abc_method = "loclinear",
                    seed = 42)

    abc$fit()
    abc$predict()

    pred_2 = abc$predictions()

    expect_equal(pred_1, pred_2)

    # Expect a different ouotput with a different seed
    abc = abcnn$new(theta_training,
                    sumstats_training,
                    sumstats_observed,
                    method = m,
                    scale_input = "none",
                    scale_target = "none",
                    num_hidden_layers = 3,
                    num_hidden_dim = 128,
                    epochs = 3,
                    batch_size = 32,
                    tol = 0.1,
                    num_posterior_samples = num_posterior_samples,
                    abc_method = "loclinear",
                    seed = 94)

    abc$fit()
    abc$predict()

    pred_3 = abc$predictions()

    expect_false(isTRUE(all.equal(pred_1, pred_3)))

  }

})







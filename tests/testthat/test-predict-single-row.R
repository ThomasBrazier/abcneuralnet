# Predicting on a single observed sample
#
# A data frame with a single row is the most common case in practice (one real
# dataset to infer parameters from), and it is also the case where R silently
# drops dimensions: a one-row matrix becomes a vector, `apply()` over a degenerate
# array margin returns a vector instead of a matrix, and a one-row slice of a
# data frame loses its `data.frame` class without `drop = FALSE`.
#
# These tests check that `predict()` on a single row returns the same numbers as
# the batch prediction that contains that row, and that the shape of every slot
# it fills stays `1 x output_dim`.


# Two parameters and three summary statistics, so that a single-row prediction
# still has more than one column and exercises the `output_dim > 1` branches
make_training_data = function(n = 2000) {
  theta = data.frame(param1 = runif(n, 0, 1),
                     param2 = runif(n, 2, 4))

  sumstat = data.frame(stat1 = theta$param1 + rnorm(n, 0, 0.05),
                       stat2 = theta$param2 + rnorm(n, 0, 0.05),
                       stat3 = theta$param1 * theta$param2 + rnorm(n, 0, 0.1))

  return(list(theta = theta, sumstat = sumstat))
}

# Three observed samples, the middle one is the row predicted alone
make_observed = function() {
  data.frame(stat1 = c(0.2, 0.5, 0.8),
             stat2 = c(2.5, 3.0, 3.5),
             stat3 = c(0.5, 1.5, 2.8))
}

# A small model, kept cheap so that the whole file stays fast to run
fit_abcnn = function(method, training, observed, num_posterior_samples = 500) {
  abc = abcnn$new(training$theta,
                  training$sumstat,
                  observed,
                  method = method,
                  scale_input = "minmax",
                  scale_target = "minmax",
                  num_hidden_layers = 2,
                  num_hidden_dim = 32,
                  batch_size = 64,
                  epochs = 5,
                  num_networks = 3,
                  num_conformal = 100,
                  num_posterior_samples = num_posterior_samples,
                  tol = 0.1,
                  validation_split = 0.1,
                  test_split = 0.1,
                  verbose = FALSE)

  abc$fit()

  return(abc)
}

# The numerical columns of the tidy data frame returned by `predictions()`
prediction_columns = c("predictive_mean",
                       "epistemic_uncertainty",
                       "overall_uncertainty",
                       "epistemic_conformal_lower",
                       "epistemic_conformal_upper",
                       "overall_conformal_lower",
                       "overall_conformal_upper")

# A forward pass on a batch of one and on a batch of three do not sum in the same
# order, so identical inputs agree only to floating point noise. The predictive
# mean carries that noise as is, but the spreads are computed as
# `sqrt(E[mu^2] - E[mu]^2)`, a cancellation of two nearly equal numbers that
# amplifies it by a few orders of magnitude, and the conformal bounds inherit it.
mean_tolerance = 1e-5
spread_tolerance = 1e-3

# The tolerance to compare a given column of `predictions()` with
column_tolerance = function(column) {
  if (column == "predictive_mean") mean_tolerance else spread_tolerance
}


# `deep ensemble` is the only method whose prediction is deterministic (no
# dropout mask is sampled at prediction time), so single-row and batch results
# can be compared exactly rather than statistically.
test_that("predict() on a single row returns the batch prediction of that row (deep ensemble)", {
  set.seed(123)

  training = make_training_data()
  observed = make_observed()

  abc = fit_abcnn("deep ensemble", training, observed)

  # Reference: predict the three observed samples at once
  abc$predict()
  batch_mean = abc$predictive_mean
  batch_epistemic = abc$epistemic_uncertainty
  batch_aleatoric = abc$aleatoric_uncertainty
  batch_overall = abc$overall_uncertainty
  batch_predictions = abc$predictions()

  # The same second sample, this time on its own
  single_row = observed[2, , drop = FALSE]
  expect_equal(nrow(single_row), 1)

  abc$predict(single_row)

  # Shapes are kept, nothing collapsed to a vector
  expect_true(is.data.frame(abc$predictive_mean))
  expect_equal(dim(abc$predictive_mean), c(1, ncol(training$theta)))
  expect_equal(dim(abc$epistemic_uncertainty), c(1, ncol(training$theta)))
  expect_equal(dim(abc$aleatoric_uncertainty), c(1, ncol(training$theta)))
  expect_equal(dim(abc$overall_uncertainty), c(1, ncol(training$theta)))
  expect_equal(colnames(abc$predictive_mean), colnames(training$theta))
  expect_equal(abc$n_obs, 1)

  # Same numbers as in the batch prediction, up to the floating point noise of a
  # forward pass on a batch of 1 versus a batch of 3
  expect_equal(as.numeric(abc$predictive_mean[1, ]),
               as.numeric(batch_mean[2, ]),
               tolerance = mean_tolerance)
  expect_equal(as.numeric(abc$epistemic_uncertainty[1, ]),
               as.numeric(batch_epistemic[2, ]),
               tolerance = spread_tolerance)
  expect_equal(as.numeric(abc$aleatoric_uncertainty[1, ]),
               as.numeric(batch_aleatoric[2, ]),
               tolerance = spread_tolerance)
  expect_equal(as.numeric(abc$overall_uncertainty[1, ]),
               as.numeric(batch_overall[2, ]),
               tolerance = spread_tolerance)

  # The tidy predictions agree too, on the original parameter scale
  single_predictions = abc$predictions()

  expect_equal(nrow(single_predictions), ncol(training$theta))
  expect_true(all(single_predictions$sample == 1))
  expect_equal(single_predictions$parameter, colnames(training$theta))

  batch_row = batch_predictions[batch_predictions$sample == 2, ]
  expect_equal(single_predictions$parameter, batch_row$parameter)

  for (column in prediction_columns) {
    expect_equal(single_predictions[[column]], batch_row[[column]],
                 tolerance = column_tolerance(column),
                 info = paste("column", column))
  }

  # Predicting the same single row twice gives exactly the same answer. Both
  # passes have the same batch size here, so nothing is left to round differently.
  first = abc$predictions()
  abc$predict(single_row)
  second = abc$predictions()
  for (column in prediction_columns) {
    expect_equal(second[[column]], first[[column]],
                 info = paste("column", column))
  }
})


# For the sampling-based methods the prediction is a Monte Carlo average, so
# single-row and batch results cannot be identical. They must still agree well
# within the Monte Carlo error: the gap is compared to the epistemic standard
# deviation, of which the standard error of the mean is only a 1/sqrt(n) fraction.
test_that("predict() on a single row agrees with the batch prediction (monte carlo dropout)", {
  set.seed(123)

  training = make_training_data()
  observed = make_observed()

  abc = fit_abcnn("monte carlo dropout", training, observed,
                  num_posterior_samples = 2000)

  abc$predict()
  batch_mean = as.numeric(abc$predictive_mean[2, ])
  batch_sd = as.numeric(abc$epistemic_uncertainty[2, ])

  abc$predict(observed[2, , drop = FALSE])
  single_mean = as.numeric(abc$predictive_mean[1, ])

  expect_equal(dim(abc$predictive_mean), c(1, ncol(training$theta)))
  expect_true(all(is.finite(single_mean)))

  # Half a standard deviation is roughly fifteen times the standard error of the
  # difference between the two means with 2,000 posterior samples
  expect_true(all(abs(single_mean - batch_mean) < 0.5 * batch_sd))
})


# The remaining slots and methods that read them must survive a single row, for
# every method
methods = c("monte carlo dropout", "gaussian monte carlo dropout",
            "concrete dropout", "deep ensemble", "tabnet-abc")

for (method in methods) {
  test_that(paste0("predict() keeps a consistent shape on a single row for ", method), {
    set.seed(123)

    training = make_training_data()
    observed = make_observed()
    output_dim = ncol(training$theta)

    abc = fit_abcnn(method, training, observed)

    expect_no_error(abc$predict(observed[3, , drop = FALSE]))

    expect_equal(abc$n_obs, 1)
    expect_equal(dim(abc$predictive_mean), c(1, output_dim))
    expect_equal(dim(abc$epistemic_uncertainty), c(1, output_dim))
    expect_equal(dim(abc$aleatoric_uncertainty), c(1, output_dim))
    expect_equal(dim(abc$overall_uncertainty), c(1, output_dim))
    expect_equal(colnames(abc$predictive_mean), colnames(training$theta))

    expect_true(all(is.finite(as.matrix(abc$predictive_mean))))
    expect_true(all(is.finite(as.matrix(abc$epistemic_uncertainty))))
    expect_true(all(as.matrix(abc$epistemic_uncertainty) > 0))
    if (method %in% c("gaussian monte carlo dropout", "concrete dropout", "deep ensemble")) {
      expect_true(all(is.finite(as.matrix(abc$aleatoric_uncertainty))))
    }

    # The posterior samples of the sampling-based methods keep their observation axis
    if (method %in% c("monte carlo dropout", "tabnet-abc")) {
      expect_equal(dim(abc$posterior_samples)[2:3], c(1, output_dim))
    }
    if (method %in% c("gaussian monte carlo dropout", "concrete dropout")) {
      expect_equal(dim(abc$posterior_samples)[2:3], c(1, 2 * output_dim))
    }

    # One row per parameter for the single observed sample
    predictions = abc$predictions()

    expect_equal(nrow(predictions), output_dim)
    expect_true(all(predictions$sample == 1))
    expect_equal(predictions$parameter, colnames(training$theta))
    expect_true(all(is.finite(predictions$predictive_mean)))
    expect_true(all(predictions$epistemic_conformal_lower < predictions$epistemic_conformal_upper))
    expect_true(all(predictions$overall_conformal_lower < predictions$overall_conformal_upper))

    # The methods reading those slots must also handle the single row
    expect_no_error(abc$plot_prediction())
    expect_no_error(abc$plot_posterior())

    draws = abc$draw_from_posterior(n = 5)
    expect_equal(length(draws), 1)
    expect_equal(dim(draws[[1]]), c(5, output_dim))
    expect_equal(colnames(draws[[1]]), colnames(training$theta))
  })
}

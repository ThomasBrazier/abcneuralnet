# Test specific ABC methods implementations
methods = c("monte carlo dropout",
            "concrete dropout",
            "deep ensemble",
            "tabnet-abc")

for (m in methods) {
  test_that(paste0(m, " method works correctly"), {
    set.seed(123)
    n_samples = 10000
    theta_training = data.frame(param1 = runif(n_samples, 0, 1))
    sumstats_training = data.frame(
      stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.05),
      stat2 = theta_training$param1^2 + rnorm(n_samples, 0, 0.1)
    )
    sumstats_observed = data.frame(stat1 = 0.4, stat2 = 0.2)
    
    num_posterior_samples= 50 
    
    abc = abcnn$new(
      theta_training,
      sumstats_training,
      sumstats_observed,
      method = m,
      dropout = 0.3,
      batch_size = 128,
      epochs = 3,
      tol = 0.1,
      abc_method = "rejection",
      num_posterior_samples = num_posterior_samples,
      verbose = FALSE
    )
    
    # Test fitting
    expect_no_error(abc$fit())
    expect_true(abc$n_train == 7000)
    expect_true(abc$n_obs == 1)
    
    # Test prediction
    expect_no_error(abc$predict())
    expect_equal(dim(abc$predictive_mean), c(1, 1))
    expect_equal(dim(abc$epistemic_uncertainty), c(1, 1))
    
    # Test posterior sampling
    # expect_no_error(abc$posterior())
    if (m %in% c("monte carlo dropout")) {
      expect_equal(dim(abc$posterior_samples), c(num_posterior_samples, 1, 1))
    } else {
      if (m %in% c("concrete dropout")) {
        expect_equal(dim(abc$posterior_samples), c(num_posterior_samples, 1, 2))
      }
    }
    
    # Test credible intervals
    expect_no_error(abc$predictions())
    expect_equal(dim(abc$predictions()), c(1, 11))
  })
  
}

test_that("Method-specific parameters are validated", {
  set.seed(123)
  n_samples = 10000
  theta_training = data.frame(param1 = runif(n_samples, 0, 1))
  sumstats_training = data.frame(
    stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.05),
    stat2 = theta_training$param1^2 + rnorm(n_samples, 0, 0.1)
  )
  sumstats_observed = data.frame(stat1 = 0.4, stat2 = 0.2)
  
  # Test Monte Carlo Dropout parameter validation
  expect_error(
    abcnn$new(
      theta_training, sumstats_training, sumstats_observed,
      method = "monte carlo dropout", dropout = 0.8
    ),
    "The 'dropout' rate must be between 0.1 and 0.5."
  )
  
  expect_error(
    abcnn$new(
      theta_training, sumstats_training, sumstats_observed,
      method = "monte carlo dropout", dropout = -0.1
    ),
    "The 'dropout' rate must be between 0.1 and 0.5."
  )
  
})

test_that("Method-specific outputs are correct", {
  set.seed(123)
  n_samples = 10000
  theta_training = data.frame(param1 = runif(n_samples, 0, 1))
  sumstats_training = data.frame(
    stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.05),
    stat2 = theta_training$param1^2 + rnorm(n_samples, 0, 0.1)
  )
  sumstats_observed = data.frame(stat1 = 0.4, stat2 = 0.2)
  
  # Test Monte Carlo Dropout outputs
  abc_mc = abcnn$new(
    theta_training, sumstats_training, sumstats_observed,
    method = "monte carlo dropout", epochs = 2, verbose = FALSE
  )
  abc_mc$fit()
  abc_mc$predict()
  
  # MC dropout doesn't have aleatoric uncertainty
  expect_true(is.na(abc_mc$aleatoric_uncertainty))
  expect_true(!is.na(abc_mc$epistemic_uncertainty))
  expect_true(is.na(abc_mc$overall_uncertainty))
  
  
  # Test Concrete Dropout outputs
  abc_cd = abcnn$new(
    theta_training, sumstats_training, sumstats_observed,
    method = "concrete dropout", epochs = 2, verbose = FALSE
  )
  abc_cd$fit()
  abc_cd$predict()
  
  expect_true(!is.na(abc_cd$aleatoric_uncertainty))
  expect_true(!is.na(abc_cd$epistemic_uncertainty))
  expect_true(!is.na(abc_cd$overall_uncertainty))

  # Test Deep Ensemble outputs
  abc_de = abcnn$new(
    theta_training, sumstats_training, sumstats_observed,
    method = "deep ensemble", epochs = 2, verbose = FALSE
  )
  abc_de$fit()
  abc_de$predict()
  
  expect_true(!is.na(abc_de$aleatoric_uncertainty))
  expect_true(!is.na(abc_de$epistemic_uncertainty))
  expect_true(!is.na(abc_de$overall_uncertainty))
  
  # Test TabNet-ABC outputs
  abc_tab = abcnn$new(
    theta_training, sumstats_training, sumstats_observed,
    method = "tabnet-abc", tol = 0.1, epochs = 2, verbose = FALSE
  )
  abc_tab$fit()
  abc_tab$predict()
  
  expect_true(is.na(abc_tab$aleatoric_uncertainty))
  expect_true(!is.na(abc_tab$epistemic_uncertainty))
  expect_true(is.na(abc_tab$overall_uncertainty))
})


test_that("Methods handle different data dimensions correctly", {
  set.seed(123)
  n_samples = 100
  
  # Test 1D case
  theta_1d = data.frame(param1 = runif(n_samples, 0, 10))
  sumstats_1d = data.frame(stat1 = theta_1d$param1 + rnorm(n_samples, 0, 0.5))
  observed_1d = data.frame(stat1 = 5.2)
  
  methods_1d = c("monte carlo dropout", "concrete dropout", "deep ensemble")
  for (method in methods_1d) {
    abc = abcnn$new(
      theta_1d, sumstats_1d, observed_1d,
      method = method, epochs = 2, verbose = FALSE
    )
    abc$fit()
    abc$predict()
    
    expect_equal(dim(abc$predictive_mean), c(1, 1))
    expect_equal(dim(abc$predictive_variance), c(1, 1))
  }
  
  # Test multi-dimensional case
  theta_md = data.frame(
    param1 = runif(n_samples, 0, 10),
    param2 = runif(n_samples, -5, 5)
  )
  sumstats_md = data.frame(
    stat1 = theta_md$param1 + rnorm(n_samples, 0, 0.5),
    stat2 = theta_md$param2 + rnorm(n_samples, 0, 0.3),
    stat3 = theta_md$param1 * theta_md$param2 + rnorm(n_samples, 0, 1)
  )
  observed_md = data.frame(stat1 = 5.2, stat2 = 1.1, stat3 = 8.5)
  
  for (method in methods_1d) {
    abc = abcnn$new(
      theta_md, sumstats_md, observed_md,
      method = method, epochs = 2, verbose = FALSE
    )
    abc$fit()
    abc$predict()
    
    expect_equal(dim(abc$predictive_mean), c(1, 2))
    expect_equal(dim(abc$predictive_variance), c(1, 2))
  }
}
)
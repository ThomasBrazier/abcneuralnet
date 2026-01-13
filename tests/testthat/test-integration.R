# Integration tests for complete end-to-end workflows
test_that("Complete workflow works for Monte Carlo Dropout", {
  set.seed(42)
  
  # Generate realistic test data
  n_samples <- 200
  true_param <- 5.5
  
  # Simulate parameter values
  theta_training <- data.frame(param = runif(n_samples, 0, 10))
  
  # Simulate summary statistics with known relationship
  sumstats_training <- data.frame(
    stat1 = theta_training$param + rnorm(n_samples, 0, 0.3),
    stat2 = theta_training$param^2 + rnorm(n_samples, 0, 1),
    stat3 = sin(theta_training$param) + rnorm(n_samples, 0, 0.1)
  )
  
  # Observed statistics
  sumstats_observed <- data.frame(
    stat1 = true_param + rnorm(1, 0, 0.05),
    stat2 = true_param^2 + rnorm(1, 0, 0.1),
    stat3 = sin(true_param) + rnorm(1, 0, 0.02)
  )
  
  # Complete workflow
  abc <- abcnn$new(
    theta_training,
    sumstats_training,
    sumstats_observed,
    method = "monte carlo dropout",
    dropout = 0.2,
    scale_input = "minmax",
    scale_target = "minmax",
    num_hidden_layers = 3,
    num_hidden_dim = 64,
    batch_size = 32,
    learning_rate = 0.001,
    epochs = 10,
    validation_split = 0.2,
    early_stopping = TRUE,
    patience = 5,
    num_posterior_samples = 1000,
    verbose = FALSE
  )
  
  # Step 1: Fit model
  expect_no_error(abc$fit())
  expect_true(abc$trained)
  
  # Step 2: Make predictions
  expect_no_error(abc$predict())
  expect_equal(dim(abc$predictive_mean), c(1, 1))
  expect_equal(dim(abc$predictive_variance), c(1, 1))
  
  # Step 3: Generate posterior samples
  expect_no_error(abc$posterior())
  expect_equal(dim(abc$posterior_samples), c(1000, 1))
  
  # Step 4: Calculate credible intervals
  expect_no_error(abc$credible_interval())
  expect_equal(length(abc$credible_interval), 2)
  
  # Step 5: Generate plots
  expect_no_error(abc$plot_training())
  expect_no_error(abc$plot_prediction())
  expect_no_error(abc$plot_posterior())
  
  # Step 6: Get summary
  summary_result <- abc$summary()
  expect_s3_class(summary_result, "data.frame")
  
  # Step 7: Save and load model
  temp_prefix <- tempfile("mc_dropout_integration_")
  expect_no_error(save_abcnn(abc, prefix = temp_prefix))
  
  abc_loaded <- load_abcnn(prefix = temp_prefix)
  expect_s3_class(abc_loaded, "abcnn")
  expect_equal(abc_loaded$method, "monte carlo dropout")
  
  # Step 8: Verify loaded model works
  expect_no_error(abc_loaded$predict())
  expect_equal(dim(abc_loaded$predictive_mean), c(1, 1))
  
  # Step 9: Clean up
  unlink(paste0(temp_prefix, "*"))
  
  # Validate results
  expect_true(is.finite(abc$predictive_mean[1, 1]))
  expect_true(abc$predictive_variance[1, 1] > 0)
  expect_true(all(is.finite(abc$posterior_samples)))
  expect_true(abc$credible_interval[1] < abc$credible_interval[2])
})

test_that("Complete workflow works for Concrete Dropout with uncertainty quantification", {
  set.seed(42)
  
  # Generate test data
  n_samples <- 150
  theta_training <- data.frame(param1 = runif(n_samples, -5, 5), param2 = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(
    stat1 = theta_training$param1 + theta_training$param2 + rnorm(n_samples, 0, 0.5),
    stat2 = theta_training$param1^2 + rnorm(n_samples, 0, 1)
  )
  sumstats_observed <- data.frame(stat1 = 8.2, stat2 = 9.5)
  
  # Complete workflow with uncertainty quantification
  abc <- abcnn$new(
    theta_training,
    sumstats_training,
    sumstats_observed,
    method = "concrete dropout",
    weight_regularizer = 1e-6,
    dropout_regularizer = 1e-5,
    prior_length_scale = 1e-3,
    scale_input = "robustscaler",
    scale_target = "robustscaler",
    num_hidden_layers = 2,
    num_hidden_dim = 32,
    epochs = 8,
    num_posterior_samples = 500,
    num_conformal = 200,
    verbose = FALSE
  )
  
  # Complete workflow
  abc$fit()
  abc$predict()
  abc$posterior()
  abc$uncertainty_decomposition()
  abc$conformal_prediction()
  abc$credible_interval()
  
  # Validate uncertainty quantification
  expect_true("aleatoric" %in% names(abc$uncertainty))
  expect_true("epistemic" %in% names(abc$uncertainty))
  expect_true("conformal_bounds" %in% names(abc))
  
  # Check that uncertainties are reasonable
  expect_true(all(abc$uncertainty$aleatoric >= 0))
  expect_true(all(abc$uncertainty$epistemic >= 0))
  expect_true(length(abc$conformal_bounds) == 2)
  
  # Test plotting with uncertainty
  expect_no_error(abc$plot_training())
  expect_no_error(abc$plot_prediction())
  expect_no_error(abc$plot_posterior())
})

test_that("Complete workflow works for Deep Ensemble", {
  set.seed(42)
  
  # Generate test data
  n_samples <- 180
  theta_training <- data.frame(param = runif(n_samples, 0, 15))
  sumstats_training <- data.frame(
    stat1 = theta_training$param + rnorm(n_samples, 0, 0.4),
    stat2 = log(theta_training$param + 1) + rnorm(n_samples, 0, 0.2)
  )
  sumstats_observed <- data.frame(stat1 = 7.8, stat2 = 2.1)
  
  # Complete workflow with adversarial training
  abc <- abcnn$new(
    theta_training,
    sumstats_training,
    sumstats_observed,
    method = "deep ensemble",
    num_hidden_layers = 2,
    num_hidden_dim = 32,
    epsilon_adversarial = 0.01,
    variance_clamping = c(-1, 1),
    scale_input = "minmax",
    scale_target = "none",
    epochs = 6,
    num_posterior_samples = 300,
    num_conformal = 150,
    verbose = FALSE
  )
  
  # Complete workflow
  abc$fit()
  abc$predict()
  abc$posterior()
  abc$uncertainty_decomposition()
  abc$conformal_prediction()
  
  # Validate ensemble results
  expect_equal(dim(abc$predictive_mean), c(1, 1))
  expect_equal(dim(abc$predictive_variance), c(1, 1))
  expect_equal(dim(abc$posterior_samples), c(300, 1))
  
  # Check uncertainty decomposition
  expect_true("aleatoric" %in% names(abc$uncertainty))
  expect_true("epistemic" %in% names(abc$uncertainty))
  
  # Test ensemble-specific features
  expect_no_error(abc$plot_training())
  expect_no_error(abc$plot_prediction())
  expect_no_error(abc$plot_posterior())
})

test_that("Complete workflow works for TabNet-ABC with explanation", {
  set.seed(42)
  
  # Generate test data with more features for TabNet
  n_samples <- 300
  theta_training <- data.frame(param = runif(n_samples, 0, 20))
  
  # Create multiple summary statistics
  sumstats_training <- data.frame(
    stat1 = theta_training$param + rnorm(n_samples, 0, 0.5),
    stat2 = theta_training$param^2 + rnorm(n_samples, 0, 2),
    stat3 = sin(theta_training$param) + rnorm(n_samples, 0, 0.3),
    stat4 = cos(theta_training$param) + rnorm(n_samples, 0, 0.3),
    stat5 = log(theta_training$param + 1) + rnorm(n_samples, 0, 0.2)
  )
  
  sumstats_observed <- data.frame(
    stat1 = 10.5, stat2 = 112.3, stat3 = -0.88,
    stat4 = -0.48, stat5 = 2.4
  )
  
  # Complete TabNet-ABC workflow
  abc <- abcnn$new(
    theta_training,
    sumstats_training,
    sumstats_observed,
    method = "tabnet-abc",
    tol = 0.05,
    abc_method = "rejection",
    sampling = "rejection",
    kernel = "rbf",
    length_scale = 1.0,
    bandwidth = "max",
    scale_input = "none",
    scale_target = "none",
    epochs = 5,
    verbose = FALSE
  )
  
  # Step 1: Fit TabNet model
  expect_no_error(abc$fit())
  expect_true(abc$trained)
  
  # Step 2: Generate TabNet predictions
  expect_no_error(abc$predict())
  expect_true("tabnet_predictions" %in% names(abc))
  expect_equal(dim(abc$tabnet_predictions), c(n_samples, 5))
  
  # Step 3: Perform ABC inference
  expect_no_error(abc$abc_inference())
  expect_true(abc$abc_performed)
  expect_true("abc_posterior" %in% names(abc))
  
  # Step 4: Generate explanations
  expect_no_error(abc$explain_tabnet())
  expect_true("explanations" %in% names(abc))
  expect_true("feature_importance" %in% names(abc$explanations))
  
  # Step 5: Create TabNet-specific plots
  expect_no_error(abc$plot_tabnet_attention())
  expect_no_error(abc$plot_feature_importance())
  
  # Step 6: Test different ABC methods
  abc_methods <- c("loclinear", "neuralnet", "ridge")
  for (method in abc_methods) {
    abc$abc_method <- method
    expect_no_error(abc$abc_inference())
    expect_true(abc$abc_performed)
  }
  
  # Step 7: Test different kernels
  kernels <- c("epanechnikov")
  for (kernel in kernels) {
    abc$kernel <- kernel
    expect_no_error(abc$abc_inference())
    expect_true(abc$abc_performed)
  }
  
  # Validate results
  expect_true(all(is.finite(abc$tabnet_predictions)))
  expect_true(all(is.finite(abc$abc_posterior)))
  expect_true(nrow(abc$abc_posterior) > 0)
})

test_that("Multi-method comparison workflow works", {
  set.seed(42)
  
  # Generate test data
  n_samples <- 150
  theta_training <- data.frame(param = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(
    stat1 = theta_training$param + rnorm(n_samples, 0, 0.3),
    stat2 = theta_training$param^2 + rnorm(n_samples, 0, 1)
  )
  sumstats_observed <- data.frame(stat1 = 5.5, stat2 = 30.8)
  
  # Test all methods with same data
  methods <- c("monte carlo dropout", "concrete dropout", "deep ensemble")
  results <- list()
  
  for (method in methods) {
    abc <- abcnn$new(
      theta_training,
      sumstats_training,
      sumstats_observed,
      method = method,
      epochs = 5,
      num_posterior_samples = 200,
      verbose = FALSE
    )
    
    # Complete workflow
    abc$fit()
    abc$predict()
    abc$posterior()
    
    # Store results
    results[[method]] <- list(
      predictive_mean = abc$predictive_mean,
      predictive_variance = abc$predictive_variance,
      posterior_samples = abc$posterior_samples,
      method = method
    )
    
    # Test uncertainty methods if available
    if (method %in% c("concrete dropout", "deep ensemble")) {
      abc$uncertainty_decomposition()
      results[[method]]$uncertainty <- abc$uncertainty
    }
  }
  
  # Compare results
  expect_equal(length(results), 3)
  
  # Check that all methods produce reasonable results
  for (method in methods) {
    expect_true(is.finite(results[[method]]$predictive_mean[1, 1]))
    expect_true(results[[method]]$predictive_variance[1, 1] > 0)
    expect_equal(dim(results[[method]]$posterior_samples), c(200, 1))
  }
  
  # Test that methods give different results (they should)
  means <- sapply(results, function(x) x$predictive_mean[1, 1])
  expect_true(length(unique(means)) > 1)
})

test_that("Cross-validation workflow works", {
  set.seed(42)
  
  # Generate larger dataset for cross-validation
  n_samples <- 300
  theta_training <- data.frame(param = runif(n_samples, 0, 10))
  sumstats_training <- data.frame(
    stat1 = theta_training$param + rnorm(n_samples, 0, 0.3),
    stat2 = theta_training$param^2 + rnorm(n_samples, 0, 1)
  )
  sumstats_observed <- data.frame(stat1 = 5.5, stat2 = 30.8)
  
  # Split data for cross-validation
  train_idx <- sample(1:n_samples, size = 0.8 * n_samples)
  theta_train <- theta_training[train_idx, ]
  sumstats_train <- sumstats_training[train_idx, ]
  theta_val <- theta_training[-train_idx, ]
  sumstats_val <- sumstats_training[-train_idx, ]
  
  # Train model
  abc <- abcnn$new(
    theta_train,
    sumstats_train,
    sumstats_observed,
    method = "concrete dropout",
    epochs = 8,
    validation_split = 0.2,
    early_stopping = TRUE,
    patience = 3,
    verbose = FALSE
  )
  
  abc$fit()
  abc$predict()
  
  # Validate on hold-out set (conceptual test)
  expect_true(abc$trained)
  expect_true(is.finite(abc$predictive_mean[1, 1]))
  
  # Test model persistence in cross-validation context
  temp_prefix <- tempfile("cv_test_")
  save_abcnn(abc, prefix = temp_prefix)
  
  abc_cv <- load_abcnn(prefix = temp_prefix)
  expect_no_error(abc_cv$predict())
  
  unlink(paste0(temp_prefix, "*"))
})

test_that("Real-world simulation workflow works", {
  set.seed(42)
  
  # Simulate population genetics scenario
  n_samples <- 250
  
  # Simulate theta parameters (e.g., population size, mutation rate)
  theta_training <- data.frame(
    pop_size = runif(n_samples, 1000, 10000),
    mut_rate = runif(n_samples, 1e-8, 1e-6),
    sel_coeff = runif(n_samples, -0.1, 0.1)
  )
  
  # Simulate summary statistics (e.g., heterozygosity, allele frequencies)
  sumstats_training <- data.frame(
    het_obs = theta_training$pop_size * theta_training$mut_rate + rnorm(n_samples, 0, 0.01),
    pi = 4 * theta_training$pop_size * theta_training$mut_rate + rnorm(n_samples, 0, 0.02),
    tajd = -0.5 + 0.1 * theta_training$sel_coeff + rnorm(n_samples, 0, 0.1),
    fay_wu = theta_training$sel_coeff + rnorm(n_samples, 0, 0.05)
  )
  
  # "Observed" data from real population
  sumstats_observed <- data.frame(
    het_obs = 0.012, pi = 0.015, tajd = -0.2, fay_wu = 0.05
  )
  
  # Complete workflow with appropriate method
  abc <- abcnn$new(
    theta_training,
    sumstats_training,
    sumstats_observed,
    method = "deep ensemble",
    scale_input = "robustscaler",
    scale_target = "robustscaler",
    num_hidden_layers = 3,
    num_hidden_dim = 64,
    epochs = 10,
    num_posterior_samples = 1000,
    num_conformal = 300,
    verbose = FALSE
  )
  
  # Full workflow
  abc$fit()
  abc$predict()
  abc$posterior()
  abc$uncertainty_decomposition()
  abc$conformal_prediction()
  abc$credible_interval()
  
  # Validate multi-dimensional output
  expect_equal(dim(abc$predictive_mean), c(1, 3))
  expect_equal(dim(abc$predictive_variance), c(1, 3))
  expect_equal(dim(abc$posterior_samples), c(1000, 3))
  expect_equal(dim(abc$credible_interval), c(2, 3))
  
  # Test plotting for multi-dimensional case
  expect_no_error(abc$plot_training())
  expect_no_error(abc$plot_prediction())
  expect_no_error(abc$plot_posterior())
  
  # Test summary for multi-dimensional case
  summary_result <- abc$summary()
  expect_s3_class(summary_result, "data.frame")
  expect_equal(nrow(summary_result), 3)
  
  # Validate results are biologically plausible
  expect_true(all(abc$predictive_mean[1, ] > 0))  # Parameters should be positive
  expect_true(all(abc$predictive_variance[1, ] > 0))  # Variance should be positive
}
)
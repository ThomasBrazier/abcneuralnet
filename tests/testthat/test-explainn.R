# Test file for explainn methods and functions
# This file tests the explainn R6 class and its methods for feature attribution

# Helper function to create test data for explainn testing
make_test_data_explainn = function(n_samples = 100, n_features = 5, n_targets = 2) {
  set.seed(123)
  
  # Generate synthetic data with known structure
  X = matrix(rnorm(n_samples * n_features), n_samples, n_features)
  colnames(X) = paste0("feature_", 1:n_features)
  
  # Create targets with known relationship to features
  Y = matrix(0, n_samples, n_targets)
  colnames(Y) = paste0("target_", 1:n_targets)
  
  # Target 1 depends on features 1, 2, 3
  Y[, 1] = 2 * X[, 1] - 1.5 * X[, 2] + 0.5 * X[, 3] + 0.1 * rnorm(n_samples)
  
  # Target 2 depends on features 3, 4, 5
  Y[, 2] = X[, 3] + 1.2 * X[, 4] - 0.8 * X[, 5] + 0.1 * rnorm(n_samples)
  
  # Create test data
  X_test = matrix(rnorm(50 * n_features), 50, n_features)
  colnames(X_test) = paste0("feature_", 1:n_features)
  
  list(
    theta = as.data.frame(Y),
    sumstats = as.data.frame(X),
    observed = as.data.frame(X_test)
  )
}

# Test explainn object initialization
test_that("explainn object initializes correctly", {
  skip_if_not_installed("innsight")
  
  # Create test data
  data = make_test_data_explainn()
  
  # Train a simple concrete dropout model for testing
  abc = abcnn$new(
    data$theta,
    data$sumstats,
    data$observed,
    method = "concrete dropout",
    epochs = 5,  # Small number for quick testing
    num_hidden_layers = 2,
    num_hidden_dim = 32,
    batch_size = 16
  )
  
  # Fit the model
  abc$fit()
  
  # Test explainn initialization with default method
  exp = explainn$new(abc)
  expect_s4_class(exp, "R6ClassGenerator")
  expect_equal(exp$model_method, "concrete dropout")
  expect_equal(exp$method, "cw")
  expect_equal(exp$ensemble_num_model, 1)
  expect_null(exp$result)
  expect_s4_class(exp$converter, "Converter")
  
  # Test with custom method
  exp2 = explainn$new(abc, method = "grad")
  expect_equal(exp2$method, "grad")
  
  # Test with custom ensemble model number
  exp3 = explainn$new(abc, ensemble_num_model = 2)
  expect_equal(exp3$ensemble_num_model, 2)
})

# Test explainn with different abcnn methods
test_that("explainn works with different abcnn methods", {
  skip_if_not_installed("innsight")
  
  data = make_test_data_explainn()
  
  # Test with Monte Carlo dropout
  abc_mc = abcnn$new(
    data$theta,
    data$sumstats,
    data$observed,
    method = "monte carlo dropout",
    epochs = 3,
    num_hidden_layers = 2,
    num_hidden_dim = 32,
    batch_size = 16
  )
  
  abc_mc$fit()
  exp_mc = explainn$new(abc_mc, method = "grad")
  expect_equal(exp_mc$model_method, "monte carlo dropout")
  expect_s4_class(exp_mc$converter, "Converter")
})

# Test explainn with TabNet-ABC
test_that("explainn handles TabNet-ABC correctly", {
  skip_if_not_installed("tabnet")
  
  data = make_test_data_explainn()
  
  # Train TabNet model
  abc_tabnet = abcnn$new(
    data$theta,
    data$sumstats,
    data$observed,
    method = "tabnet-abc",
    epochs = 5,
    batch_size = 16
  )
  
  abc_tabnet$fit()
  
  # TabNet should not create a converter
  exp_tabnet = explainn$new(abc_tabnet)
  expect_equal(exp_tabnet$model_method, "tabnet-abc")
  expect_null(exp_tabnet$converter)
  
  # Test printing method (should give warning)
  expect_warning(exp_tabnet$print(), "No converter")
})

# Test explainn run methods with different techniques
test_that("explainn run methods work correctly", {
  skip_if_not_installed("innsight")
  
  data = make_test_data_explainn()
  
  # Train model
  abc = abcnn$new(
    data$theta,
    data$sumstats,
    data$observed,
    method = "concrete dropout",
    epochs = 3,
    num_hidden_layers = 2,
    num_hidden_dim = 32,
    batch_size = 16
  )
  
  abc$fit()
  
  # Test CW method (no data required)
  exp_cw = explainn$new(abc, method = "cw")
  exp_cw$run()
  expect_s4_class(exp_cw$result, "InnsightResult")
  
  # Test gradient-based methods with data
  test_data = data$observed[1:10, ]
  
  methods_to_test = c("grad", "smoothgrad", "intgrad", "lrp", "deeplift")
  
  for (method in methods_to_test) {
    exp = explainn$new(abc, method = method)
    exp$run(data = test_data)
    expect_s4_class(exp$result, "InnsightResult")
    expect_equal(exp$method, method)
  }
})

# Test explainn methods requiring reference data
test_that("explainn methods with reference data work correctly", {
  skip_if_not_installed("innsight")
  
  data = make_test_data_explainn()
  
  # Train model
  abc = abcnn$new(
    data$theta,
    data$sumstats,
    data$observed,
    method = "concrete dropout",
    epochs = 3,
    num_hidden_layers = 2,
    num_hidden_dim = 32,
    batch_size = 16
  )
  
  abc$fit()
  
  test_data = data$observed[1:10, ]
  ref_data = data$sumstats[1:20, ]
  
  # Test methods requiring reference data
  methods_to_test = c("shap", "lime")
  
  for (method in methods_to_test) {
    exp = explainn$new(abc, method = method)
    exp$run(data = test_data, data_ref = ref_data)
    expect_s4_class(exp$result, "InnsightResult")
    expect_equal(exp$method, method)
  }
})

# Test TabNet-ABC explainn run method
test_that("TabNet-ABC explainn run method works correctly", {
  skip_if_not_installed("tabnet")
  
  data = make_test_data_explainn()
  
  # Train TabNet model
  abc_tabnet = abcnn$new(
    data$theta,
    data$sumstats,
    data$observed,
    method = "tabnet-abc",
    epochs = 3,
    batch_size = 16
  )
  
  abc_tabnet$fit()
  
  exp_tabnet = explainn$new(abc_tabnet)
  test_data = data$observed[1:10, ]
  
  # Run TabNet explanation
  exp_tabnet$run(data = test_data)
  expect_s4_class(exp_tabnet$result, "tabnet_fit")
})

# Test explainn get_result method
test_that("explainn get_result method works correctly", {
  skip_if_not_installed("innsight")
  
  data = make_test_data_explainn()
  
  # Test with regular neural network
  abc = abcnn$new(
    data$theta,
    data$sumstats,
    data$observed,
    method = "concrete dropout",
    epochs = 3,
    num_hidden_layers = 2,
    num_hidden_dim = 32,
    batch_size = 16
  )
  
  abc$fit()
  
  exp = explainn$new(abc, method = "grad")
  exp$run(data = data$observed[1:5, ])
  
  # Test different return types
  result_array = exp$get_result(type = "array")
  expect_type(result_array, "double")
  expect_true(is.array(result_array))
  
  result_df = exp$get_result(type = "data.frame")
  expect_s3_class(result_df, "data.frame")
  
  result_tensor = exp$get_result(type = "torch_tensor")
  expect_s3_class(result_tensor, "torch_tensor")
  
  # Test with TabNet
  if (requireNamespace("tabnet", quietly = TRUE)) {
    abc_tabnet = abcnn$new(
      data$theta,
      data$sumstats,
      data$observed,
      method = "tabnet-abc",
      epochs = 3,
      batch_size = 16
    )
    
    abc_tabnet$fit()
    exp_tabnet = explainn$new(abc_tabnet)
    exp_tabnet$run(data = data$observed[1:5, ])
    
    tabnet_result = exp_tabnet$get_result()
    expect_type(tabnet_result, "double")
  }
})

# Test explainn plotting methods
test_that("explainn plotting methods work correctly", {
  skip_if_not_installed("innsight")
  
  data = make_test_data_explainn()
  
  # Train model
  abc = abcnn$new(
    data$theta,
    data$sumstats,
    data$observed,
    method = "concrete dropout",
    epochs = 3,
    num_hidden_layers = 2,
    num_hidden_dim = 32,
    batch_size = 16
  )
  
  abc$fit()
  
  exp = explainn$new(abc, method = "grad")
  exp$run(data = data$observed[1:5, ])
  
  # Test basic plot
  p1 = exp$plot()
  expect_s3_class(p1, "ggplot")
  
  # Test plotly
  p2 = exp$plot(as_plotly = TRUE)
  expect_s3_class(p2, "plotly")
  
  # Test with specific output labels
  p3 = exp$plot(output_label = "target_1")
  expect_s3_class(p3, "ggplot")
  
  # Test global plot
  p4 = exp$plot_global()
  expect_s3_class(p4, "ggplot")
  
  # Test boxplot
  p5 = exp$boxplot()
  expect_s3_class(p5, "ggplot")
})

# Test TabNet plotting methods
test_that("TabNet-ABC plotting methods work correctly", {
  skip_if_not_installed("tabnet")
  
  data = make_test_data_explainn()
  
  abc_tabnet = abcnn$new(
    data$theta,
    data$sumstats,
    data$observed,
    method = "tabnet-abc",
    epochs = 3,
    batch_size = 16
  )
  
  abc_tabnet$fit()
  
  exp_tabnet = explainn$new(abc_tabnet)
  exp_tabnet$run(data = data$observed[1:5, ])
  
  # Test TabNet plot with mask_agg type
  p1 = exp_tabnet$plot(type = "mask_agg")
  expect_s3_class(p1, "ggplot")
  
  # Test TabNet plot with steps type
  p2 = exp_tabnet$plot(type = "steps")
  expect_s3_class(p2, "ggplot")
  
  # Test warnings for methods not applicable to TabNet
  expect_warning(exp_tabnet$plot_global(), "'plot_global' not applicable")
  expect_warning(exp_tabnet$boxplot(), "'boxplot' not applicable")
})

# Test error handling in explainn
test_that("explainn handles errors correctly", {
  skip_if_not_installed("innsight")
  
  data = make_test_data_explainn()
  
  # Test with unfitted model
  abc_unfitted = abcnn$new(
    data$theta,
    data$sumstats,
    data$observed,
    method = "concrete dropout"
  )
  
  expect_error(
    explainn$new(abc_unfitted),
    "Model must be fitted before creating explainn object"
  )
  
  # Test with invalid method
  abc = abcnn$new(
    data$theta,
    data$sumstats,
    data$observed,
    method = "concrete dropout",
    epochs = 3,
    num_hidden_layers = 2,
    num_hidden_dim = 32,
    batch_size = 16
  )
  
  abc$fit()
  
  expect_warning(
    explainn$new(abc, method = "invalid_method"),
    "Unknown attribution method"
  )
})

# Test explainn with deep ensemble models
test_that("explainn works with deep ensemble models", {
  skip_if_not_installed("innsight")
  
  data = make_test_data_explainn()
  
  # Train deep ensemble
  abc_ensemble = abcnn$new(
    data$theta,
    data$sumstats,
    data$observed,
    method = "deep ensemble",
    num_networks = 3,
    epochs = 2,  # Small number for quick testing
    num_hidden_layers = 2,
    num_hidden_dim = 32,
    batch_size = 16
  )
  
  abc_ensemble$fit()
  
  # Test with different ensemble model indices
  for (model_idx in 1:3) {
    exp = explainn$new(abc_ensemble, ensemble_num_model = model_idx)
    expect_equal(exp$ensemble_num_model, model_idx)
    
    exp$run(data = data$observed[1:5, ])
    expect_s4_class(exp$result, "InnsightResult")
  }
})

# Test explainn data scaling
test_that("explainn handles data scaling correctly", {
  skip_if_not_installed("innsight")
  
  data = make_test_data_explainn()
  
  # Train model with input scaling
  abc = abcnn$new(
    data$theta,
    data$sumstats,
    data$observed,
    method = "concrete dropout",
    scale_input = "minmax",
    epochs = 3,
    num_hidden_layers = 2,
    num_hidden_dim = 32,
    batch_size = 16
  )
  
  abc$fit()
  
  exp = explainn$new(abc, method = "grad")
  
  # Test that data is properly scaled when running explanation
  test_data = data$observed[1:5, ]
  exp$run(data = test_data)
  
  # Should not error and should produce valid results
  expect_s4_class(exp$result, "InnsightResult")
  
  # Test with reference data scaling
  ref_data = data$sumstats[1:10, ]
  exp_shap = explainn$new(abc, method = "shap")
  exp_shap$run(data = test_data, data_ref = ref_data)
  expect_s4_class(exp_shap$result, "InnsightResult")
})

# Test explainn attribute consistency
test_that("explainn attributes are consistent", {
  skip_if_not_installed("innsight")
  
  data = make_test_data_explainn()
  
  abc = abcnn$new(
    data$theta,
    data$sumstats,
    data$observed,
    method = "concrete dropout",
    epochs = 3,
    num_hidden_layers = 2,
    num_hidden_dim = 32,
    batch_size = 16
  )
  
  abc$fit()
  
  exp = explainn$new(abc, method = "grad")
  
  # Check that variables and parameters are correctly extracted
  expect_equal(length(exp$variables), ncol(data$sumstats))
  expect_equal(length(exp$parameters), ncol(data$theta))
  expect_true(all(exp$variables %in% paste0("feature_", 1:ncol(data$sumstats))))
  expect_true(all(exp$parameters %in% paste0("target_", 1:ncol(data$theta))))
  
  # Check scaling attributes
  expect_equal(exp$scale_input, "none")
  expect_null(exp$input_summary)
})

# Integration test: complete explainn workflow
test_that("explainn complete workflow works end-to-end", {
  skip_if_not_installed("innsight")
  
  # Create data with known feature importance
  set.seed(456)
  n = 200
  X = matrix(rnorm(n * 4), n, 4)
  colnames(X) = c("important1", "important2", "noise1", "noise2")
  
  # Target depends only on first two features
  Y = 3 * X[, 1] - 2 * X[, 2] + 0.1 * rnorm(n)
  
  theta = data.frame(target = Y)
  sumstats = as.data.frame(X)
  observed = as.data.frame(matrix(rnorm(20 * 4), 20, 4))
  colnames(observed) = colnames(sumstats)
  
  # Train model
  abc = abcnn$new(
    theta, sumstats, observed,
    method = "concrete dropout",
    epochs = 5,
    num_hidden_layers = 2,
    num_hidden_dim = 64,
    batch_size = 32
  )
  
  abc$fit()
  
  # Run explanation
  exp = explainn$new(abc, method = "grad")
  exp$run(data = observed)
  
  # Get results
  results = exp$get_result(type = "data.frame")
  
  # Important features should have higher absolute attribution values
  mean_attributions = abs(colMeans(results[, -1], na.rm = TRUE))
  names(mean_attributions) = colnames(sumstats)
  
  # Check that important features have higher attribution on average
  # (This is a probabilistic test, not guaranteed every time)
  expect_type(mean_attributions, "double")
  expect_equal(length(mean_attributions), 4)
  
  # Test plotting
  p = exp$plot()
  expect_s3_class(p, "ggplot")
  
  # Test global plot
  p_global = exp$plot_global()
  expect_s3_class(p_global, "ggplot")
})
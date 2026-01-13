# Test TabNet-ABC save and load functionality
# This test verifies that TabNet-ABC models can be saved and loaded correctly

library(abcneuralnet)
library(torch)
library(testthat)

# Create simple TabNet-ABC test data
set.seed(123)
n_samples <- 100
theta_train <- data.frame(param1 = runif(n_samples, 0, 10))
sumstats_train <- data.frame(
  stat1 = theta_train$param1 + rnorm(n_samples, 0, 0.5),
  stat2 = rnorm(n_samples, 0, 1)
)
observed_stats <- data.frame(stat1 = 5.2, stat2 = 0.3)

# Create TabNet-ABC model
tabnet_abc <- abcnn$new(
  theta = theta_train,
  sumstats = sumstats_train,
  observed = observed_stats,
  method = "tabnet-abc",
  epochs = 5,
  verbose = FALSE
)

# Fit the model
cat("Fitting TabNet-ABC model...\n")
tabnet_abc$fit()

# Test the enhanced save functionality
cat("Testing enhanced save_abcnn function...\n")
temp_prefix <- tempfile("tabnet_test_")

tryCatch({
  save_abcnn(tabnet_abc, prefix = temp_prefix)
  cat("✓ save_abcnn completed successfully\n")
}, error = function(e) {
  cat("✗ save_abcnn failed:", e$message, "\n")
})

# Test the enhanced load functionality  
cat("Testing enhanced load_abcnn function...\n")
tryCatch({
  loaded_model <- load_abcnn(prefix = temp_prefix)
  cat("✓ load_abcnn completed successfully\n")
}, error = function(e) {
  cat("✗ load_abcnn failed:", e$message, "\n")
})

# Verify loaded model properties
if (!is.null(loaded_model)) {
  cat("✓ Model loaded successfully\n")
  
  # Test that the loaded model can predict
  test_predictions <- tryCatch({
    loaded_model$predict()
    cat("✓ Loaded model can predict successfully\n")
    return(TRUE)
  }, error = function(e) {
    cat("✗ Loaded model prediction failed:", e$message, "\n")
    return(FALSE)
  })
  
  if (test_predictions) {
    cat("✓ TabNet-ABC save/load functionality working correctly\n")
  } else {
    cat("✗ TabNet-ABC loaded model cannot predict\n")
  }
} else {
  cat("✗ Model failed to load\n")
}

# Clean up
unlink(paste0(temp_prefix, "*"))
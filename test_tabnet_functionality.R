# Simple test to verify TabNet-ABC functionality
# Run this from the package directory to test actual functions

source("R/utils.R")
source("R/abcnn.R")

# Create minimal test data
theta_test <- data.frame(param1 = 1:5)
sumstats_test <- data.frame(stat1 = 1:5)
observed_test <- data.frame(stat1 = 3)

# Create TabNet-ABC model (minimal)
abc_test <- abcnn$new(
  theta = theta_test,
  sumstats = sumstats_test,
  observed = observed_test,
  method = "tabnet-abc",
  epochs = 2,
  verbose = FALSE
)

# Fit model
cat("Fitting model...\n")
abc_test$fit()

# Test save functionality
cat("Testing save_abcnn...\n")
temp_prefix <- tempdir()

tryCatch({
  save_abcnn(abc_test, prefix = file.path(temp_prefix, "test_abc"))
  cat("✓ Save successful\n")
}, error = function(e) {
  cat("✗ Save failed:", e$message, "\n")
})

# Test load functionality
cat("Testing load_abcnn...\n")
tryCatch({
  loaded_abc <- load_abcnn(prefix = file.path(temp_prefix, "test_abc"))
  cat("✓ Load successful\n")
  
  # Test prediction capability
  loaded_abc$predict()
  cat("✓ Loaded model can predict\n")
  
}, error = function(e) {
  cat("✗ Load failed:", e$message, "\n")
})

# Test that loaded model has correct properties
if (!is.null(loaded_abc) && loaded_abc$method == "tabnet-abc") {
  cat("✓ TabNet-ABC save/load working correctly\n")
} else {
  cat("✗ TabNet-ABC save/load failed\n")
}

# Cleanup
unlink(file.path(temp_prefix, "test_abc*"))
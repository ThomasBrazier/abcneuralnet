# AGENTS.md - Development Guidelines for abcneuralnet

This file contains guidelines for agentic coding assistants working on the abcneuralnet R package.

## Project Overview

**Package**: abcneuralnet - Bayesian Deep Learning and Approximate Bayesian Computation for parameter inference in population genetics
**Language**: R (with torch backend)
**License**: GPL-3.0
**Status**: Development version (0.1), not yet on CRAN

## Build, Test, and Development Commands

### Package Building and Checking
```bash
# Build the package
R CMD build .

# Check the package (comprehensive)
R CMD check abcneuralnet_0.1.tar.gz

# Install development version
devtools::install(".", dependencies = TRUE)

# Install with specific arguments (from .Rproj)
devtools::install(".", args = "--no-multiarch --with-keep.source")
```

### Testing Commands
```bash
# Run all tests
devtools::test()

# Run specific test file
devtools::test("tests/testthat/test-abcnn.R")

# Run tests with coverage
devtools::test_coverage()

# Run tests with specific filter
testthat::test_file("tests/testthat/test-abcnn.R")
```

### Documentation
```bash
# Generate documentation
devtools::document()

# Build vignettes
devtools::build_vignettes()
```

### Code Quality
```bash
# No explicit linting configuration found
# Consider using lintr for code linting:
# install.packages("lintr")
# lintr::lint_package()
```

## Code Style Guidelines

### General Formatting
- **Indentation**: 2 spaces (configured in .Rproj)
- **Line endings**: Posix (Unix-style)
- **Encoding**: UTF-8
- **Trailing whitespace**: Stripped automatically
- **Newlines**: Auto-appended at end of files

### Naming Conventions
- **Functions**: snake_case (e.g., `save_abcnn`, `load_abcnn`)
- **Variables**: snake_case (e.g., `data_x1a`, `num_hidden_layers`)
- **Classes**: PascalCase for R6 classes (e.g., `abcnn`)
- **Constants**: UPPER_SNAKE_CASE
- **Files**: snake_case.R (e.g., `abcnn.R`, `utils.R`)

### R6 Class Structure
- Use R6 for object-oriented programming
- Public methods should be documented with roxygen2
- Private methods use `private$` prefix
- Active bindings for computed properties
- Initialize method for constructor logic

### Import Organization
```r
# @import statements at top of file
#' @import torch
#' @import luz
#' @import R6

# Namespace imports in NAMESPACE (auto-generated)
```

### Documentation Style
- Use roxygen2 for all exported functions
- Include `@param`, `@return`, `@export`, `@description`
- Use markdown formatting in roxygen comments
- Cross-reference functions with backticks
- Include examples where appropriate

### Error Handling
- Use `assertthat` for input validation
- Provide informative error messages
- Use `stop()` for critical errors
- Use `warning()` for non-critical issues
- Handle torch-specific errors gracefully

### Type Safety
- Use type hints in roxygen2 documentation
- Validate input types with `assertthat::assert_that()`
- Convert between data structures explicitly
- Handle NA/NULL values appropriately

## Package Structure

### Source Organization
```
R/
├── abcnn.R              # Main R6 class
├── concrete_dropout.R   # Concrete dropout implementation
├── deep_ensemble.R      # Deep ensemble method
├── mc_dropout.R         # Monte Carlo dropout
├── tabnet.R            # TabNet-ABC method
├── abc_sampling.R      # ABC sampling functions
├── explainn.R          # Model explanation utilities
└── utils.R             # General utilities (save/load, scaling)
```

### Test Structure
```
tests/testthat/
├── test-abcnn.R        # Core functionality tests
├── test-utils.R        # Utility function tests
└── test-save_load.R    # Model persistence tests
```

## Dependencies and Imports

### Core Dependencies
- **torch**: Deep learning framework
- **luz**: High-level API for torch
- **R6**: Object-oriented programming
- **abc**: Approximate Bayesian Computation
- **tabnet**: Tabular neural networks

### Data Manipulation
- **tidyverse**: Data manipulation and visualization
- **dplyr**: Data wrangling
- **ggplot2**: Visualization
- **plotly**: Interactive plots

### Utilities
- **assertthat**: Input validation
- **bundle**: Model serialization
- **here**: File path management

## Development Workflow

### Making Changes
1. Modify source files in `R/`
2. Update tests in `tests/testthat/`
3. Run `devtools::document()` for docs
4. Run `devtools::test()` for testing
5. Check package with `R CMD check`

### Adding New Methods
1. Create new R file in `R/`
2. Implement as R6 class or function
3. Add comprehensive tests
4. Update documentation
5. Add to NAMESPACE if needed

### Model Serialization
- Use `bundle::bundle()` for torch models
- Save fitted models with `luz::luz_save()`
- Implement `save_abcnn()` and `load_abcnn()` for persistence
- Handle TabNet serialization with comprehensive error handling and fallback mechanisms
- TabNet-ABC specific improvements include:
  - Primary method: saves torch model directly and bundles fitted object
  - Fallback method: uses standard luz save/load if primary fails
  - Enhanced error handling with informative warnings
  - Validation of loaded objects before unbundling
  - File existence checking before loading attempts

## Testing Guidelines

### Test Structure
- Use `testthat` framework (edition 3)
- Organize tests by functionality
- Use descriptive test names
- Test edge cases and error conditions

### Test Data
- Use reproducible synthetic data
- Store test data in `tests/data/`
- Keep test datasets small
- Test multiple input dimensions

### Coverage
- Aim for high test coverage
- Test all public methods
- Include integration tests
- Test model serialization

## Performance Considerations

### GPU Support
- Automatically detect CUDA via luz
- Handle CPU fallback gracefully
- Test on both CPU and GPU when possible

### Memory Management
- Use torch for efficient tensor operations
- Clean up intermediate tensors
- Monitor memory usage in training loops

### Parallel Processing
- Use `ncores` parameter for parallel operations
- Implement parallel ABC sampling where appropriate
- Consider memory overhead of parallelization

## Common Patterns

### Model Training
```r
# Standard luz training pattern
fitted <- model %>% 
  luz::setup_loss(loss) %>% 
  luz::setup_optimizer(optimizer) %>% 
  luz::fit(
    data = dataloader,
    epochs = epochs,
    callbacks = list(...)
  )
```

### Data Scaling
```r
# Input/target scaling options
scale_input = c("none", "minmax", "robustscaler")
scale_target = c("none", "minmax", "robustscaler")
```

### ABC Methods
- Four main methods: "monte carlo dropout", "concrete dropout", "deep ensemble", "tabnet-abc"
- Each method has specific uncertainty quantification
- Conformal prediction available for uncertainty methods

## CI/CD Integration

### GitHub Actions
- Automated R CMD check on Ubuntu
- Torch installation in CI pipeline
- Test execution on each push/PR
- Snapshot upload for debugging

### Local Development
- Use `devtools::check()` for comprehensive checking
- Test on multiple R versions if possible
- Verify torch installation separately

## Notes for AI Assistants

1. **Torch Integration**: This package heavily uses R torch - be careful with tensor operations and device management
2. **Scientific Computing**: Code is for research - prioritize correctness and reproducibility
3. **Uncertainty Quantification**: Multiple methods for uncertainty - understand the differences
4. **Model Persistence**: Serialization is complex due to torch models - use provided save/load functions
5. **Testing**: Always test with synthetic data before real applications
6. **Documentation**: Maintain comprehensive roxygen2 documentation
7. **Error Handling**: Provide clear error messages for scientific users
8. **Performance**: Consider computational efficiency for population genetics applications
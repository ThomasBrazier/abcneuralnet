# conda-forge recipe installation instructions

## Overview
This directory contains conda-forge recipes for building and distributing the `abcneuralnet` R package through conda.

## Package Information
- **Package Name**: `r-abcneuralnet`
- **Version**: 0.1
- **License**: GPL-3.0
- **Primary Use**: Bayesian Deep Learning and Approximate Bayesian Computation for parameter inference

## Prerequisites
- conda-build package
- conda-forge channel access
- R (>= 4.0) toolchain

## Installation Methods

### Method 1: Direct Installation from Conda-Forge (when available)
```bash
conda install -c conda-forge r-abcneuralnet
```

### Method 2: Local Build from Recipe

#### Step 1: Clone the repository
```bash
git clone https://github.com/ThomasBrazier/abcneuralnet.git
cd abcneuralnet/conda-recipe
```

#### Step 2: Build the package
```bash
# Build the conda package
conda build .

# Or build for specific platforms
conda build . --python=3.9
conda build . --python=3.10
```

#### Step 3: Install locally
```bash
# Install the built package
conda install --use-local r-abcneuralnet
```

### Method 3: Create Environment with Dependencies Only
If you prefer to install the R package directly from GitHub but want conda-managed dependencies:

```bash
# Create environment with all dependencies
conda create -n abcneuralnet-env \
    r-base=4.3 \
    r-r6 \
    r-rcolorbrewer \
    r-rdpack \
    r-knitr \
    r-abc \
    r-bundle \
    r-dplyr \
    r-ggplot2 \
    r-ggpubr \
    r-innsight \
    r-janitor \
    r-luz \
    r-plotly \
    r-tabnet \
    r-tibble \
    r-tidyr \
    r-tidyverse \
    r-torch \
    r-devtools \
    r-assertthat \
    r-here

# Activate environment
conda activate abcneuralnet-env

# Install package from GitHub
R -e "devtools::install_github('ThomasBrazier/abcneuralnet')"
```

## Key Dependencies

### Core Dependencies
- **R (>= 4.0)**: Base R system
- **torch**: Deep learning framework
- **luz**: High-level API for torch
- **abc**: Approximate Bayesian Computation
- **tabnet**: Tabular neural networks
- **innsight**: Feature attribution methods

### Data Science Dependencies
- **tidyverse**: Data manipulation and visualization
- **ggplot2**: Grammar of graphics
- **plotly**: Interactive plots
- **dplyr**: Data wrangling
- **tibble**: Modern data frames

### System Dependencies
- **R6**: Object-oriented programming
- **devtools**: Development tools
- **assertthat**: Input validation
- **here**: File path management

## GPU Support

### CUDA Support
The package supports GPU acceleration through torch CUDA backend:

```bash
# Create environment with CUDA support (if available)
conda create -n abcneuralnet-gpu \
    r-base=4.3 \
    r-torch \
    pytorch-cuda=11.8 \
    cuda-toolkit=11.8 \
    r-abcneuralnet
```

### CPU-Only Installation
For systems without GPU support:

```bash
conda install -c conda-forge r-abcneuralnet
# Package will automatically detect and use CPU
```

## Testing

### Run Package Tests
```bash
# After installation
conda activate abcneuralnet-env
R -e "testthat::test_package('abcneuralnet')"
```

### Build Testing
```bash
# Test build process without installing
conda build . --test
```

## Platform Support

### Supported Platforms
- **Linux**: x86_64, aarch64
- **macOS**: x86_64, arm64 (Apple Silicon)
- **Windows**: x86_64

### Architecture-Specific Notes
- **macOS Apple Silicon**: Use `osx-arm64` builds for better performance
- **Windows**: Requires Rtools40 toolchain
- **Linux**: Most stable platform with full GPU support

## Environment Files

### Minimal Environment
```yaml
# environment.yml
name: abcneuralnet-minimal
channels:
  - conda-forge
dependencies:
  - r-base>=4.0
  - r-abcneuralnet
```

### Development Environment
```yaml
# environment-dev.yml
name: abcneuralnet-dev
channels:
  - conda-forge
  - pytorch
dependencies:
  - r-base=4.3
  - r-abcneuralnet
  - r-devtools
  - r-testthat
  - r-roxygen2
  - python=3.10
  - pytorch
  - cuda-toolkit=11.8  # [linux64]
  - jupyterlab
  - nb_conda_kernels
```

## Troubleshooting

### Common Issues

#### 1. Torch Installation Problems
```bash
# Install torch separately first
conda install -c conda-forge r-torch

# Then install abcneuralnet
conda install -c conda-forge r-abcneuralnet
```

#### 2. CUDA Compatibility
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch CUDA toolkit
conda install pytorch-cuda=11.8  # Adjust version as needed
```

#### 3. Memory Issues
```bash
# Increase build memory
conda build . --memory 8G
```

#### 4. Permission Issues (Linux/macOS)
```bash
# Use user installation
R -e "devtools::install_local('.', dependencies = TRUE)"
```

### Verification Commands
```bash
# Check package installation
R -e "library(abcneuralnet); packageVersion('abcneuralnet')"

# Check torch backend
R -e "torch::torch_is_installed(); torch::torch_cuda_is_available()"

# Test basic functionality
R -e "library(abcneuralnet); data(mtcars); head(abcnn:::make_test_data())"
```

## Contributing to Conda-Forge

### Submitting to conda-forge
1. Fork the conda-forge feedstock repository
2. Submit a pull request with updated recipe
3. Follow conda-forge review process

### Recipe Maintenance
- Update version numbers for new releases
- Add new dependencies as needed
- Test on all supported platforms
- Update documentation for breaking changes

## Additional Resources

### Documentation
- **Package Website**: https://ThomasBrazier.github.io/abcneuralnet/
- **GitHub Repository**: https://github.com/ThomasBrazier/abcneuralnet
- **Conda-Forge**: https://anaconda.org/conda-forge/r-abcneuralnet

### Related Packages
- **torch**: https://torch.mlverse.org/
- **luz**: https://luz.mlverse.org/
- **tabnet**: https://github.com/dream-faster/r-tabnet
- **innsight**: https://bips-hb.github.io/innsight/

### Support
- **Issues**: https://github.com/ThomasBrazier/abcneuralnet/issues
- **Discussions**: https://github.com/ThomasBrazier/abcneuralnet/discussions
- **Conda-Forge Issues**: https://github.com/conda-forge/r-abcneuralnet-feedstock/issues
#!/bin/bash

# Conda installation script for abcneuralnet

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed or not in PATH"
        print_status "Please install Miniconda or Anaconda first:"
        echo "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
        echo "  - Anaconda: https://www.anaconda.com/products/distribution"
        exit 1
    fi
    
    print_status "Found conda installation: $(conda --version)"
}

# Create conda environment
create_environment() {
    ENV_NAME=${1:-abcneuralnet}
    
    print_status "Creating conda environment: $ENV_NAME"
    
    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        print_warning "Environment '$ENV_NAME' already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing existing environment..."
            conda env remove -n $ENV_NAME -y
        else
            print_status "Using existing environment"
            conda activate $ENV_NAME
            return 0
        fi
    fi
    
    # Create environment with dependencies
    conda create -n $ENV_NAME \
        -c conda-forge \
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
        r-here \
        -y
    
    print_status "Environment created successfully"
}

# Install abcneuralnet
install_package() {
    METHOD=${1:-conda-forge}
    
    print_status "Installing abcneuralnet using method: $METHOD"
    
    case $METHOD in
        "conda-forge")
            print_status "Installing from conda-forge..."
            conda install -c conda-forge r-abcneuralnet -y
            ;;
        "github")
            print_status "Installing from GitHub..."
            R -e "devtools::install_github('ThomasBrazier/abcneuralnet', dependencies = TRUE)"
            ;;
        "local")
            print_status "Installing from local source..."
            if [ ! -f "DESCRIPTION" ]; then
                print_error "No DESCRIPTION file found in current directory"
                print_status "Please run this script from the abcneuralnet root directory"
                exit 1
            fi
            R -e "devtools::install('.', dependencies = TRUE)"
            ;;
        *)
            print_error "Unknown installation method: $METHOD"
            print_status "Available methods: conda-forge, github, local"
            exit 1
            ;;
    esac
}

# Setup torch (including CUDA if available)
setup_torch() {
    print_status "Setting up torch..."
    
    # Check if CUDA is available
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected, setting up CUDA support..."
        R -e "torch::install_torch(type = 'cuda')"
    else
        print_status "No NVIDIA GPU detected, using CPU version..."
        R -e "torch::install_torch(type = 'cpu')"
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Test package loading
    R -e "
    library(abcneuralnet)
    cat('Package version:', packageVersion('abcneuralnet'), '\n')
    cat('Torch installation status:', torch::torch_is_installed(), '\n')
    cat('CUDA available:', torch::torch_cuda_is_available(), '\n')
    "
    
    if [ $? -eq 0 ]; then
        print_status "Installation verified successfully!"
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Run basic tests
run_tests() {
    print_status "Running basic tests..."
    
    R -e "
    tryCatch({
        testthat::test_package('abcneuralnet')
        cat('All tests passed!\n')
    }, error = function(e) {
        cat('Some tests failed:', e$message, '\n')
    })
    "
}

# Main installation function
main() {
    print_status "Starting abcneuralnet conda installation..."
    
    # Parse command line arguments
    ENV_NAME="abcneuralnet"
    INSTALL_METHOD="conda-forge"
    RUN_TESTS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--name)
                ENV_NAME="$2"
                shift 2
                ;;
            -m|--method)
                INSTALL_METHOD="$2"
                shift 2
                ;;
            -t|--test)
                RUN_TESTS=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  -n, --name NAME      Environment name (default: abcneuralnet)"
                echo "  -m, --method METHOD  Installation method (conda-forge, github, local)"
                echo "  -t, --test          Run tests after installation"
                echo "  -h, --help          Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run installation steps
    check_conda
    create_environment $ENV_NAME
    conda activate $ENV_NAME
    setup_torch
    install_package $INSTALL_METHOD
    verify_installation
    
    if [ "$RUN_TESTS" = true ]; then
        run_tests
    fi
    
    print_status "Installation completed successfully!"
    print_status "To activate the environment, run: conda activate $ENV_NAME"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
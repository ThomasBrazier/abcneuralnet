@echo off
REM Conda installation script for abcneuralnet (Windows)

setlocal enabledelayedexpansion

REM Default values
set ENV_NAME=abcneuralnet
set INSTALL_METHOD=conda-forge
set RUN_TESTS=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :main
if /i "%~1"=="-n" (
    set ENV_NAME=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--name" (
    set ENV_NAME=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="-m" (
    set INSTALL_METHOD=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--method" (
    set INSTALL_METHOD=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="-t" (
    set RUN_TESTS=true
    shift
    goto :parse_args
)
if /i "%~1"=="--test" (
    set RUN_TESTS=true
    shift
    goto :parse_args
)
if /i "%~1"=="-h" goto :show_help
if /i "%~1"=="--help" goto :show_help

echo Unknown option: %~1
exit /b 1

:show_help
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   -n, --name NAME      Environment name (default: abcneuralnet)
echo   -m, --method METHOD  Installation method (conda-forge, github, local)
echo   -t, --test          Run tests after installation
echo   -h, --help          Show this help message
exit /b 0

:main
echo [INFO] Starting abcneuralnet conda installation...

REM Check if conda is installed
where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Conda is not installed or not in PATH
    echo [INFO] Please install Anaconda or Miniconda first:
    echo   - Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo   - Anaconda: https://www.anaconda.com/products/distribution
    exit /b 1
)

echo [INFO] Found conda installation
conda --version

REM Check if environment already exists
conda env list | findstr /r /c:"^%ENV_NAME% " >nul
if not errorlevel 1 (
    echo [WARNING] Environment '%ENV_NAME%' already exists
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "!RECREATE!"=="y" (
        echo [INFO] Removing existing environment...
        conda env remove -n %ENV_NAME% -y
    ) else (
        echo [INFO] Using existing environment
        conda activate %ENV_NAME%
        goto :setup_torch
    )
)

REM Create environment
echo [INFO] Creating conda environment: %ENV_NAME%
conda create -n %ENV_NAME% ^
    -c conda-forge ^
    r-base=4.3 ^
    r-r6 ^
    r-rcolorbrewer ^
    r-rdpack ^
    r-knitr ^
    r-abc ^
    r-bundle ^
    r-dplyr ^
    r-ggplot2 ^
    r-ggpubr ^
    r-innsight ^
    r-janitor ^
    r-luz ^
    r-plotly ^
    r-tabnet ^
    r-tibble ^
    r-tidyr ^
    r-tidyverse ^
    r-torch ^
    r-devtools ^
    r-assertthat ^
    r-here ^
    -y

if errorlevel 1 (
    echo [ERROR] Failed to create conda environment
    exit /b 1
)

echo [INFO] Environment created successfully

:setup_torch
echo [INFO] Setting up torch...

REM Activate environment
call conda activate %ENV_NAME%

REM Install torch CPU version (Windows CUDA support is limited)
echo [INFO] Installing torch CPU version...
R -e "torch::install_torch(type = 'cpu')"

if errorlevel 1 (
    echo [WARNING] Torch installation may have issues on Windows
    echo [INFO] You may need to install torch manually in R
)

REM Install abcneuralnet
echo [INFO] Installing abcneuralnet using method: %INSTALL_METHOD%

if /i "%INSTALL_METHOD%"=="conda-forge" (
    echo [INFO] Installing from conda-forge...
    conda install -c conda-forge r-abcneuralnet -y
) else if /i "%INSTALL_METHOD%"=="github" (
    echo [INFO] Installing from GitHub...
    R -e "devtools::install_github('ThomasBrazier/abcneuralnet', dependencies = TRUE)"
) else if /i "%INSTALL_METHOD%"=="local" (
    echo [INFO] Installing from local source...
    if not exist "DESCRIPTION" (
        echo [ERROR] No DESCRIPTION file found in current directory
        echo [INFO] Please run this script from the abcneuralnet root directory
        exit /b 1
    )
    R -e "devtools::install('.', dependencies = TRUE)"
) else (
    echo [ERROR] Unknown installation method: %INSTALL_METHOD%
    echo [INFO] Available methods: conda-forge, github, local
    exit /b 1
)

if errorlevel 1 (
    echo [ERROR] Package installation failed
    exit /b 1
)

REM Verify installation
echo [INFO] Verifying installation...
R -e "library(abcneuralnet); cat('Package version:', packageVersion('abcneuralnet'), '\n'); cat('Torch installation status:', torch::torch_is_installed(), '\n')"

if errorlevel 1 (
    echo [ERROR] Installation verification failed
    exit /b 1
)

REM Run tests if requested
if /i "%RUN_TESTS%"=="true" (
    echo [INFO] Running basic tests...
    R -e "tryCatch({testthat::test_package('abcneuralnet'); cat('All tests passed!\n')}, error = function(e) {cat('Some tests failed:', e$message, '\n')})"
)

echo [INFO] Installation completed successfully!
echo [INFO] To activate the environment, run: conda activate %ENV_NAME%

pause
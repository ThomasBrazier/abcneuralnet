#!/bin/bash
# recipe/build.sh

# Disable setuptools for R packages
export SETUPTOOLS_USE_DISTUTILS=stdlib

# Install the R package
$R CMD INSTALL --build .

if [ $? -ne 0 ]; then
    exit 1
fi
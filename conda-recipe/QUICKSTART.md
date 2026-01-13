# Quick Start Guide

## One-line Installation (Linux/macOS)
```bash
curl -sSL https://raw.githubusercontent.com/ThomasBrazier/abcneuralnet/main/conda-recipe/install.sh | bash
```

## One-line Installation (Windows) 
```powershell
iwr -useb https://raw.githubusercontent.com/ThomasBrazier/abcneuralnet/main/conda-recipe/install.bat | cmd
```

## Custom Installation Options

### Linux/macOS
```bash
# Custom environment name
./install.sh --name my-abc-env

# Install from GitHub
./install.sh --method github

# Install from local source
./install.sh --method local

# Run tests after installation
./install.sh --test
```

### Windows
```cmd
# Custom environment name
install.bat --name my-abc-env

# Install from GitHub  
install.bat --method github

# Install from local source
install.bat --method local

# Run tests after installation
install.bat --test
```
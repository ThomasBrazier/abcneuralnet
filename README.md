# ABCNeuralNet


![license](https://badgen.net/badge/license/GPL-3.0/blue)
![release](https://badgen.net/badge/release/0.1.0/blue?icon=github)
[![rworkflows](https://github.com/ThomasBrazier/abcneuralnet/actions/workflows/r.yml/badge.svg)](https://github.com/ThomasBrazier/abcneuralnet/actions/workflows/r.yml)



## Installation

`ABCCN` requires some dependencies before to be installed. If you want to use the `gpu` to accelerate the training and inferences, you must install `CUDA` on a system with a compatible `gpu`.

### Install `CUDA` for GPU usage

The list of CUDA-compatible [GPUs](https://developer.nvidia.com/cuda-gpus#compute)

[CUDA toolkit](https://docs.nvidia.com/cuda/archive/11.7.0/)

Instructions to install [cudnn](https://developer.nvidia.com/cudnn)

### Install `Torch`

See the mlverse [Torch](https://torch.mlverse.org/docs/articles/installation) documentation for installing the `torch` R package.

```
install.packages("torch")
library(torch) # Will install all libraries necessary (lantern, torch)

# Check if CUDA installed
cuda_is_available()
```

### Install `ABCNN`


Install the main branch from github.


```
devtools::install_github("ThomasBrazier/abcneuralnet")
```

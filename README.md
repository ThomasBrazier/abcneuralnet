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

So far, `abcneuralnet` is still in development and not available from CRAN.


You can install the development version with:

```
devtools::install_github("ThomasBrazier/abcneuralnet")
```


## How to use ABCNeuralNet

You can see a detailed example on how to use the `abcneuralnet` package in the vignettes. A basic example is:

```
library(abcneuralnet) 

abc = abcnn$new(theta,
            sumstats,
            observed,
            method = 'concrete dropout',
            scale_input = "none",
            scale_target = "none",
            num_hidden_layers = 3,
            num_hidden_dim = 256,
            epochs = 30,
            batch_size = 32,
            l2_weight_decay = 1e-5)
            
abc$fit()
abc$plot_training()

abc$predict()
abc$plot_prediction()
```



## Contributing


No matter your current skills it’s possible to contribute to `abcneuralnet` development (contact me for details).


It is important for me to have comments and feature requests. Feel free to open an issue if you find a typo or a bug, if you have a question how to use the package, or to submit a feature request.

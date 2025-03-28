# abcneuralnet



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
library(torch)

# Check if CUDA installed
cuda_is_available()
```

The `keras3` R package need to be installed for helper functions.


```
install.package("keras3")
```


### Install `ABCNN`
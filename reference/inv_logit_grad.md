# The derivative of the backward logit transform

The derivative of
[`inv_logit()`](https://thomasbrazier.github.io/abcneuralnet/reference/inv_logit.md)
with respect to `z`. As `unsqueeze()` is affine in `p` with slope
`n / (n - 1)`, and `d plogis(z) / dz = p * (1 - p)`, the chain rule
gives `(b - a) * n / (n - 1) * p * (1 - p)`.

## Usage

``` r
inv_logit_grad(z, a, b, n)
```

## Arguments

- z:

  a vector of numerical values

- a:

  the min value of the training set

- b:

  the max value of the training set

- n:

  the number of samples in the training set

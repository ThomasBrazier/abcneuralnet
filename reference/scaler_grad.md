# The gradient of the backward scaling transform

Returns `|g'(z)|`, the absolute derivative of the backward transform
applied by `scaler(type = "backward")`, evaluated at the scaled values
`z`.

This is the Jacobian factor used to carry a standard deviation from the
scaled space, where the neural network is trained, to the original
parameter scale with the delta method:
`sd_original ~ |g'(z)| * sd_scaled`.

The factor is a constant, and the delta method therefore exact, for the
affine methods (`none`, `minmax`, `robustscaler` and `normalization`).
For `log` and `logit` the backward transform is non-linear, so the
result is only a local linearisation around `z` and the resulting
symmetric interval may fall outside the support of the parameter. Prefer
the conformal or posterior-quantile intervals returned by
`abcnn$predictions()`, which are built by transforming interval
endpoints and are exact under any monotone scaling.

## Usage

``` r
scaler_grad(z, sum_stats, method = "minmax")
```

## Arguments

- z:

  a data frame of scaled values at which to evaluate the gradient,
  typically the scaled predictive mean. Each column is treated
  separately.

- sum_stats:

  list, summary statistics learned on the training data. See
  [`scaler()`](https://thomasbrazier.github.io/abcneuralnet/reference/scaler.md).

- method:

  the scaling method, either `minmax`, `robustscaler`, `normalization`,
  `log`, `logit` or `none`. Can be a single character (same
  transformation applied to all columns) or a vector of characters with
  one transformation per column.

## Value

a data frame of gradients with the same dimensions as `z`

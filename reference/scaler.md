# A scaling function for targets and inputs

The function allows to back-transform the numerical values to their
original scale. For this, it requires a list of summary statistics
learned on the training set.

## Usage

``` r
scaler(x, sum_stats, method = "minmax", type = "forward")
```

## Arguments

- x:

  a data frame to scale, each column is scaled separately

- sum_stats:

  list, summary statistics learned on the data to back-transform

- method:

  the scaling method, either `minmax`, `robustscaler`, `normalization`
  or `none`

- type:

  is `forward` when scaling inputs or targets and `backward` when
  back-transforming targets at prediction time

## Value

a data frame with scaled values

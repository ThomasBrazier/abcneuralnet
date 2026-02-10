# Compute the log1pexp trick

This is a more stable version of log(1 + exp(x)). Note that log(1 +
exp(x)) is approximately equal to x when x is large enough. See
https://stackoverflow.com/questions/60903821/how-to-prevent-inf-while-working-with-exponential
for details

## Usage

``` r
log1pexp(x, threshold = 10)
```

## Arguments

- x:

  a tensor

- threshold:

  the threshold value under which the trick is applied to avoid `Inf`
  values

## Value

a tensor with values corrected with the log1pexp trick

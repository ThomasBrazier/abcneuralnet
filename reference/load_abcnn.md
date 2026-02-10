# Load an `abcnn` object and the serialized luz fitted model

The function reconstructs an `abcnn` object from the `_luz.Rds`,
`_model.Rds` and `_abcnn.Rds` files.

## Usage

``` r
load_abcnn(prefix = "")
```

## Arguments

- prefix:

  character, the prefix with path of the saved .Rds object

## Value

an `abcnn` object

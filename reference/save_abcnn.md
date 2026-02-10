# Save the `abcnn` object and the serialized luz fitted model

The function will save a `_luz.Rds`, a `_model.Rds` and a `_abcnn.Rds`,
which will contain the `luz` fitted model, the original `torch` model
and the `abcnn` model. The `abcnn` model will be reconstructed with
[`load_abcnn()`](https://thomasbrazier.github.io/abcneuralnet/reference/load_abcnn.md).

## Usage

``` r
save_abcnn(object, prefix = "")
```

## Arguments

- object:

  an `abcnn` object with a `luz` fitted model

- prefix:

  character, the prefix with path of the saved .Rds object

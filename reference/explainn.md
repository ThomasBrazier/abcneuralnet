# An `explainn` object for feature attribution A R6 class object

This module function allows to apply a diverse set of Feature attribtion
methods on a fitted `abcnn` neural network and a given observed dataset,
in order to compute the weight of each summary statistic on predictions.
Summary statistics with the higher weight (or importance) are those
contributing the most to the prediction.

This feature importance method is useful to perform feature selection
(removing summary statistics that don't explain well the output) and to
interpret properly the output of the model.

## Value

an `explainn` object

## Details

All the methods used in explain are implemented in the `innsight` R
package, that is part of the R torch ecosystem. These methods are:

- Vanilla Gradient and GradientxInput

- SmoothGrand and SmoothGradxInput

- Integrated Gradients

- Expected Gradients

- Layer-Wise Relevance Propagation (LRP)

- Deep learning important features (DeepLift)

- Deep Shapley additive explanations (DeepSHAP)

- Connection weights method

- Local interpretable model-agnostic explanations (LIME)

- Shapley values (SHAP)

See `https://bips-hb.github.io/innsight/` for details.

## Slots

- `converter`:

  Stores the `innsight::converter` object

- `result`:

  stores results of the `explainn$run()` method.

- `model_method`:

  method of the trained neural network (e.g. "concrete dropout")

- `variables`:

  names of the variables (summary statistics)

- `parameters`:

  names of the parameter to infer

- `ensemble_num_model`:

  index of the model when the network is a deep ensemble

- `scale_input`:

  the `abcnn$scale_input` slot from the `abcnn` input object

- `input_summary`:

  the `abcnn$input_summary` slot from the `abcnn` input object

## Public fields

- `x`:

  an `abcnn` object

- `method`:

  the `innsight` method to apply: `grad`, `cw`, `smoothgrad`, `intgrad`,
  `expgrad`, `lrp`, `deeplift`, `deepshap`, `shap`, `lime`

- `converter`:

  the torch/luz model converted to an `innsight` object

- `result`:

  the result of the explainability method

- `model_method`:

  the method in the `abcnn` object

- `variables`:

  names of the variables (input dimensions)

- `parameters`:

  names of the parameters to estimate (output dimensions)

- `ensemble_num_model`:

  index of the model to explain in Deep Ensemble (default is first
  model)

- `scale_input`:

  method used to scale input dimensions

- `input_summary`:

  summary statistics for the input scaling method

## Methods

### Public methods

- [`explainn$new()`](#method-explainn-new)

- [`explainn$print()`](#method-explainn-print)

- [`explainn$run()`](#method-explainn-run)

- [`explainn$get_result()`](#method-explainn-get_result)

- [`explainn$plot()`](#method-explainn-plot)

- [`explainn$plot_global()`](#method-explainn-plot_global)

- [`explainn$boxplot()`](#method-explainn-boxplot)

- [`explainn$clone()`](#method-explainn-clone)

------------------------------------------------------------------------

### Method `new()`

Create a new `explainn` object

#### Usage

    explainn$new(x, method = "cw", ensemble_num_model = 1)

#### Arguments

- `x`:

  an `abcnn` model

- `method`:

  the explainability method to use (see `innsight` for details) (defauls
  is `cw`)

- `ensemble_num_model`:

  index of the model to explain in Deep Ensemble (default is first
  model)

------------------------------------------------------------------------

### Method [`print()`](https://rdrr.io/r/base/print.html)

Print the converter

#### Usage

    explainn$print()

------------------------------------------------------------------------

### Method `run()`

Apply the `method` to the passed `data` to be explained

The method is run on a `data` object (see `innsight` manual)

#### Usage

    explainn$run(data, data_ref = NULL, method = NULL)

#### Arguments

- `data`:

  (array, data.frame, torch_tensor or list) The data to which the method
  is to be applied. These must have the same format as the input data of
  the passed model to the converter object. This means either an array,
  data.frame, torch_tensor or array-like format of size (batch_size,
  dim_in), if e.g., the model has only one input layer, or a list with
  the corresponding input data (according to the upper point) for each
  of the input layers.

  Note: For the model-agnostic methods, only models with a single input
  and output layer is allowed!

- `data_ref`:

  (array, data.frame or torch_tensor) The dataset to which the method is
  to be applied. These must have the same format as the input data of
  the passed model and has to be either matrix, an array, a data.frame
  or a torch_tensor. Note: For the model-agnostic methods, only models
  with a single input and output layer is allowed!

- `method`:

  The method to run. Change the method specified in `new()`

------------------------------------------------------------------------

### Method `get_result()`

Get the results of the Feature Attribution method

#### Usage

    explainn$get_result(type = "array")

#### Arguments

- `type`:

  the results can be returned as an `array`, `data.frame`, or
  `torch_tensor`

#### Details

Note that when the `abcnn` model is `tabnet-abc`, `get_result()` returns
importances weigths of the fitted model.

------------------------------------------------------------------------

### Method [`plot()`](https://rdrr.io/r/graphics/plot.default.html)

Plot the results of the Feature Attribution method for single data
points

#### Usage

    explainn$plot(as_plotly = FALSE, type = "barplot", output_label = NULL)

#### Arguments

- `as_plotly`:

  If `TRUE`, plot the figure as a plotly object (default = `FALSE`)

- `type`:

  a character value. The type of plot for `Tabnet`, passed to the Tabnet
  autoplot method. Either `barplot` for importance scores averaged
  across masks, `mask_agg`, for a single heatmap of aggregated mask
  importance per predictor along the dataset, or `steps` for one heatmap
  at each mask step.

- `output_label`:

  character, the names of the variables to plot (if NULL, all variables
  are plotted)

#### Details

Note that when the `abcnn` model is `tabnet-abc`,
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) returns the
`autoplot()` function on the results of the `tabnet` model.

------------------------------------------------------------------------

### Method `plot_global()`

Plot the results of the Feature Attribution method for the global
dataset

#### Usage

    explainn$plot_global(as_plotly = FALSE, output_label = NULL)

#### Arguments

- `as_plotly`:

  If `TRUE`, plot the figure as a plotly object (default = `FALSE`)

- `output_label`:

  character, the names of the variables to plot (if NULL, all variables
  are plotted)

------------------------------------------------------------------------

### Method [`boxplot()`](https://rdrr.io/r/graphics/boxplot.html)

Alias for `plot_global` for tabular and signal data

#### Usage

    explainn$boxplot(as_plotly = FALSE)

#### Arguments

- `as_plotly`:

  If `TRUE`, plot the figure as a plotly object (default = `FALSE`)

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    explainn$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

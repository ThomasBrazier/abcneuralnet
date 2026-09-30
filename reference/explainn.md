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

The network is trained on scaled targets (see `scale_target` in
`abcnn`), so raw attributions are expressed in units of the scaled
target, which differ between parameters and between scaling methods. By
default, `run()` rescales them to a common scale, the fraction of each
parameter's prior range, so that they are comparable across parameters.
See the `output_scale` argument of `run()`.

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

- `scale_target`:

  method used to scale the targets (parameters) in the `abcnn` object

- `target_summary`:

  summary statistics for the target scaling method (see
  `abcnn$target_summary`)

- `target_center`:

  median of the scaled training targets, the point at which `cw`
  attributions are rescaled under the non-linear `log` and `logit`
  target scalings

- `model`:

  the explained network (mu head only) as a `torch` module on cpu, used
  to compute the scaled predictions needed to rescale attributions

- `output_scale`:

  the scale of the attributions in `result` (`prior_range`, `relative`
  or `none`), see `run()`

## Methods

### Public methods

- [`explainn$new()`](#method-explainn-initialize)

- [`explainn$print()`](#method-explainn-print)

- [`explainn$run()`](#method-explainn-run)

- [`explainn$get_result()`](#method-explainn-get_result)

- [`explainn$plot()`](#method-explainn-plot)

- [`explainn$plot_global()`](#method-explainn-plot_global)

- [`explainn$boxplot()`](#method-explainn-boxplot)

- [`explainn$clone()`](#method-explainn-clone)

------------------------------------------------------------------------

### `explainn$new()`

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

### `explainn$print()`

Print the converter

#### Usage

    explainn$print()

------------------------------------------------------------------------

### `explainn$run()`

Apply the `method` to the passed `data` to be explained

The method is run on a `data` object (see `innsight` manual)

#### Usage

    explainn$run(
      data,
      data_ref = NULL,
      method = NULL,
      output_scale = "prior_range"
    )

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

- `output_scale`:

  The scale on which attributions are reported, so that they are
  comparable across parameters whatever their `scale_target` (ignored
  for Tabnet-ABC):

  - `prior_range` (default): the change in each parameter, expressed as
    a fraction of the width of its prior support
    (`target_summary$max - target_summary$min`). This is the scale
    `minmax` already trains on, so `minmax` attributions are left
    unchanged.

  - `relative`: per sample and per parameter, attributions are divided
    by the sum of their absolute values over all summary statistics,
    giving signed shares whose absolute values sum to 1. Magnitude
    across parameters is lost, but only relative importance is kept.

  - `none`: raw attributions of the network output, i.e. on the *scaled*
    target space. These are not comparable across parameters with
    different `scale_target` or ranges.

------------------------------------------------------------------------

### `explainn$get_result()`

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

### `explainn$plot()`

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

### `explainn$plot_global()`

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

### `explainn$boxplot()`

Alias for `plot_global` for tabular and signal data

#### Usage

    explainn$boxplot(as_plotly = FALSE)

#### Arguments

- `as_plotly`:

  If `TRUE`, plot the figure as a plotly object (default = `FALSE`)

------------------------------------------------------------------------

### `explainn$clone()`

The objects of this class are cloneable with this method.

#### Usage

    explainn$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

# An `abcnn` R6 class object for parameter inference with Bayesian Neural Networks and Approximate Bayesian Computation

`abcnn` constructs a `R6` class object for parameter inference with ABC
and neural networks. It implements four different methods mixing ABC and
neural networks implemented in R `torch`.

The `initialize` function (`abcnn$new()`) takes as arguments three data
frames of training summary statistics, training theta values and
observed summary statistics. Public slots can be accessed and modified.
A new `abcnn` object is created with `abcnn$new()`.

## Value

A `R6::abcnn` object

an `abcnn` object that can be used to fit(), predict() and plot
predictions

Returns a list with n random samples for each observed sample

Returns metrics computed on the cross-validation dataset

## Details

Four methods are available for parameter inference. The two core methods
are `concrete dropout`, an implementation of Gal et al. (2017), and
`deep ensemble`, an implementation of Lakshminarayanan et al. (2017),
that allow to estimate both the aleatoric and epistemic uncertainty for
each sample. `monte carlo dropout` is an implementation of Gal and
Ghahramani (2016), that provides a simpler model that is easier to
train, despite its limitations (the dropout rate must be arbitrary
chosen). The `gaussian monte carlo dropout` is a version of
`monte carlo dropout` where the loss function is the same as in
`concrete dropout` allowing to estimate both aleatoric and epistemic
uncertainty.

A fourth method is `tabnet-abc`. This is a new method, combining regular
ABC inference with the `abc` R package, and a Tabnet neural network, as
in Arik et al. (2021) and implemented in the `tabnet` R package. This is
the same idea than in Åkesson et al. (2021) or Jiang et al. (2017),
except than te MLP/CNN used to estimate summary statistics is replaced
by a `tabnet` model specifically designed to handle tabular data and
feature selection through an attention map on features. The `tabnet`
neural network is trained to predict summary statistics from the
observed summary statistics. Then these predictions are used as a
supplementary set of summary statistics and regular ABC inference is
performed on it. Explain methods are specific the `tabnet-abc` model.

In addition, the credible interval is calibrated with conformal
prediction, as in Baragatti et al. (2024). As it requires a proxy of
uncertainty, conformal prediction is only available for
`concrete dropout`, `deep ensemble`, `gaussian monte carlo dropout` and
`monte carlo dropout` (only for the epistemic uncertainty for this last
method).

The neural networks are implemented with the `torch` R package and
support CUDA devices for training. The `luz` package is used as a higher
level API for training and predictions with `torch`. The device (`CUDA`
or `cpu`) is automatically detected by `luz`.

The `abcnn` object has public methods to perform each inference step and
visualizations.

- `new()` to create a new `abcnn` object

- `fit()` to fit a neural network

- [`predict()`](https://rdrr.io/r/stats/predict.html) to compute
  conformal predictions from the fitted model

- [`summary()`](https://rdrr.io/r/base/summary.html) to print a summary
  of the `abcnn` object

- `predictions()` to print predictions

- `plot_training()` to plot the training curves

- `plot_prediction()` to plot all predictions with their credible
  intervals

- `plot_posterior()` to plot the prior and posterior distributions, with
  the mean and credible intervals, of a single sample

- `draw_from_posterior()` to draw `n` samples from the posterior
  distribution (e.g. for posterior predictive check)

- `cross_validation()` to compute cross-validation metrics by comparing
  ground truth and predictions on unseen pseudo-observed data (i.e.
  simulations)

- `plot_cross_validation()` to plot the cross-validation scatter plot

The hyperparameters of the neural network can be configured in the
`new()` method or modified directly in the corresponding public `slot`.

## Slots

- `theta`:

  parameters of the pseudo-observed samples (i.e. simulations)

- `sumstat`:

  summary statistics of the pseudo-observed samples (i.e. simulations)

- `observed`:

  summary statistics of the observed samples

- `model`:

  the `luz` model

- `method`:

  the ABC-NN method used

- `scale_input`:

  the scaling method for summary statistics

- `scale_target`:

  the scaling method for targets (i.e. theta)

- `num_hidden_layers`:

  number of hidden layers in the neural network

- `num_hidden_dim`:

  number of dimensions (neurons) in each layer

- `validation_split`:

  proportion of samples retained for validation at the end of training

- `num_conformal`:

  number of samples retained for conformal prediction

- `credible_interval_p`:

  significance level for the credible interval, between 0 and 1

- `test_split`:

  proportion of samples retained for test at each training iteration

- `dropout`:

  dropout rate

- `batch_size`:

  batch size

- `epochs`:

  number of epochs for training

- `early_stopping`:

  whether to do early stopping

- `patience`:

  patience hyperparameter for early stopping. See
  [`luz::luz_callback_early_stopping()`](https://mlverse.github.io/luz/reference/luz_callback_early_stopping.html)

- `callbacks`:

  custom callbacks in `luz` (in development)

- `verbose`:

  whether to print messages

- `optimizer`:

  `torch` optimizer nn module

- `learning_rate`:

  learning rate

- `l2_weight_decay`:

  L2 weigth decay (regularization)

- `variance_clamping`:

  `c(min, max)` values for variance clamping during training

- `loss`:

  `torch` nn loss function

- `tol`:

  tolerance rate for `abc` functions (only for `tabnet-abc`)

- `abc_method`:

  ABC sampling method in `abc` function (only for `tabnet-abc`)

- `num_posterior_samples`:

  number of samples to generate for the posterior distribution

- `prior_length_scale`:

  prior length scale hyperparameter value

- `weight_regularizer`:

  `concrete dropout` regularization term for weights

- `dropout_regularizer`:

  `concrete dropout` regularization term for dropout

- `num_networks`:

  number of networks in `deep ensemble`

- `epsilon_adversarial`:

  the amount of perturbation for adversarial training in `deep ensemble`

- `device`:

  `luz`/`torch` device for tensors

- `input_dim`:

  number of input dimensions of the neural network

- `output_dim`:

  number of output dimensions of the neural network

- `n_train`:

  number of training samples

- `sumstat_names`:

  names of summary statistics

- `output_names`:

  output names

- `theta_names`:

  names of theta to estimate

- `n_obs`:

  number of observed samples

- `prior_lower`:

  lower boundary of priors (for figures)

- `prior_upper`:

  upper boundary of priors (for figures)

- `fitted`:

  the fitted `luz` model

- `evaluation`:

  the evaluation metric

- `eval_metrics`:

  `torch` nn metrics for evaluation

- `posterior_samples`:

  array of posterior samples

- `quantile_posterior`:

  quantiles computed on the posterior samples, given the
  `credible_interval_p`

- `predictive_mean`:

  values predicted by the model for each observed sample

- `aleatoric_uncertainty`:

  aleatoric uncertainty for each observed sample

- `epistemic_uncertainty`:

  epistemic uncertainty for each observed sample

- `overall_uncertainty`:

  overall uncertainty for each observed sample (epistemic + aleatoric)

- `epistemic_conformal_quantile`:

  the quantile factor to get the conformalized credible interval for
  epistemic uncertainty

- `overall_conformal_quantile`:

  the quantile factor to get the conformalized credible interval for
  overall uncertainty

- `dropout_rates`:

  the dropout rate hyperparameter estimated by concrete dropout

- `input_summary`:

  statistics computed on input data (for scaling)

- `target_summary`:

  statistics computed on target data (for scaling)

- `sumstat_adj`:

  adjusted training summary statistics after scaling

- `observed_adj`:

  adjusted observed summary statistics after scaling

- `theta_adj`:

  adjusted training theta after scaling

- `calibration_theta`:

  adjusted theta for conformal prediction after scaling (calibration
  set)

- `calibration_sumstat`:

  adjusted summary statistics for conformal prediction after scaling
  (calibration set)

- `ncores`:

  number of cores for parallelized steps

- `call`:

  the call to the new() initialisation function

## References

Baragatti M, Céline C, Cloez B, Métivier D, Sanchez I (2024).
“Approximate bayesian computation with deep learning and conformal
prediction.” *arXiv preprint arXiv:2406.04874*. Gal Y, Ghahramani Z
(2016). “Dropout as a Bayesian Approximation: Representing Model
Uncertainty in Deep Learning.” In Balcan MF, Weinberger KQ (eds.),
*Proceedings of The 33rd International Conference on Machine Learning*,
volume 48 of *Proceedings of Machine Learning Research*, 1050–1059.
<https://proceedings.mlr.press/v48/gal16.html>. Gal Y, Hron J, Kendall A
(2017). “Concrete dropout.” *Advances in neural information processing
systems*, **30**. Lakshminarayanan B, Pritzel A, Blundell C (2017).
“Simple and scalable predictive uncertainty estimation using deep
ensembles.” *Advances in neural information processing systems*, **30**.
Arik SÖ, Pfister T (2021). “Tabnet: Attentive interpretable tabular
learning.” In *Proceedings of the AAAI conference on artificial
intelligence*, volume 35(8), 6679–6687. Falbel D (2025). *tabnet: Fit
'TabNet' Models for Classification and Regression*. R package version
0.7.0, <https://CRAN.R-project.org/package=tabnet>. Åkesson M, Singh P,
Wrede F, Hellander A (2021). “Convolutional neural networks as summary
statistics for approximate Bayesian computation.” *IEEE/ACM Transactions
on Computational Biology and Bioinformatics*, **19**(6), 3353–3365.
Jiang B, Wu T, Zheng C, Wong WH (2017). “Learning summary statistic for
approximate Bayesian computation via deep neural network.” *Statistica
Sinica*, 1595–1618.

## See also

[`R6::R6()`](https://r6.r-lib.org/reference/R6Class.html)

## Public fields

- `theta`:

  parameters of the pseudo-observed samples (i.e. simulations)

- `sumstat`:

  summary statistics of the pseudo-observed samples (i.e. simulations)

- `observed`:

  summary statistics of the observed samples

- `model`:

  the `luz` model

- `method`:

  the ABC-NN method used, whether `tabnet-abc`, `monte carlo dropout`,
  `gaussian monte carlo dropout`, `concrete dropout` or `deep ensemble`

- `scale_input`:

  the scaling method for summary statistics

- `scale_target`:

  the scaling method for targets (i.e.e theta)

- `num_hidden_layers`:

  number of hidden layers in the neural network

- `num_hidden_dim`:

  number of hidden dimensions (neurons) in each hidden layer

- `validation_split`:

  proportion of training samples to retain for validation at the end of
  training

- `num_conformal`:

  number of training samples to retain for conformal prediction (not
  used during training)

- `credible_interval_p`:

  proportion, the level of significance for credible intervals

- `test_split`:

  proportion of training samples to retain for testing (at each training
  iteration)

- `dropout`:

  dropout rate to apply in `monte carlo dropout` and
  `gaussian monte carlo dropout`

- `batch_size`:

  batch size in `luz`

- `epochs`:

  number of epochs for training

- `early_stopping`:

  logical, whether to do early stopping in `luz` (not implemented yet
  for the `deep ensemble` method)

- `callbacks`:

  list of `luz` callbacks (not implemented for the method 'deep
  ensemble')

- `verbose`:

  logical, whether to print messages and progress bars for the user

- `patience`:

  patience hyperparameter for `luz` `early stopping`, the number of
  epochs without improving until stoping training

- `optimizer`:

  `torch` custom optimizer

- `learning_rate`:

  learningrate in `luz`

- `l2_weight_decay`:

  L2 weight decay for regularization in the
  [`torch::optimizer`](https://torch.mlverse.org/docs/reference/optimizer.html)

- `variance_clamping`:

  `c(min, max)` values for variance clamping during training

- `loss`:

  custom `torch` loss function (nn module)

- `tol`:

  tolerance rate in `abc` for the `tabnet-abc` method

- `abc_method`:

  `abc` method for `tabnet-abc`

- `num_posterior_samples`:

  number of posterior samples to generate in `monte carlo dropout`,
  `gaussian monte carlo dropout` and `concrete dropout`

- `abc_keep_original_sumstats`:

  (logical or numeric) Whether to merge the new set of summary
  statistics with the original ones (TRUE), or just keep the new ones
  (FALSE, default value). If a proportion p (\> 0 and \< 1) is given,
  then the original summary statistics with a relative importance \> p
  are kept. If an integer n \>= 1 is given, then the n most important
  original summary statistics are kept. The variable importances are the
  one computed with `tabnet`.

- `prior_length_scale`:

  hyperparameter for `concrete dropout`

- `weight_regularizer`:

  hyperparameter for `concrete dropout`

- `dropout_regularizer`:

  hyperparameter for `concrete dropout`

- `num_networks`:

  number of neural networks in `deep ensemble`

- `epsilon_adversarial`:

  the factor by which perturbating training samples for adversarial
  training in `deep ensemble`

- `device`:

  device used in `luz` and `torch`, whether 'cpu' or 'cuda' (GPU)

- `input_dim`:

  number of input dimensions (columns in summary statistics)

- `output_dim`:

  number of output dimensions (columns in theta, the number of variables
  to predict)

- `n_train`:

  number of samples for training

- `sumstat_names`:

  names of the summary statistics

- `output_names`:

  neural network output names (mean and variance)

- `theta_names`:

  theta names (variables to predict)

- `n_obs`:

  number of observations (rows in 'observed')

- `prior_lower`:

  lower boundaries of priors (for plotting)

- `prior_upper`:

  upper boundaries of priors (for plotting)

- `fitted`:

  a model fitted with `luz`

- `evaluation`:

  numerical value of the evaluation metric

- `eval_metrics`:

  list of custom metrics to use at evaluation (not implemented yet)

- `posterior_samples`:

  array of all posterior samples predicted in `monte carlo dropout`,
  `gaussian monte carlo dropout` and `concrete dropout`

- `quantile_posterior`:

  quantiles of the posterior distributions, given the credible interval
  significance required

- `predictive_mean`:

  mean predicted value

- `aleatoric_uncertainty`:

  aleatoric uncertainty

- `epistemic_uncertainty`:

  epistemic uncertainty

- `overall_uncertainty`:

  overall uncertainty

- `epistemic_conformal_quantile`:

  conformal quantile of epistemic uncertainty calibrated with conformal
  prediction

- `overall_conformal_quantile`:

  conformal quantile of overall uncertainty calibrated with conformal
  prediction

- `dropout_rates`:

  dropout rates inferred by `concrete dropout` (not implemented yet)

- `input_summary`:

  summary statistics of input data for scaling

- `target_summary`:

  summary statistics of target data (theta) for scaling

- `sumstat_adj`:

  scaled training summary statistics

- `observed_adj`:

  scaled observed summary statistics

- `theta_adj`:

  scaled training target

- `calibration_theta`:

  theta saved for calibration with conformal prediction

- `calibration_sumstat`:

  summary statistics saved for calibration with conformal prediction

- `ncores`:

  number of cores for parallel procedures

- `seed`:

  a random seed when initializing the network

- `cross_validation_data`:

  the unseen dataset for cross-validation

- `cross_validation_predictions`:

  predictions for the cross-validation dataset

- `call`:

  the call to the new() initialisation function

## Methods

### Public methods

- [`abcnn$new()`](#method-abcnn-new)

- [`abcnn$fit()`](#method-abcnn-fit)

- [`abcnn$predict()`](#method-abcnn-predict)

- [`abcnn$dataloader()`](#method-abcnn-dataloader)

- [`abcnn$conformal_prediction()`](#method-abcnn-conformal_prediction)

- [`abcnn$predictions()`](#method-abcnn-predictions)

- [`abcnn$summary()`](#method-abcnn-summary)

- [`abcnn$plot_training()`](#method-abcnn-plot_training)

- [`abcnn$plot_prediction()`](#method-abcnn-plot_prediction)

- [`abcnn$plot_posterior()`](#method-abcnn-plot_posterior)

- [`abcnn$draw_from_posterior()`](#method-abcnn-draw_from_posterior)

- [`abcnn$cross_validation()`](#method-abcnn-cross_validation)

- [`abcnn$plot_cross_validation()`](#method-abcnn-plot_cross_validation)

- [`abcnn$clone()`](#method-abcnn-clone)

------------------------------------------------------------------------

### Method `new()`

Create a new `abcnn` object

#### Usage

    abcnn$new(
      theta,
      sumstat,
      observed,
      model = NULL,
      method = "concrete dropout",
      scale_input = "none",
      scale_target = "none",
      num_hidden_layers = 3,
      num_hidden_dim = 128,
      validation_split = 0.1,
      num_conformal = 1000,
      credible_interval_p = 0.95,
      test_split = 0.1,
      dropout = 0.5,
      batch_size = 128,
      epochs = 20,
      early_stopping = FALSE,
      verbose = TRUE,
      patience = 4,
      optimizer = torch::optim_adam,
      learning_rate = 0.001,
      l2_weight_decay = 1e-05,
      variance_clamping = c(-1e+15, 1e+15),
      loss = torch::nn_mse_loss(),
      abc_method = "loclinear",
      tol = 0.1,
      abc_keep_original_sumstats = FALSE,
      num_posterior_samples = 1000,
      prior_length_scale = 1e-04,
      weight_regularizer = 1e-06,
      dropout_regularizer = 1e-05,
      num_networks = 5,
      epsilon_adversarial = 0,
      ncores = 1,
      seed = round(runif(1, 0, 10000), digits = 0)
    )

#### Arguments

- `theta`:

  parameters of the pseudo-observed samples (i.e. simulations)

- `sumstat`:

  summary statistics of the pseudo-observed samples (i.e. simulations)

- `observed`:

  summary statistics of the observed samples

- `model`:

  a `luz` model

- `method`:

  the ABC-NN method used

- `scale_input`:

  the scaling method for summary statistics, whether `minmax`,
  `robustscaler`, `normalization` or `none`

- `scale_target`:

  the scaling method for targets (i.e. theta), whether `minmax`,
  `robustscaler`, `normalization` or `none`

- `num_hidden_layers`:

  number of hidden layers in the neural network

- `num_hidden_dim`:

  number of hidden dimensions (neurons) in each hidden layer

- `validation_split`:

  proportion of samples retained for validation at the end of training

- `num_conformal`:

  number of samples retained for conformal prediction

- `credible_interval_p`:

  significance level for the credible interval, between 0 and 1

- `test_split`:

  proportion of samples retained for test at each training iteration

- `dropout`:

  dropout rate

- `batch_size`:

  batch size

- `epochs`:

  number of epochs for training

- `early_stopping`:

  whether to do early stopping (not implemented for the method 'deep
  ensemble')

- `verbose`:

  whether to print messages

- `patience`:

  patience hyperparameter for early stopping. See
  [`luz::luz_callback_early_stopping()`](https://mlverse.github.io/luz/reference/luz_callback_early_stopping.html)
  (not implemented yet for `deep ensemble`)

- `optimizer`:

  `torch` optimizer nn module

- `learning_rate`:

  learning rate

- `l2_weight_decay`:

  L2 weigth decay (regularization)

- `variance_clamping`:

  `c(min, max)` values for variance clamping during training

- `loss`:

  `torch` nn loss function

- `abc_method`:

  ABC sampling method in `abc` function (only for `tabnet-abc`)

- `tol`:

  tolerance rate for `abc` functions (only for `tabnet-abc`)

- `abc_keep_original_sumstats`:

  (logical or numeric) Whether to merge the new set of summary
  statistics with the original ones (TRUE), or just keep the new ones
  (FALSE, default value). If a proportion p (\> 0 and \< 1) is given,
  then the original summary statistics with a relative importance \> p
  are kept. If an integer n \>= 1 is given, then the n most important
  original summary statistics are kept. The variable importances are the
  one computed with `tabnet`.

- `num_posterior_samples`:

  number of samples to generate for the posterior distribution

- `prior_length_scale`:

  prior length scale hyperparameter value

- `weight_regularizer`:

  `concrete dropout` regularization term for weights

- `dropout_regularizer`:

  `concrete dropout` regularization term for dropout

- `num_networks`:

  number of networks in `deep ensemble`

- `epsilon_adversarial`:

  the amount of perturbation for adversarial training in `deep ensemble`
  (experimental)

- `ncores`:

  number of cores for parallelized steps

- `seed`:

  a random seed

- `callbacks`:

  custom callbacks

------------------------------------------------------------------------

### Method `fit()`

Train the neural network

The neural network is trained with `luz` and `torch`

#### Usage

    abcnn$fit()

------------------------------------------------------------------------

### Method [`predict()`](https://rdrr.io/r/stats/predict.html)

Predict parameters from a vector/array of observed summary statistics

Predict theta for the observed summary statistics. Conformal prediction
is also performed at this step on an independent calibration set.

#### Usage

    abcnn$predict(data = NULL)

#### Arguments

- `data`:

  a new set of data to predict

------------------------------------------------------------------------

### Method [`dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html)

Prepare the torch dataloader from sumstat/theta (input/target)

Build and return a dataloader object

#### Usage

    abcnn$dataloader()

------------------------------------------------------------------------

### Method `conformal_prediction()`

Estimate a calibrated credible interval with Conformal Prediction

#### Usage

    abcnn$conformal_prediction()

------------------------------------------------------------------------

### Method `predictions()`

Returns a tidy tibble with predictions and credible intervals

#### Usage

    abcnn$predictions()

------------------------------------------------------------------------

### Method [`summary()`](https://rdrr.io/r/base/summary.html)

Print a summary of the `abcnn` object

#### Usage

    abcnn$summary()

------------------------------------------------------------------------

### Method `plot_training()`

Plot the training curves (training/validation)

#### Usage

    abcnn$plot_training(discard_first = FALSE)

#### Arguments

- `discard_first`:

  Discard the first epoch, as it may have a large loss compared to next
  ones (for plotting only)

------------------------------------------------------------------------

### Method `plot_prediction()`

Plot predicted values and their credible intervals

#### Usage

    abcnn$plot_prediction(
      uncertainty_type = "conformal",
      epistemic_uncertainty = TRUE,
      plot_type = "line"
    )

#### Arguments

- `uncertainty_type`:

  The type of uncertainty to plot, whether `conformal` credible
  intervals (default), the `uncertainty` estimated (square root of the
  variance) or the `posterior quantile`, that are credible intervals
  computed on the distribution of posteriors.

- `epistemic_uncertainty`:

  logical. Whether to plot the epistemic uncertainty in addition to
  overall uncertainty.

- `plot_type`:

  The type of plot, whether a `line` or `errorbar` around points

------------------------------------------------------------------------

### Method `plot_posterior()`

Plot the distributions of estimates and predictions

#### Usage

    abcnn$plot_posterior(
      sample = 1,
      prior = TRUE,
      uncertainty_type = "conformal",
      epistemic_uncertainty = TRUE
    )

#### Arguments

- `sample`:

  Index of the sample to plot

- `prior`:

  logical, whether to plot the prior underneath the posterior and
  prediction

- `uncertainty_type`:

  The type of uncertainty to plot, whether `conformal` credible
  intervals (default), the `uncertainty` estimated (square root of the
  variance) or the `posterior quantile`, that are credible intervals
  computed on the distribution of posteriors.

- `epistemic_uncertainty`:

  logical. Whether to plot the epistemic uncertainty in addition to
  overall uncertainty.

------------------------------------------------------------------------

### Method `draw_from_posterior()`

Draw random samples from the posterior distribution

#### Usage

    abcnn$draw_from_posterior(n = 1)

#### Arguments

- `n`:

  the number of samples to draw from posterior

------------------------------------------------------------------------

### Method `cross_validation()`

Compute cross-validation metrics by comparing ground truth and
predictions on unseen pseudo-observed data (i.e. simulations)

Metrics:

- `n` number of cross-validation samples

- `mae` mean absolute error

- `mse` mean squared error

- `rmse` root mean squared error

- `nmae` normalized mean absolute error

- `cor` Spearman correlation coefficient

- `cov` covariance

- `mean_epistemic_interval` mean epistemic conformal credible interval

- `mean_overall_interval` mean overall conformal credible interval

#### Usage

    abcnn$cross_validation(
      cross_validation_param = NULL,
      cross_validation_sumstats = NULL
    )

#### Arguments

- `cross_validation_param`:

  the parameters in unseen simulations on which to compute
  cross-validation (if `cross_validation_data` is not provided, the
  function returns the `cross_validation_predictions` already computed)

- `cross_validation_sumstats`:

  the summary statistics of unseen simulations on which to compute
  cross-validation (if `cross_validation_data` is not provided, the
  function returns the `cross_validation_predictions` already computed)

------------------------------------------------------------------------

### Method `plot_cross_validation()`

Plot the cross-validation scatter plot

#### Usage

    abcnn$plot_cross_validation()

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    abcnn$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
if (FALSE) { # \dontrun{
# Load test data
df = readRDS("inst/extdata/test_data.Rds")

theta = df$train_y
sumstats = df$train_x
observed = df$observed_y

# Create an `abcnn` object
abc = abcnn$new(theta,
                sumstats,
                observed,
                method = "concrete dropout",
                scale_input = "none",
                scale_target = "none",
                num_hidden_layers = 1,
                num_hidden_dim = 128,
                epochs = 30,
                batch_size = 32)
 abc$fit()

 abc$predict()
 } # }


```

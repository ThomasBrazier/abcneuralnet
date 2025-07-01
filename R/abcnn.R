#' An `abcnn` R6 class object for parameter inference with Bayesian Neural Networks and Approximate Bayesian Computation
#'
#' @param observed a vector of summary statistics computed on the data
#' @param theta a vector, matrix or data frame of the simulated theta_parameter values
#' @param sumstat a vector, matrix or data frame of the simulated summary statistics
#' @param method either `monte carlo dropout`, `concrete dropout`, `tabnet-abc` or `deep ensemble`, See `details`
#' @param scale_input the method to scale summary statistics before training (`none` (default), `minmax` or `robustscaler`)
#' @param scale_target the method to scale the parameter to estimate before training (`none` (default), `minmax` or `robustscaler`)
#' @param num_hidden_layers the number of hidden layers in the Neural Network (default = 3)
#' @param num_hidden_dim the dimension of hidden layers (i.e. number of neurons) in each layer of the Neural Network (default=128)
#' @param validation_split the proportion of samples retained for validation in `luz
#' @param test_split the proportion of samples retained for evaluation in `luz
#' @param dropout the dropout rate for `monte carlo dropout`, i.e. the proportion of neurons dropped in each layer (must be between 0.1 and 0.5)
#' @param batch_size the mini-batch size
#' @param learning_rate the learning rate
#' @param epochs the number of epochs
#' @param early_stopping logical, whether to use early stopping or not
#' @param patience the patience (number of iterations) before early stopping
#' @param optimizer a "torch_optimizer_generator", the optimizer to use in `luz` (default=optim_adam)
#' @param loss a custom loss function passed to the ``monte carlo dropout` method (default=nn_mse_loss())
#' @param l2_weight_decay the L2 weigth decay value for L2 regularization
#' @param tol the tolerance rate in `abc` for the `tabnet-abc` method (`tolerance`). The required proportion of points accepted nearest the target values.
#' @param abc_method a character string indicating the type of ABC algorithm to be applied. Possible values are "rejection", "loclinear", "neuralnet" and "ridge".
#' @param num_posterior_samples the number of posterior samples to predict with the `concrete dropout` and `monte carlo dropout` methods
#' @param credible_interval_p the alpha value for the quantile credible interval (default=0.95, with alpha/2 and 1 - alpha/2 quantiles)
#' @param num_conformal the number of training samples retained for Conformal Prediction (default=1,000)
#' @param kernel the kernel function, either `rbf` or `epanechnikov`
#' @param sampling the ABC sampling function, either `rejection` or `importance`
#' @param length_scale the length scale parameter of the `rbf` kernel
#' @param bandwidth the bandwidth hyperparameter for the kernel smoothing function, either a numeric value or set to "max" if the bandwidth must be estimated as the maximum distance value
#' @param prior_length_scale the prior length scale hyperparameter for the `concrete dropout` method
#' @param num_net the number of iterations in the `regression adjustment` approach
#' @param epsilon_adversarial the multiplying factor for perturbed inputs in adversarial training in the `deep ensemble` method (EXPERIMENTAL). The perturbation is a random noise in the range `[-beta, beta]`, with beta = epsilon_adversarial * variance. Set NULL to disable adversarial training (default). Otherwise 0.01 is a good value to start with.
#' @param variance_clamping clamp all elements in the network weigths in the range `[min, max]`. See `torch_clamp()`.
#' @param verbose logical, whether to print messages and progress bars
#' @param ncores integer, the number of cores in parallel operations
#'
#'
#' @description
#'
#' `abcnn` constructs a `R6` class object for parameter inference with ABC and neural networks.
#' It implements four different method mixing ABC and neural networks implemented in R `torch`.
#'
#'
#' The `initialize` function (`abcnn$new()`) takes as arguments three data frames of training summary statistics,
#' training theta values and observed summary statistics. Public slots can be accessed and modified.
#' A new `abcnn` object is created with `abcnn$new()`.
#'
#'
#'
#' @details
#'
#' Four methods are available for parameter inference. The two core methods are `concrete dropout`,
#' an implementation of Gal et al. (2017), and `deep ensemble`, an implementation
#' of Lakshminarayanan et al. (2017), that allow to estimate both the aleatoric and epistemic uncertainty
#' for each sample. `monte carlo dropout` is an implementation of Gal and Ghahramani (2016),
#' that provides a simpler model that is easier to train, despite its limitations (the dropout rate must be arbitrary chosen).
#'
#'
#' A fourth method is `tabnet-abc`. This is a new method, combining regular ABC inference with the `abc` R package,
#' and a Tabnet neural network, as in Arik et al. (2021) and implemented in the `tabnet` R package.
#' This is the same idea than in Åkesson et al. (2021) or Jiang et al. (2017), except than te MLP/CNN used to estimate summary statistics
#' is replaced by a `tabnet` model specifically designed to handle tabular data and feature selection through
#' an attention map on features. The `tabnet` neural network is trained to predict summary statistics from the observed summary statistics.
#' Then these predictions are used as a supplementary set of summary statistics and regular ABC inference is performed on it.
#' Explain methods are specific the `tabnet-abc` model.
#'
#'
#' In addition, the credible interval is calibrated with conformal prediction, as in Baragatti et al. (2024).
#' As it requires a proxy of uncertainty, conformal prediction is only available for `concrete dropout`,
#' `deep ensemble` and `monte carlo dropout` (only for the epistemic uncertainty for this last method).
#'
#'
#' The neural networks are implemented with the `torch` R package and support CUDA devices for training.
#' The `luz` package is used as a higher level API for training and predictions with `torch`.
#' The device (`CUDA` or `cpu`) is automatically detected by `luz`.
#'
#'
#' The `abcnn` object has public methods to perform each inference step and visualizations.
#'
#' * `new()` to create a new `abcnn` object
#' * `fit()` to fit a neural network
#' * `predict()` to compute conformal predictions from the fitted model
#' * `summary()` to print a summary of the `abcnn` object
#' * `predictions()` to print predictions
#' * `plot_training()` to plot the training curves
#' * `plot_prediction()` to plot all predictions with their credible intervals
#' * `plot_posterior()` to plot the prior and posterior distributions, with the mean and credible intervals, of a single sample
#'
#'
#' The hyperparameters of the neural network can be configured in the `new()` method
#' or modified directly in the corresponding public `slot`.
#'
#' @references
#' \insertRef{baragatti2024approximate}{abcneuralnet}
#' \insertRef{gal2016}{abcneuralnet}
#' \insertRef{gal2017concrete}{abcneuralnet}
#' \insertRef{lakshminarayanan2017simple}{abcneuralnet}
#' \insertRef{arik2021tabnet}{abcneuralnet}
#' \insertRef{tabnet}{abcneuralnet}
#' \insertRef{aakesson2021convolutional}{abcneuralnet}
#' \insertRef{jiang2017learning}{abcneuralnet}
#'
#' @examples
#' \dontrun{
#' # Load test data
#' df = readRDS("inst/extdata/test_data.Rds")
#'
#' theta = df$train_y
#' sumstats = df$train_x
#' observed = df$observed_y
#'
#' # Create an `abcnn` object
#' abc = abcnn$new(theta,
#'                 sumstats,
#'                 observed,
#'                 method = "concrete dropout",
#'                 scale_input = "none",
#'                 scale_target = "none",
#'                 num_hidden_layers = 3,
#'                 num_hidden_dim = 128,
#'                 epochs = 30,
#'                 batch_size = 32)
#'  abc$fit()
#'
#'  abc$predict()
#'  }
#'
#'
#'
#' @slot theta parameters of the pseudo-observed samples (i.e. simulations)
#' @slot sumstat summary statistics of the pseudo-observed samples (i.e. simulations)
#' @slot observed summary statistics of the observed samples
#' @slot model the `luz` model
#' @slot method the ABC-NN method used
#' @slot scale_input the scaling method for summary statistics
#' @slot scale_target the scaling method for targets (i.e. theta)
#' @slot num_hidden_layers number of hidden layers in the neural network
#' @slot num_hidden_dim number of hidden dimensions (neurons) in each hidden layer
#' @slot validation_split proportion of samples retained for validation at the end of training
#' @slot num_conformal number of samples retained for conformal prediction
#' @slot credible_interval_p significance level for the credible interval, between 0 and 1
#' @slot test_split proportion of samples retained for test at each training iteration
#' @slot dropout dropout rate
#' @slot batch_size  batch size
#' @slot epochs number of epochs for training
#' @slot early_stopping whether to do early stopping
#' @slot patience patience hyperparameter for early stopping. See `luz::luz_callback_early_stopping()`
#' @slot callbacks custom callbacks
#' @slot verbose whether to print messages
#' @slot optimizer `torch` optimizer nn module
#' @slot learning_rate learning rate
#' @slot l2_weight_decay L2 weigth decay (regularization)
#' @slot variance_clamping `c(min, max)` values for variance clamping during training
#' @slot loss `torch` nn loss function
#' @slot tol tolerance rate for `abc` functions (only for `tabnet-abc`)
#' @slot abc_method ABC sampling method in `abc` function (only for `tabnet-abc`)
#' @slot num_posterior_samples number of samples to generate for the posterior distribution
#' @slot prior_length_scale prior length scale hyperparameter value
#' @slot weight_regularizer `concrete dropout` regularization term for weights
#' @slot dropout_regularizer `concrete dropout` regularization term for dropout
#' @slot num_networks number of networks in `deep ensemble`
#' @slot epsilon_adversarial the amount of perturbation for adversarial training in `deep ensemble`
#' @slot device `luz`/`torch` device for tensors
#' @slot input_dim number of input dimensions of the neural network
#' @slot output_dim number of output dimensions of the neural network
#' @slot n_train number of training samples
#' @slot sumstat_names names of summary statistics
#' @slot output_names output names
#' @slot theta_names names of theta to estimate
#' @slot n_obs number of observed samples
#' @slot prior_lower lower boundary of priors (for figures)
#' @slot prior_upper upper boundary of priors (for figures)
#' @slot fitted the fitted `luz` model
#' @slot evaluation the evaluation metric
#' @slot eval_metrics `torch` nn metrics for evaluation
#' @slot posterior_samples array of posterior samples
#' @slot quantile_posterior quantiles computed on the posterior samples, given the `credible_interval_p`
#' @slot predictive_mean values predicted by the model for each observed sample
#' @slot aleatoric_uncertainty aleatoric uncertainty for each observed sample
#' @slot epistemic_uncertainty epistemic uncertainty for each observed sample
#' @slot overall_uncertainty overall uncertainty for each observed sample (epistemic + aleatoric)
#' @slot epistemic_conformal_quantile the quantile factor to get the conformalized credible interval for epistemic uncertainty
#' @slot overall_conformal_quantile the quantile factor to get the conformalized credible interval for overall uncertainty
#' @slot dropout_rates the dropout rate hyperparameter estimated by concrete dropout
#' @slot input_summary statistics computed on input data (for scaling)
#' @slot target_summary statistics computed on target data (for scaling)
#' @slot sumstat_adj adjusted training summary statistics after scaling
#' @slot observed_adj adjusted observed summary statistics after scaling
#' @slot theta_adj adjusted training theta after scaling
#' @slot calibration_theta adjusted theta for conformal prediction after scaling (calibration set)
#' @slot calibration_sumstat adjusted summary statistics for conformal prediction after scaling (calibration set)
#' @slot ncores number of cores for parallelized steps
#'
#'
#' @return A `R6::abcnn` object
#'
#' @import torch
#' @import luz
#' @import ggplot2
#' @import tidyr
#' @import dplyr
#' @import tibble
#' @import R6
#' @import RColorBrewer
#' @import tabnet
#' @import abc
#'
#' @importFrom Rdpack reprompt
#'
#' @return an `abcnn` object that can be used to fit(), predict() and plot predictions
#'
#' @seealso [R6::R6()]
#'
#' @export
#'
abcnn = R6::R6Class("abcnn",
  public = list(
    #' @field theta parameters of the pseudo-observed samples (i.e. simulations)
    theta = NULL,
    #' @field sumstat summary statistics of the pseudo-observed samples (i.e. simulations)
    sumstat = NULL,
    #' @field observed summary statistics of the observed samples
    observed = NULL,
    #' @field model the `luz` model
    model=NULL,
    #' @field method the ABC-NN method used, whether `tabnet-abc`, `monte carlo dropout`, `concrete dropout` or `deep ensemble`
    method='concrete dropout',
    #' @field scale_input the scaling method for summary statistics
    scale_input=NULL,
    #' @field scale_target the scaling method for targets (i.e.e theta)
    scale_target=NULL,
    #' @field num_hidden_layers number of hidden layers in the neural network
    num_hidden_layers=NA,
    #' @field num_hidden_dim number of hidden dimensions (neurons) in each hidden layer
    num_hidden_dim=NA,
    #' @field validation_split proportion of training samples to retain for validation at the end of training
    validation_split=NA,
    #' @field num_conformal number of training samples to retain for conformal prediction (not used during training)
    num_conformal=NA,
    #' @field credible_interval_p proportion, the level of significance for credible intervals
    credible_interval_p = NA,
    #' @field test_split proportion of training samples to retain for testing (at each training iteration)
    test_split=NA,
    #' @field dropout dropout rate to apply in `monte carlo dropout`
    dropout=NA,
    #' @field batch_size batch size in `luz`
    batch_size=NA,
    #' @field epochs number of epochs for training
    epochs=NA,
    #' @field early_stopping logical, whether to do early stopping in `luz`
    early_stopping=FALSE,
    #' @field callbacks list of `luz` callbacks
    callbacks=NULL,
    #' @field verbose logical, whether to print messages and progress bars for the user
    verbose=NULL,
    #' @field patience patience hyperparameter for `luz``early stopping`, the number of epochs without improving until stoping training
    patience=4,
    #' @field optimizer `torch` custom optimizer
    optimizer=NULL,
    #' @field learning_rate learningrate in `luz`
    learning_rate=0.001,
    #' @field l2_weight_decay L2 weight decay for regularization in the `torch::optimizer`
    l2_weight_decay=1e-5,
    #' @field variance_clamping `c(min, max)` values for variance clamping during training
    variance_clamping=TRUE,
    #' @field loss custom `torch` loss function (nn module)
    loss=NULL,
    #' @field tol tolerance rate in `abc` for the `tabnet-abc` method
    tol=NULL,
    #' @field abc_method `abc` method for `tabnet-abc`
    abc_method=NULL,
    #' @field num_posterior_samples number of posterior samples to generate in `monte carlo dropout` and `concrete dropout`
    num_posterior_samples=1000,
    #' @field prior_length_scale hyperparameter for `concrete dropout`
    prior_length_scale=1e-4,
    #' @field weight_regularizer hyperparameter for `concrete dropout`
    weight_regularizer=NA,
    #' @field dropout_regularizer hyperparameter for `concrete dropout`
    dropout_regularizer=NA,
    #' @field num_networks number of neural networks in `deep ensemble`
    num_networks=5,
    #' @field epsilon_adversarial the factor by which perturbating training samples for adversarial training in `deep ensemble`
    epsilon_adversarial=0,
    #' @field device device used in `luz` and `torch`, whether 'cpu' or 'cuda' (GPU)
    device="cpu",
    #' @field input_dim number of input dimensions (columns in summary statistics)
    input_dim = NA,
    #' @field output_dim number of output dimensions (columns in theta, the number of variables to predict)
    output_dim = NA,
    #' @field n_train number of samples for training
    n_train = NA,
    #' @field sumstat_names names of the summary statistics
    sumstat_names = NA,
    #' @field output_names neural network output names (mean and variance)
    output_names = NA,
    #' @field theta_names theta names (variables to predict)
    theta_names = NA,
    #' @field n_obs number of observations (rows in 'observed')
    n_obs = NA,
    #' @field prior_lower lower boundaries of priors (for plotting)
    prior_lower = NA,
    #' @field prior_upper upper boundaries of priors (for plotting)
    prior_upper = NA,
    #' @field fitted a model fitted with `luz`
    fitted = NULL,
    #' @field evaluation numerical value of the evaluation metric
    evaluation=NULL,
    #' @field eval_metrics list of custom metrics to use at evaluation (not implemented yet)
    eval_metrics=NA,
    #' @field posterior_samples array of all posterior samples predicted in `monte carlo dropout` and `concrete dropout`
    posterior_samples = NA,
    #' @field quantile_posterior quantiles of the posterior distributions, given the credible interval significance required
    quantile_posterior = NA,
    #' @field predictive_mean mean predicted value
    predictive_mean = NA,
    #' @field aleatoric_uncertainty aleatoric uncertainty
    aleatoric_uncertainty = NA,
    #' @field epistemic_uncertainty epistemic uncertainty
    epistemic_uncertainty = NA,
    #' @field overall_uncertainty overall uncertainty
    overall_uncertainty = NA,
    #' @field epistemic_conformal_quantile conformal quantile of epistemic uncertainty calibrated with conformal prediction
    epistemic_conformal_quantile = NA,
    #' @field overall_conformal_quantile conformal quantile of overall uncertainty calibrated with conformal prediction
    overall_conformal_quantile = NA,
    #' @field dropout_rates dropout rates inferred by `concrete dropout` (not implemented yet)
    dropout_rates = NA,
    #' @field input_summary summary statistics of input data for scaling
    input_summary = NA,
    #' @field target_summary summary statistics of target data (theta) for scaling
    target_summary = NA,
    #' @field sumstat_adj scaled training summary statistics
    sumstat_adj = NA,
    #' @field observed_adj scaled observed summary statistics
    observed_adj = NA,
    #' @field theta_adj scaled training target
    theta_adj = NA,
    #' @field calibration_theta theta saved for calibration with conformal prediction
    calibration_theta = NA,
    #' @field calibration_sumstat summary statistics saved for calibration with conformal prediction
    calibration_sumstat = NA,
    #' @field ncores number of cores for parallel procedures
    ncores = NA,

    #' @description
    #' Create a new `abcnn` object
    #'
    #' @param theta parameters of the pseudo-observed samples (i.e. simulations)
    #' @param sumstat summary statistics of the pseudo-observed samples (i.e. simulations)
    #' @param observed summary statistics of the observed samples
    #' @param model a `luz` model
    #' @param method the ABC-NN method used
    #' @param scale_input the scaling method for summary statistics, whether `minmax`, `robustscaler`, `normalization` or `none`
    #' @param scale_target the scaling method for targets (i.e. theta), whether `minmax`, `robustscaler`, `normalization` or `none`
    #' @param num_hidden_layers number of hidden layers in the neural network
    #' @param num_hidden_dim number of hidden dimensions (neurons) in each hidden layer
    #' @param validation_split proportion of samples retained for validation at the end of training
    #' @param num_conformal number of samples retained for conformal prediction
    #' @param credible_interval_p significance level for the credible interval, between 0 and 1
    #' @param test_split proportion of samples retained for test at each training iteration
    #' @param dropout dropout rate
    #' @param batch_size  batch size
    #' @param epochs number of epochs for training
    #' @param early_stopping whether to do early stopping
    #' @param patience patience hyperparameter for early stopping. See `luz::luz_callback_early_stopping()`
    #' @param callbacks custom callbacks
    #' @param verbose whether to print messages
    #' @param optimizer `torch` optimizer nn module
    #' @param learning_rate learning rate
    #' @param l2_weight_decay L2 weigth decay (regularization)
    #' @param variance_clamping `c(min, max)` values for variance clamping during training
    #' @param loss `torch` nn loss function
    #' @param abc_method ABC sampling method in `abc` function (only for `tabnet-abc`)
    #' @param tol tolerance rate for `abc` functions (only for `tabnet-abc`)
    #' @param num_posterior_samples number of samples to generate for the posterior distribution
    #' @param prior_length_scale prior length scale hyperparameter value
    #' @param weight_regularizer `concrete dropout` regularization term for weights
    #' @param dropout_regularizer `concrete dropout` regularization term for dropout
    #' @param num_networks number of networks in `deep ensemble`
    #' @param epsilon_adversarial the amount of perturbation for adversarial training in `deep ensemble`
    #' @param ncores number of cores for parallelized steps
    #'
    initialize = function(theta,
                          sumstat,
                          observed,
                          model=NULL,
                          method='concrete dropout',
                          scale_input="none",
                          scale_target="none",
                          num_hidden_layers=3,
                          num_hidden_dim=128,
                          validation_split=0.1,
                          num_conformal=1000,
                          credible_interval_p = 0.95,
                          test_split=0.1,
                          dropout=0.5,
                          batch_size=32,
                          epochs=20,
                          early_stopping=FALSE,
                          verbose=TRUE,
                          patience=4,
                          optimizer=torch::optim_adam,
                          learning_rate=0.001,
                          l2_weight_decay=1e-5,
                          variance_clamping=c(-1e15, 1e15),
                          loss=torch::nn_mse_loss(),
                          abc_method="loclinear",
                          tol=NULL,
                          num_posterior_samples=1000,
                          prior_length_scale=1e-4,
                          weight_regularizer = 1e-6,
                          dropout_regularizer = 1e-5,
                          num_networks=5,
                          epsilon_adversarial=0,
                          ncores = 1) {
      #-----------------------------------#
      # CHECK INPUTS
      #-----------------------------------#
      # Input data validation
      if(missing(observed)) stop("'observed' is missing")
      if(missing(theta)) stop("'theta' is missing")
      if(missing(sumstat)) stop("'sumstat' is missing")
      if(!is.data.frame(theta)) stop("'theta' has to be a data.frame with column names.")
      if(!is.data.frame(sumstat)) stop("'sumstat' has to be a data.frame with column names.")
      if(!is.data.frame(observed)) stop("'observed' has to be a data.frame with column names.")
      if(dropout < 0.1 | dropout > 0.5) stop("The 'dropout' rate must be between 0.1 and 0.5.")

      # Device. Use CUDA if available
      self$device = torch::torch_device(if (torch::cuda_is_available()) {"cuda"} else {"cpu"})
      # "When fitting, luz will use the fastest possible accelerator;
      # if a CUDA-capable GPU is available it will be used, otherwise we fall back to the CPU.
      # It also automatically moves data, optimizers,
      # and models to the selected device so you don’t need to handle it manually."

      # TODO Check input dim and type
      # Coerce to data frame
      self$observed = as.data.frame(observed)
      self$theta = as.data.frame(theta)
      self$sumstat = as.data.frame(sumstat)

      # TODO Message to user: num params, num observation, training/test/validation sizes
      # TODO Print dimensions and model

      #-----------------------------------#
      # INIT ATTRIBUTES
      #-----------------------------------#
      self$method=method
      self$scale_input=scale_input
      self$scale_target=scale_target
      self$num_hidden_layers=num_hidden_layers
      self$num_hidden_dim=num_hidden_dim
      self$validation_split=validation_split
      self$test_split=test_split
      self$dropout=dropout
      self$batch_size=batch_size
      self$learning_rate=learning_rate
      self$epochs=epochs
      self$early_stopping=early_stopping
      self$patience=patience
      self$optimizer=optimizer
      self$loss=loss
      self$tol=tol
      self$abc_method=abc_method
      self$num_posterior_samples=num_posterior_samples
      self$l2_weight_decay
      self$epsilon_adversarial = epsilon_adversarial
      self$credible_interval_p = credible_interval_p
      self$variance_clamping = variance_clamping
      self$num_conformal = num_conformal
      self$verbose = verbose
      self$ncores = ncores

      # Init default value for clamping the variance estimate, or let the user set it
      # if (variance_clamping & !is.numeric(variance_clamping)) {
      #   if (method == "concrete dropout") {
      #     self$variance_clamping = c(-10, 10)
      #   }
      #   if (method == "deep ensemble") {
      #     self$variance_clamping = c(1e-6, 1e6)
      #   }
      # } else {
      #   self$variance_clamping = variance_clamping
      # }
      self$input_dim = ncol(self$sumstat)
      self$output_dim = ncol(self$theta)

      # self$n_train = nrow(self$theta)
      self$sumstat_names = colnames(self$observed)
      colnames(self$sumstat) = self$sumstat_names
      self$n_obs = nrow(self$observed)
      self$output_names = NA
      self$theta_names = colnames(self$theta)


      # Fill output slots with NA
      # Init output slots
      self$fitted = NULL
      self$posterior_samples = NA # Raw posterior samples predicted by the NN
      # The mean predicted from MC samples (MC dropout) or pointwise estimate by concrete dropout
      self$predictive_mean = as.data.frame(array(NA, dim = c(nrow(self$observed), self$output_dim)))
      colnames(self$predictive_mean) = colnames(self$theta)
      # Raw epistemic uncertainty, expressed as sd
      self$epistemic_uncertainty = as.data.frame(array(NA, dim = c(nrow(self$observed), self$output_dim)))
      colnames(self$epistemic_uncertainty) = colnames(self$theta)
      # Raw aleatoric uncertainty, expressed as sd
      self$aleatoric_uncertainty = as.data.frame(array(NA, dim = c(nrow(self$observed), self$output_dim)))
      colnames(self$aleatoric_uncertainty) = colnames(self$theta)
      # Overall uncertainty
      self$overall_uncertainty = as.data.frame(array(NA, dim = c(nrow(self$observed), self$output_dim)))
      colnames(self$overall_uncertainty) = colnames(self$theta)

      quantile_posterior = as.data.frame(array(NA, dim = c(nrow(self$observed), self$output_dim)))
      colnames(quantile_posterior) = colnames(self$theta)

      self$quantile_posterior = list(mean = quantile_posterior,
                                     median = quantile_posterior,
                                     posterior_lower_ci = quantile_posterior,
                                     posterior_upper_ci = quantile_posterior)


      epistemic_conformal_quantile = as.data.frame(array(NA, dim = c(1, self$output_dim)))
      colnames(epistemic_conformal_quantile) = colnames(self$theta)

      overall_conformal_quantile = as.data.frame(array(NA, dim = c(1, self$output_dim)))
      colnames(overall_conformal_quantile) = colnames(self$theta)

      self$epistemic_conformal_quantile = epistemic_conformal_quantile
      self$overall_conformal_quantile = overall_conformal_quantile


      self$dropout_rates = NA # The dropout rates hyperparameter estimated by concrete dropout

      #-----------------------------------#
      # CALLBACKS
      #-----------------------------------#
      self$patience = patience
      if (early_stopping) {
        early_stopping_callback = luz::luz_callback_early_stopping(patience = self$patience)
        self$callbacks = list(early_stopping_callback)
      } else {
        self$callbacks = list()
      }

      #-----------------------------------#
      # PRE-PROCESSING
      #-----------------------------------#
      # Set 64-bit default dtype for torch_exp computation
      # torch_set_default_dtype(torch_float64())

      # Keep metadata of prior boundaries for plotting
      self$prior_lower = apply(as.matrix(self$theta), 1, min)
      self$prior_upper = apply(as.matrix(self$theta), 1, max)

      #-----------------------------------#
      # INIT MODELS
      #-----------------------------------#
      if (is.null(model)) {
        if (self$method == "tabnet-abc") {
          self$num_conformal = 0
        }
        if (self$method == "monte carlo dropout") {
          self$model = mc_dropout_model %>%
            luz::setup(optimizer = self$optimizer,
                       loss = self$loss) %>%
            luz::set_hparams(num_input_dim = self$input_dim,
                        num_hidden_dim = self$num_hidden_dim,
                        num_output_dim = self$output_dim,
                        num_hidden_layers = self$num_hidden_layers,
                        dropout_hidden = self$dropout) %>%
            luz::set_opt_hparams(lr = self$learning_rate, weight_decay = self$l2_weight_decay)
        }
        if (self$method == "concrete dropout") {
          # TODO Make utility functions for weight_regularizer and weight_regularizer
          l = self$prior_length_scale
          N = self$n_train
          self$weight_regularizer = l^2 / N
          self$weight_regularizer = 2 / N

          self$model = concrete_model %>%
            luz::setup(optimizer = self$optimizer) %>%
            luz::set_hparams(num_input_dim = self$input_dim,
                        num_hidden_dim = self$num_hidden_dim,
                        num_output_dim = self$output_dim,
                        num_hidden_layers = self$num_hidden_layers,
                        weight_regularizer = self$weight_regularizer,
                        dropout_regularizer = self$weight_regularizer,
                        clamp = self$variance_clamping) %>%
            luz::set_opt_hparams(lr = self$learning_rate, weight_decay = self$l2_weight_decay)
        }
        if (self$method == "deep ensemble") {

          self$model = nn_ensemble %>%
            luz::setup() %>%
            luz::set_hparams (model = single_model,
                         learning_rate = self$learning_rate,
                         weight_decay = self$l2_weight_decay,
                         num_models = self$num_networks,
                         num_input_dim = self$input_dim,
                         num_output_dim = self$output_dim,
                         num_hidden_layers = self$num_hidden_layers,
                         num_hidden_dim = self$num_hidden_dim,
                         epsilon = NULL,
                         clamp = self$variance_clamping)
        }
      } else {
        self$model = model
      }

      # END ON INIT
    },

    #' @description
    #' Train the neural network
    #'
    #' The neural network is trained with `luz` and `torch`
    #'
    fit = function() {

      if (self$verbose) {self$summary()}

      # Load data
      dl = self$dataloader()

      if (self$method == "tabnet-abc") {
        theta = dl$theta_adj
        sumstat = dl$sumstat_adj

        config = tabnet::tabnet_config(optimizer = "adam",
                               batch_size = self$batch_size,
                               verbose = self$verbose,
                               drop_last = TRUE,
                               early_stopping_patience = self$patience)

        self$fitted = tabnet::tabnet_fit(sumstat,
                                 theta,
                                 epochs = self$epochs,
                                 valid_split = self$validation_split,
                                 learn_rate = self$learning_rate,
                                 config = config)
      }

      if (self$method == 'monte carlo dropout') {
        # Load data
        # dl = self$dataloader()

        # Fit
        self$fitted = self$model %>%
          luz::fit(dl$train,
              epochs = self$epochs,
              valid_data = dl$valid,
              callbacks = self$callbacks)
        # self$model = self$fitted$model

      }

      if (self$method == 'concrete dropout') {
        # Load data
        # dl = self$dataloader()

        # Fit
        self$fitted = self$model %>%
          luz::fit(dl$train,
              epochs = self$epochs,
              valid_data = dl$valid,
              callbacks = self$callbacks)

        # self$model = self$fitted$model

        self$dropout_rates = self$fitted$model$p
      }

      if (self$method == 'deep ensemble') {
        # Load data
        # dl = self$dataloader()

        # The range of noise to add to perturbed inputs in adversarial training
        if (is.null(self$epsilon_adversarial)) {
          epsilon = 0
        } else {
          if (is.na(self$epsilon_adversarial)) {
            epsilon = 0
          } else {
            epsilon = self$epsilon_adversarial
          }
        }

        # Fit
        # Redefine model with epsilon based on training data
        self$fitted = nn_ensemble %>%
          luz::setup() %>%
          luz::set_hparams (model = single_model,
                       learning_rate = self$learning_rate,
                       weight_decay = self$l2_weight_decay,
                       num_models = self$num_networks,
                       num_input_dim = self$input_dim,
                       num_output_dim = self$output_dim,
                       num_hidden_layers = self$num_hidden_layers,
                       num_hidden_dim = self$num_hidden_dim,
                       epsilon = epsilon,
                       clamp = self$variance_clamping) %>%
          luz::fit(dl$train,
                   epochs = self$epochs,
                   valid_data = dl$valid,
                   callbacks = self$callbacks)
        # self$model = self$fitted$model
      }
      # self$model = self$fitted$model

      # Evaluation
      # Plot training
      # plot(self$fitted)

      # Monitor loss and metrics
      # metrics = get_metrics(self$fitted)
      # print(metrics)

      # TODO luz::evaluate currently not working with Deep Ensemble
      # fitted returns n values named value.x instead of a single value
      if (self$method == "monte carlo dropout" | self$method == "concrete dropout") {
        print("Evaluation")
        self$evaluation = self$fitted %>% luz::evaluate(data = dl$test)
        self$eval_metrics = luz::get_metrics(self$evaluation)
        print(self$eval_metrics)
        print(self$evaluation)
      }
    },

    #' @description
    #' Predict parameters from a vector/array of observed summary statistics
    #'
    #' Predict theta for the observed summary statistics.
    #' Conformal prediction is also performed at this step on an independent calibration set.
    #'
    #' @param data a new set of data to predict
    #'
    predict = function(data = NULL) {

      if (!is.null(data)) {
        # Replace data with the new set
        self$observed = data
        self$observed_adj = scaler(self$observed,
                                   self$input_summary,
                                   method = self$scale_input,
                                   type = "forward")
        self$n_obs = nrow(observed)
      }

      observed = torch::torch_tensor(as.matrix(self$observed_adj), device = self$device)

      if (self$verbose) {print(paste0("Making predictions with ", nrow(observed), " samples."))}

      if (is.null(self$fitted)) {
        warning("The model has not been fitted. NAs returned.")
      } else {

        if (self$method == "tabnet-abc") {
          # Predict theta with Tabnet and use it as summary statistics
          tabnet_train = predict(self$fitted, self$sumstat_adj)
          tabnet_observed = predict(self$fitted, self$observed_adj)

          new_sumstats_train = cbind(tabnet_train, self$sumstat_adj)
          new_sumstats_observed = cbind(tabnet_observed, self$observed_adj)

          self$num_posterior_samples = nrow(new_sumstats_train) * self$tol
          nsamples = nrow(self$observed_adj)
          ndim = ncol(self$theta_adj)
          mc_samples = array(0, dim = c(self$num_posterior_samples, nsamples, ndim))

          pb = txtProgressBar(min = 1, max = nrow(self$observed_adj), style = 3)
          for (i in 1:nrow(self$observed_adj)) {
            setTxtProgressBar(pb, i)

            suppressWarnings({abc_res = abc::abc(new_sumstats_observed[i,],
                               self$theta_adj,
                               new_sumstats_train,
                               tol = self$tol,
                               method = self$abc_method)})

            if (self$abc_method == "rejection") {
              mc_samples[,i,] = abc_res$unadj.values
            } else {
              mc_samples[,i,] = abc_res$adj.values
            }

          }
          close(pb)

          self$posterior_samples = mc_samples

          # average over the MC samples
          # If more than one parameter to estimate
          # TODO Generalize to any number of output dim
          means = mc_samples[, , 1:self$output_dim, drop = F]

          if (self$output_dim > 1) {
            # Lines are observations
            # Columns are parameters
            predictive_mean = apply(means, 3, function(x) apply(x, 2, mean))

            posterior_median = apply(means, 3, function(x) apply(x, 2, median))
            posterior_lower_ci = apply(means, 3, function(x) apply(x, 2, function(x) quantile(x, (1 - self$credible_interval_p)/2)))
            posterior_upper_ci = apply(means, 3, function(x) apply(x, 2, function(x) quantile(x, (self$credible_interval_p + (1 - self$credible_interval_p)/2))))

            epistemic_uncertainty = apply(means, 3, function(x) apply(x, 2, var))
          } else {
            predictive_mean = apply(means, 2, mean)

            posterior_median = apply(means, 2, median)
            posterior_lower_ci = apply(means, 2, function(x) quantile(x, (1 - self$credible_interval_p)/2))
            posterior_upper_ci = apply(means, 2, function(x) quantile(x, (self$credible_interval_p + (1 - self$credible_interval_p)/2)))

            epistemic_uncertainty = apply(means, 2, var)
          }

          predictive_mean = as.data.frame(array(predictive_mean, dim = c(nsamples, ndim)))
          colnames(predictive_mean) = colnames(self$theta)
          self$predictive_mean = predictive_mean

          epistemic_uncertainty = as.data.frame(array(epistemic_uncertainty, dim = c(nsamples, ndim)))
          colnames(epistemic_uncertainty) = colnames(self$theta)
          self$epistemic_uncertainty = epistemic_uncertainty

          aleatoric_uncertainty = as.data.frame(array(NA, dim = c(nsamples, ndim)))
          colnames(aleatoric_uncertainty) = colnames(self$theta)
          self$aleatoric_uncertainty = aleatoric_uncertainty

          self$overall_uncertainty = self$epistemic_uncertainty + self$aleatoric_uncertainty

          posterior_median = as.data.frame(array(posterior_median, dim = c(nsamples, ndim)))
          colnames(posterior_median) = colnames(self$theta)

          posterior_lower_ci = as.data.frame(array(posterior_lower_ci, dim = c(nsamples, ndim)))
          colnames(posterior_lower_ci) = colnames(self$theta)

          posterior_upper_ci = as.data.frame(array(posterior_upper_ci, dim = c(nsamples, ndim)))
          colnames(posterior_upper_ci) = colnames(self$theta)

          self$quantile_posterior = list(mean = predictive_mean,
                                         median = posterior_median,
                                         posterior_lower_ci = posterior_lower_ci,
                                         posterior_upper_ci = posterior_upper_ci)

        }

        if (self$method == 'monte carlo dropout') {

          # Set model to evaluation mode
          # TODO Implement this directly in the torch module
          self$fitted$model$eval()

          # MC dropout predictions
          # Approximate posterior samples in lines (axis 1)
          # Each observation is a column (axis 2)
          # Mean prediction on axis 3, multiplied by the number of parameters to estimate
          pb = txtProgressBar(min = 1, max = self$num_posterior_samples, style = 3)

          mc_samples = array(0, dim = c(self$num_posterior_samples, observed$shape[1], self$output_dim))
          for (k in 1:self$num_posterior_samples) {
            preds = self$fitted$model(observed)
            mc_samples[k, , 1:self$output_dim] = as.array(preds)
            setTxtProgressBar(pb, k)
          }
          close(pb)

          self$posterior_samples = mc_samples

          means = mc_samples[, , 1:self$output_dim, drop = F]

          # average over the MC samples
          # If more than one parameter to estimate
          # TODO Generalize to any number of output dim
          if (self$output_dim > 1) {
            # Lines are observations
            # Columns are parameters
            predictive_mean = apply(means, 3, function(x) apply(x, 2, mean))

            posterior_median = apply(means, 3, function(x) apply(x, 2, median))
            posterior_lower_ci = apply(means, 3, function(x) apply(x, 2, function(x) quantile(x, (1 - self$credible_interval_p)/2)))
            posterior_upper_ci = apply(means, 3, function(x) apply(x, 2, function(x) quantile(x, (self$credible_interval_p + (1 - self$credible_interval_p)/2))))

            epistemic_uncertainty = apply(means, 3, function(x) apply(x, 2, var))
          } else {
            predictive_mean = apply(means, 2, mean)

            posterior_median = apply(means, 2, median)
            posterior_lower_ci = apply(means, 2, function(x) quantile(x, (1 - self$credible_interval_p)/2))
            posterior_upper_ci = apply(means, 2, function(x) quantile(x, (self$credible_interval_p + (1 - self$credible_interval_p)/2)))

            epistemic_uncertainty = apply(means, 2, var)
          }

          predictive_mean = as.data.frame(array(predictive_mean, dim = c(observed$shape[1], self$output_dim)))
          colnames(predictive_mean) = colnames(self$theta)
          self$predictive_mean = predictive_mean

          epistemic_uncertainty = as.data.frame(array(epistemic_uncertainty, dim = c(observed$shape[1], self$output_dim)))
          colnames(epistemic_uncertainty) = colnames(self$theta)
          self$epistemic_uncertainty = epistemic_uncertainty

          aleatoric_uncertainty = as.data.frame(array(NA, dim = c(observed$shape[1], self$output_dim)))
          colnames(aleatoric_uncertainty) = colnames(self$theta)
          self$aleatoric_uncertainty = aleatoric_uncertainty

          self$overall_uncertainty = self$epistemic_uncertainty + self$aleatoric_uncertainty

          posterior_median = as.data.frame(array(posterior_median, dim = c(observed$shape[1], self$output_dim)))
          colnames(posterior_median) = colnames(self$theta)

          posterior_lower_ci = as.data.frame(array(posterior_lower_ci, dim = c(observed$shape[1], self$output_dim)))
          colnames(posterior_lower_ci) = colnames(self$theta)

          posterior_upper_ci = as.data.frame(array(posterior_upper_ci, dim = c(observed$shape[1], self$output_dim)))
          colnames(posterior_upper_ci) = colnames(self$theta)

          self$quantile_posterior = list(mean = predictive_mean,
                                         median = posterior_median,
                                         posterior_lower_ci = posterior_lower_ci,
                                         posterior_upper_ci = posterior_upper_ci)
        }

        if (self$method == 'concrete dropout') {
          pb = txtProgressBar(min = 1, max = self$num_posterior_samples, style = 3)

          mc_samples = array(0, dim = c(self$num_posterior_samples, observed$shape[1], 2 * self$output_dim))
          for (k in 1:self$num_posterior_samples) {
            preds = self$fitted$model(observed)
            mc_samples[k, , ] = cbind(as.matrix(preds[1]), as.matrix(preds[2]))
            setTxtProgressBar(pb, k)
          }
          close(pb)

          # the means are in the first output column
          means = mc_samples[, , 1:self$output_dim, drop = F]
          logvar = mc_samples[, , (self$output_dim + 1):(self$output_dim * 2), drop = F]

          self$posterior_samples = mc_samples
          self$output_names = unlist(lapply(c("mu", "sigma"), function(x) paste(colnames(self$theta), x, sep = "_")))

          # average over the MC samples
          # If more than one parameter to estimate
          # TODO Generalize to any number of output dim
          if (self$output_dim > 1) {
            # Lines are observations
            # Columns are parameters
            predictive_mean = apply(means, 3, function(x) apply(x, 2, mean))
            epistemic_uncertainty = apply(means, 3, function(x) apply(x, 2, var))
            aleatoric_uncertainty = apply(logvar, 3, function(x) exp(colMeans(x)))

            posterior_median = apply(means, 3, function(x) apply(x, 2, median))
            posterior_lower_ci = apply(means, 3, function(x) apply(x, 2, function(x) quantile(x, (1 - self$credible_interval_p)/2)))
            posterior_upper_ci = apply(means, 3, function(x) apply(x, 2, function(x) quantile(x, (self$credible_interval_p + (1 - self$credible_interval_p)/2))))

          } else {
            predictive_mean = apply(means, 2, mean)
            epistemic_uncertainty = apply(means, 2, var)
            aleatoric_uncertainty = exp(colMeans(logvar))

            posterior_median = apply(means, 2, median)
            posterior_lower_ci = apply(means, 2, function(x) quantile(x, (1 - self$credible_interval_p)/2))
            posterior_upper_ci = apply(means, 2, function(x) quantile(x, (self$credible_interval_p + (1 - self$credible_interval_p)/2)))
          }


          predictive_mean = as.data.frame(array(predictive_mean, dim = c(observed$shape[1], self$output_dim)))
          colnames(predictive_mean) = colnames(self$theta)
          self$predictive_mean = predictive_mean

          epistemic_uncertainty = as.data.frame(array(epistemic_uncertainty, dim = c(observed$shape[1], self$output_dim)))
          colnames(epistemic_uncertainty) = colnames(self$theta)
          self$epistemic_uncertainty = sqrt(epistemic_uncertainty)

          aleatoric_uncertainty = as.data.frame(array(aleatoric_uncertainty, dim = c(observed$shape[1], self$output_dim)))
          colnames(aleatoric_uncertainty) = colnames(self$theta)
          self$aleatoric_uncertainty = sqrt(aleatoric_uncertainty)

          posterior_median = as.data.frame(array(posterior_median, dim = c(observed$shape[1], self$output_dim)))
          colnames(posterior_median) = colnames(self$theta)

          posterior_lower_ci = as.data.frame(array(posterior_lower_ci, dim = c(observed$shape[1], self$output_dim)))
          colnames(posterior_lower_ci) = colnames(self$theta)

          posterior_upper_ci = as.data.frame(array(posterior_upper_ci, dim = c(observed$shape[1], self$output_dim)))
          colnames(posterior_upper_ci) = colnames(self$theta)

          self$quantile_posterior = list(mean = predictive_mean,
                                         median = posterior_median,
                                         posterior_lower_ci = posterior_lower_ci,
                                         posterior_upper_ci = posterior_upper_ci)

          self$overall_uncertainty = sqrt(aleatoric_uncertainty) + sqrt(epistemic_uncertainty)

          # Store dropout rates inferred
          params = self$fitted$model$named_parameters()
          p_logit = params[grepl("p_logit", names(params))]
          p = lapply(p_logit, function(x) torch::torch_sigmoid(x))
          p = unlist(lapply(p, function(x) as.numeric(x)))
          self$dropout_rates = p
        }

        if (self$method == 'deep ensemble') {
          # Use forward to get mean prediction + variance

          # Infer epistemic + aleatoric uncertainty
          # output for ensemble network
          n_obs = nrow(self$observed_adj)
          out_mu_sample  = torch::torch_zeros(c(n_obs, self$output_dim, self$num_networks))
          out_sig_sample = torch::torch_zeros(c(n_obs, self$output_dim, self$num_networks))

          for (i in 1:self$num_networks) {
            pb = txtProgressBar(min = 1, max = self$num_networks, style = 3)

            # print(paste("Network", i))

            preds = self$fitted$model$model_list[[i]](observed)

            # print(preds)
            mu_sample = preds[1,,]
            # print(mu_sample)

            sig_sample = preds[2,,]
            # sig_sample = torch_logsumexp(sig_sample, 1, keepdim = TRUE) + 1e-6
            sig_sample = log1pexp(sig_sample) # Numerically stable approximation
            # print(sig_sample)

            out_mu_sample[,,i]  = mu_sample
            out_sig_sample[,,i] = sig_sample

            setTxtProgressBar(pb, i)
          }
          close(pb)

          # print("Compute predictive mean")

          # Mean prediction across networks
          out_mu_sample_final  = torch::torch_mean(out_mu_sample, dim = 3)
          predictive_mean = as.data.frame(array(as.numeric(out_mu_sample_final), dim = c(observed$shape[1], self$output_dim)))
          colnames(predictive_mean) = colnames(self$theta)

          out_sig_sample_final = torch::torch_sqrt(torch::torch_mean(out_sig_sample, dim = 3) +
                                                     torch::torch_mean(torch_square(out_mu_sample), dim = 3) -
                                                     torch::torch_square(out_mu_sample_final))

          out_sig_sample_aleatoric = torch::torch_sqrt(torch::torch_mean(out_sig_sample, dim = 3))
          out_sig_sample_epistemic = torch::torch_sqrt(torch::torch_mean(torch::torch_square(out_mu_sample), dim = 3) -
                                                         torch::torch_square(out_mu_sample_final))

          epistemic_uncertainty = as.data.frame(array(as.numeric(out_sig_sample_epistemic), dim = c(observed$shape[1], self$output_dim)))
          colnames(epistemic_uncertainty) = colnames(self$theta)

          aleatoric_uncertainty = as.data.frame(array(as.numeric(out_sig_sample_aleatoric), dim = c(observed$shape[1], self$output_dim)))
          colnames(aleatoric_uncertainty) = colnames(self$theta)

          # Mean prediction across networks
          self$predictive_mean  = predictive_mean

          self$aleatoric_uncertainty = aleatoric_uncertainty
          self$epistemic_uncertainty = epistemic_uncertainty

          self$overall_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        }

        # Conformal prediction
        if (self$num_conformal > 0) {
          self$conformal_prediction()
        }
      }

    },

    #' @description
    #' Prepare the torch dataloader from sumstat/theta (input/target)
    #'
    #' Build and return a dataloader object
    #'
    dataloader = function() {
      #-----------------------------------#
      # SAMPLING
      #-----------------------------------#
      # ABC sampling before the neural network
      # ABC sampling before scaling
      # if (!is.null(self$tol)) {
      #   abc = self$abc_sampling()
      #   theta = as.matrix(abc$theta)
      #   sumstat = as.matrix(abc$sumstat)
      # } else {
      #   theta = as.matrix(self$theta)
      #   sumstat = as.matrix(self$sumstat)
      # }
      theta = as.matrix(self$theta)
      sumstat = as.matrix(self$sumstat)

      n_total = nrow(sumstat)

      # Randomly sample indexes
      n_val = round(n_total * self$validation_split, digits=0)
      n_test = round(n_total * self$test_split, digits=0)
      n_train = n_total - n_val - n_test - self$num_conformal
      num_conformal = self$num_conformal

      self$n_train = n_train

      random_idx = sample(1:nrow(theta), replace = FALSE)
      train_idx = random_idx[1:n_train]
      valid_idx = random_idx[(n_train + 1):(n_train + n_val)]
      test_idx = random_idx[(n_train + n_val + 1):(n_train + n_val + n_test)]

      #-----------------------------------#
      # PRE-PROCESSING DATA
      #-----------------------------------#
      # Scale summary statistics (optional)
      # Scale on training set just before training (hence scaling will be adjusted to each training pass)
      # De-scale only in self$predictions() method (all computations are done in the scaled space)
      # scaler = preProcess(train, method = "range")
      # predict(scaler, train)
      # predict(scaler, test)

      input_min = apply(self$sumstat[train_idx,,drop = F], 2, function(x) min(x, na.rm = TRUE))
      input_max = apply(self$sumstat[train_idx,,drop = F], 2, function(x) max(x, na.rm = TRUE))
      input_mean = apply(self$sumstat[train_idx,,drop = F], 2, function(x) mean(x, na.rm = TRUE))
      input_sd = apply(self$sumstat[train_idx,,drop = F], 2, function(x) sd(x, na.rm = TRUE))
      quantile_25 = apply(self$sumstat[train_idx,,drop = F], 2, function(x) quantile(x, 0.25, na.rm = TRUE))
      quantile_75 = apply(self$sumstat[train_idx,,drop = F], 2, function(x) quantile(x, 0.75, na.rm = TRUE))

      self$input_summary = list(min = input_min,
                                max = input_max,
                                mean = input_mean,
                                sd = input_sd,
                                quantile_25 = quantile_25,
                                quantile_75 = quantile_75)

      target_min = apply(self$theta[train_idx,,drop = F], 2, function(x) min(x, na.rm = TRUE))
      target_max = apply(self$theta[train_idx,,drop = F], 2, function(x) max(x, na.rm = TRUE))
      target_mean = apply(self$theta[train_idx,,drop = F], 2, function(x) mean(x, na.rm = TRUE))
      target_sd = apply(self$theta[train_idx,,drop = F], 2, function(x) sd(x, na.rm = TRUE))
      quantile_25 = apply(self$theta[train_idx,,drop = F], 2, function(x) quantile(x, 0.25, na.rm = TRUE))
      quantile_75 = apply(self$theta[train_idx,,drop = F], 2, function(x) quantile(x, 0.75, na.rm = TRUE))

      self$target_summary = list(min = target_min,
                                 max = target_max,
                                 mean = target_mean,
                                 sd = target_sd,
                                 quantile_25 = quantile_25,
                                 quantile_75 = quantile_75)

      # Scale with summary statistics learned on training set only
      scaled_input = scaler(self$sumstat,
                            self$input_summary,
                            method = self$scale_input,
                            type = "forward")

      scaled_observed = scaler(self$observed,
                            self$input_summary,
                            method = self$scale_input,
                            type = "forward")


      scaled_target = scaler(self$theta,
                            self$target_summary,
                            method = self$scale_target,
                            type = "forward")

      self$sumstat_adj = scaled_input
      self$observed_adj = scaled_observed
      self$theta_adj = scaled_target

      if (num_conformal > 0) {
        conformal_idx = random_idx[(n_train + n_val + n_test + 1):(n_train + n_val + n_test + num_conformal)]
        self$calibration_theta = self$theta_adj[conformal_idx,,drop=F]
        self$calibration_sumstat = self$sumstat_adj[conformal_idx,,drop=F]
      }

      #-----------------------------------#
      # MAKE TENSOR DATASET
      #-----------------------------------#
      sumstat_tensor = torch::torch_tensor(as.matrix(self$sumstat_adj), dtype = torch::torch_float())
      theta_tensor = torch::torch_tensor(as.matrix(self$theta_adj), dtype = torch::torch_float())

      ds = torch::tensor_dataset(sumstat_tensor, theta_tensor)

      valid_ds = torch::dataset_subset(ds, valid_idx)
      valid_dl = torch::dataloader(valid_ds, batch_size = self$batch_size, shuffle = TRUE, drop_last = TRUE)

      test_ds = torch::dataset_subset(ds, test_idx)
      test_dl = torch::dataloader(test_ds, batch_size = self$batch_size, shuffle = TRUE, drop_last = TRUE)

      if (self$method == 'tabnet-abc') {
        return(list(sumstat_adj = scaled_input, theta_adj = scaled_target))
      }

      if (self$method == 'monte carlo dropout' | self$method == 'concrete dropout') {
        # Data loader (MC dropout and Concrete dropout)
        train_ds = torch::dataset_subset(ds, train_idx)
        train_dl = torch::dataloader(train_ds, batch_size = self$batch_size, shuffle = TRUE, drop_last = TRUE)

        return(list(train = train_dl, valid = valid_dl, test = test_dl))
      }

      if (self$method == 'deep ensemble') {
        # Make Ensemble dataset
        train_x = sumstat_tensor[train_idx,]
        train_y = theta_tensor[train_idx,]
        train_ds_list = ensemble_dataset(train_x, train_y, self$num_networks, randomize = TRUE)
        train_dl_list = train_ds_list %>% torch::dataloader(batch_size = self$batch_size, shuffle = TRUE, drop_last = TRUE)

        return(list(train = train_dl_list, valid = valid_dl, test = test_dl))
      }

    },

    # ABC sampling of the simulations closest to the observed summary statistics
    # abc_sampling = function() {
    #   if (self$verbose) {cat("ABC sampling with kernel", self$kernel, "and tolerance", self$tol,"\n")}
    #   # Apply kernel weighting to subset the simulations closest to observed
    #   # Select the theta values that best match the observed data
    #   # Improved Kernel sampling
    #   x1 = self$sumstat
    #   x2 = self$observed
    #
    #   kernel_values = density_kernel(x1,
    #                                  x2,
    #                                  kernel = self$kernel,
    #                                  bandwidth = self$bandwidth,
    #                                  length_scale = self$length_scale,
    #                                  ncores = self$ncores)
    #
    #   # Normalize the kernel values to form a proper weighting
    #   self$kernel_values = kernel_values/sum(kernel_values)
    #   # Sampling
    #   # Select sumstats and priors (parameters) in the given sampled region
    #   idx = abc_sampling(self$kernel_values,
    #                      method = self$sampling,
    #                      theta = self$theta,
    #                      tol = self$tol)
    #
    #   theta = as.matrix(self$theta[idx,])
    #   sumstat = as.matrix(self$sumstat[idx,])
    #
    #   return(list(theta = theta, sumstat = sumstat))
    # },

    #' @description
    #' Estimate a calibrated credible interval with Conformal Prediction
    #'
    conformal_prediction = function() {
      if (self$verbose) {cat("Performing conformal prediction\n\n")}

      # see https://forgemia.inra.fr/mistea/codes_articles/abcdconformal/-/blob/main/R/GaussianFields/Comparaison_ABCD_conv2d_conformal.Rmd?ref_type=heads
      # Monte Carlo Dropout prediction on the calibration set
      # Copy Conformal dataset
      # which has already been adjusted/scaled, same as training and validation sets
      calibration_set = self$calibration_sumstat
      calibration_truth = self$calibration_theta
      n_cal = nrow(calibration_set)

      # Copy the `abcnn` object and make predictions on the calibration set
      abcnn_conformal = self$clone(deep = TRUE)
      abcnn_conformal$num_conformal = 0 # avoid recursivity
      # Replace observed by conformal calibration set
      abcnn_conformal$observed_adj = calibration_set

      # Computation of the conformal quantile on the calibration set
      abcnn_conformal$predict()

      # Compute the calibration score sj using the score function
      # see https://forgemia.inra.fr/mistea/codes_articles/abcdconformal/-/blob/main/R/LoktaVolterra/Lokta_Volterra_last-one.Rmd?ref_type=heads
      # j = seq(1, n_cal)

      # a) Epistemic uncertainty
      scores_epistemic = matrix(nrow = nrow(calibration_truth), ncol = ncol(calibration_truth))

      for (i in 1:n_cal) {
        true = as.matrix(calibration_truth[i,,drop=F])
        pred = as.matrix(abcnn_conformal$predictive_mean[i,,drop=F])
        uncertainty = as.matrix(abcnn_conformal$epistemic_uncertainty[i,,drop=F])
        # uncertainty = uncertainty^2 # Uncertainty is already sqrt transformed
        scores_epistemic[i,] = abs(true - pred) / uncertainty # Simpler calculation, one score per sample and output dim
        # scores_epistemic[i,] = sqrt((true - pred) * uncertainty^(-1) * (true - pred)) # formula (9) of the paper
        # scores_epistemic[i,] = sqrt(t(true - pred) %*% solve(uncertainty) %*% (true-pred))  # formula (9) of the paper
      }

      # b) Overall uncertainty
      scores_overall = matrix(nrow = nrow(calibration_truth), ncol = ncol(calibration_truth))

      for (i in 1:n_cal) {
        true = as.matrix(calibration_truth[i,,drop=F])
        pred = as.matrix(abcnn_conformal$predictive_mean[i,,drop=F])
        uncertainty = as.matrix(abcnn_conformal$overall_uncertainty[i,,drop=F])
        # uncertainty = uncertainty^2 # Uncertainty is already sqrt transformed
        scores_overall[i,] = abs(true - pred)/uncertainty # Simpler calculation, one score per sample and output dim
        # scores_overall[i,] = sqrt((true - pred) * uncertainty^(-1) * (true - pred)) # formula (9) of the paper
        # scores_overall[i,] = sqrt(t(true - pred) %*% solve(uncertainty) %*% (true-pred))  # formula (9) of the paper
      }

      # Compute the conformal quantile
      alpha = 1 - self$credible_interval_p

      q_level = ceiling((n_cal + 1)*(1 - alpha))/n_cal
      qhat = sort(scores_epistemic)[q_level*n_cal]
      # # It is the same as above
      quantile(scores_epistemic, ((n_cal + 1)*(1 - alpha))/n_cal, na.rm = TRUE)

      # For the new data sample x, approximation of Eπ[θ | x] and confidence set for θ :
      # apply(scores_epistemic, 2, function(x) sort(x[q_level * n_cal]))

      epistemic_conformal_quantile = apply(scores_epistemic, 2, function(x) quantile(x, ((n_cal + 1)*(1 - alpha))/n_cal, na.rm = TRUE))
      epistemic_conformal_quantile = as.data.frame(t(epistemic_conformal_quantile))
      colnames(epistemic_conformal_quantile) = abcnn_conformal$theta_names

      overall_conformal_quantile = apply(scores_overall, 2, function(x) quantile(x, ((n_cal + 1)*(1 - alpha))/n_cal, na.rm = TRUE))
      overall_conformal_quantile = as.data.frame(t(overall_conformal_quantile))
      colnames(overall_conformal_quantile) = abcnn_conformal$theta_names

      self$epistemic_conformal_quantile = epistemic_conformal_quantile
      self$overall_conformal_quantile = overall_conformal_quantile

      # Clean up deep copies
      rm(abcnn_conformal)
    },

    #' @description
    #' Returns a tidy tibble with predictions and credible intervals
    #'
    predictions = function() {
      # Back-transform predictions to original scale
      if (self$verbose) {cat("Back-transform scaled parameters with method:", self$scale_target,"\n")}

      predictive_mean = scaler(self$predictive_mean,
                                           self$target_summary,
                                           method = self$scale_target,
                                           type = "backward")
      epistemic_uncertainty = scaler(self$epistemic_uncertainty,
                                                 self$target_summary,
                                                 method = self$scale_target,
                                                 type = "backward")
      aleatoric_uncertainty = scaler(self$aleatoric_uncertainty,
                                                 self$target_summary,
                                                 method = self$scale_target,
                                                 type = "backward")
      overall_uncertainty = scaler(self$overall_uncertainty,
                                               self$target_summary,
                                               method = self$scale_target,
                                               type = "backward")

      if (self$method == "monte carlo dropout" | self$method == "concrete dropout" | self$method == "tabnet-abc") {
        posterior_median = scaler(self$quantile_posterior$median,
                                    self$target_summary,
                                    method = self$scale_target,
                                    type = "backward")
        posterior_lower_ci = scaler(self$quantile_posterior$posterior_lower_ci,
                                    self$target_summary,
                                    method = self$scale_target,
                                    type = "backward")
        posterior_upper_ci = scaler(self$quantile_posterior$posterior_upper_ci,
                                    self$target_summary,
                                    method = self$scale_target,
                                    type = "backward")
      }

      # Conformal predictions
      # quantile * sqrt(variance heuristic)
      df_epistemic = epistemic_uncertainty
      for (j in 1:ncol(df_epistemic)) {
        df_epistemic[,j] = self$epistemic_conformal_quantile[,j] * self$epistemic_uncertainty[,j]
      }

      df_overall = overall_uncertainty
      for (j in 1:ncol(df_overall)) {
        df_overall[,j] = self$overall_conformal_quantile[,j] * self$overall_uncertainty[,j]
      }

      # Scale back
      df_overall = scaler(df_overall,
                          self$target_summary,
                          method = self$scale_target,
                          type = "backward")

      df_epistemic = scaler(df_epistemic,
                            self$target_summary,
                            method = self$scale_target,
                            type = "backward")


      # Tidy data
      pred_mean = tidyr::gather(predictive_mean,
                         key = "variable")

      aleatoric_uncertainty = tidyr::gather(aleatoric_uncertainty,
                         key = "variable")

      epistemic_uncertainty = tidyr::gather(epistemic_uncertainty,
                         key = "variable")

      overall_uncertainty = tidyr::gather(overall_uncertainty,
                                     key = "variable")

      epistemic_conformal = tidyr::gather(df_epistemic,
                                          key = "variable")

      overall_conformal = tidyr::gather(df_overall,
                                          key = "variable")

      sample = data.frame(sample = rep(seq(1, self$n_obs), self$output_dim))

      predictions = cbind(sample,
                          pred_mean,
                          epistemic_uncertainty[,2],
                          aleatoric_uncertainty[,2],
                          overall_uncertainty[,2],
                          epistemic_conformal[,2],
                          overall_conformal[,2])

      colnames(predictions) = c("sample", "parameter",
                                "predictive_mean",
                                "epistemic_uncertainty",
                                "aleatoric_uncertainty",
                                "overall_uncertainty",
                                "epistemic_conformal_credible_interval",
                                "overall_conformal_credible_interval")

      if (self$method == "monte carlo dropout" | self$method == "concrete dropout" | self$method == "tabnet-abc") {
        predictions$posterior_median = tidyr::gather(posterior_median,
                                key = "variable")[,2]
        predictions$posterior_lower_ci = tidyr::gather(posterior_lower_ci,
                                                key = "variable")[,2]
        predictions$posterior_upper_ci = tidyr::gather(posterior_upper_ci,
                                                    key = "variable")[,2]
      } else {
        predictions$posterior_median = NA
        predictions$posterior_lower_ci = NA
        predictions$posterior_upper_ci = NA
      }

      return(predictions)
    },

    #' @description
    #' Print a summary of the `abcnn` object
    #'
    summary = function() {
      # TODO Print number of samples and basic information on methods
      cat("ABC parameter inference with the method:", self$method, "\n")

      cat("The number of samples is", nrow(self$theta), "for the training set (simulations) and", nrow(self$observed), "observations for predictions.\n")
      cat("The validation split during training is", self$validation_split, ", the test split after training is", self$test_split, ", and", self$num_conformal, "simulations retained for conformal prediction.\n")
      # CUDA is installed?
      cat("\n")
      cat("Is CUDA available? ")
      print(torch::cuda_is_available())
      cat("\n")

      cat("Device is: ")
      print(self$device)
      cat("\n")
    },

    #' @description
    #' Plot the training curves (training/validation)
    #'
    #' @param discard_first Discard the first epoch, as it may have a large loss compared to next ones (for plotting only)
    #'
    plot_training = function(discard_first = FALSE) {
      if (is.null(self$fitted)) {
        warning("The model has not been fitted.")
      } else {

        if (self$method == "tabnet-abc") {
          autoplot(self$fitted)
        } else {
          if (self$method != "deep ensemble") {
            train_metric = as.numeric(unlist(self$fitted$records$metrics$train))
            valid_metric = as.numeric(unlist(self$fitted$records$metrics$valid))
            eval = ifelse(self$method == "monte carlo dropout" | self$method == "concrete dropout",
                          self$eval_metrics$value,
                          NA)

            if (discard_first) {
              train_metric[1] = NA
            }

            train_eval = data.frame(Epoch = rep(1:length(train_metric), 2),
                                    Metric = c(train_metric, valid_metric),
                                    Mode = c(rep("train", length(train_metric)), rep("validation", length(valid_metric))))

            ggplot2::ggplot(train_eval, aes(x = Epoch, y = Metric, color = Mode, fill = Mode)) +
              geom_point() +
              geom_line() +
              xlab("Epoch") + ylab("Loss") +
              geom_hline(yintercept = eval) +
              theme_bw()
          } else {
            train_metric = as.numeric(unlist(self$fitted$records$metrics$train))
            valid_metric = as.numeric(unlist(self$fitted$records$metrics$valid))

            if (discard_first) {
              train_metric[1] = NA
            }

            train_eval = data.frame(Epoch = rep(rep(1:(length(train_metric)/self$num_networks), each = self$num_networks), self$output_dim),
                                    Model = rep(1:self$num_networks, (length(train_metric)/self$num_networks)),
                                    Metric = c(train_metric, valid_metric),
                                    Mode = c(rep("train", length(train_metric)), rep("validation", length(valid_metric))))

            ggplot2::ggplot(train_eval, aes(x = Epoch, y = Metric, color = Mode, fill = Mode)) +
              geom_point() +
              geom_line() +
              xlab("Epoch") + ylab("Loss") +
              facet_wrap(~ Model) +
              theme_bw()
          }
        }

      }


    },

    #' @description
    #' Plot predicted values and their credible intervals
    #'
    #' @param uncertainty_type The type of uncertainty to plot, whether `conformal` credible intervals (default),
    #' the `uncertainty` estimated (square root of the variance) or the `posterior quantile`, that are credible intervals
    #' computed on the distribution of posteriors.
    #' @param plot_type The type of plot, whether a `line` or `errorbar` around points
    #'
    plot_prediction = function(uncertainty_type = "conformal",
                              plot_type = "line") {

      pal = RColorBrewer::brewer.pal(8, "Dark2")
      cols = c("Epistemic" = pal[3],"Overall" = pal[2])

      if (is.null(self$fitted)) {
        warning("The model has not been fitted.")
      } else {
        # If same number of parameters x and y, infer pairwise relationship between each x and y (i.e. x1 ~ y1, x2 ~ y2...)
        # Otherwise order x axis by index
        df_predicted = self$predictions()

        # Paired when the input and output dim are the same
        # hence output dim can be plotted as a function of input
        paired = (self$input_dim == self$output_dim)
        if (paired) {
          x_pos = as.numeric(unlist(self$observed))
        } else {
          x_pos = df_predicted$sample
        }

        if (uncertainty_type == "uncertainty") {
          df_predicted$ci_overall_upper = df_predicted$predictive_mean + df_predicted$overall_uncertainty
          df_predicted$ci_overall_lower = df_predicted$predictive_mean - df_predicted$overall_uncertainty

          df_predicted$ci_e_upper = df_predicted$predictive_mean + df_predicted$epistemic_uncertainty
          df_predicted$ci_e_lower = df_predicted$predictive_mean - df_predicted$epistemic_uncertainty

          df_predicted$mean = df_predicted$predictive_mean
        }

        if (uncertainty_type == "conformal") {
          df_predicted$ci_overall_upper = df_predicted$predictive_mean + df_predicted$overall_conformal_credible_interval
          df_predicted$ci_overall_lower = df_predicted$predictive_mean - df_predicted$overall_conformal_credible_interval

          df_predicted$ci_e_upper = df_predicted$predictive_mean + df_predicted$epistemic_conformal_credible_interval
          df_predicted$ci_e_lower = df_predicted$predictive_mean - df_predicted$epistemic_conformal_credible_interval

          df_predicted$mean = df_predicted$predictive_mean
        }

        if (uncertainty_type == "posterior quantile") {
          df_predicted$ci_overall_upper = df_predicted$posterior_upper_ci
          df_predicted$ci_overall_lower = df_predicted$posterior_lower_ci

          df_predicted$ci_e_upper = as.numeric(NA)
          df_predicted$ci_e_lower = as.numeric(NA)

          df_predicted$mean = df_predicted$posterior_median
        }

        df_predicted$x = x_pos

        if (plot_type == "line") {
          ggplot2::ggplot(data = df_predicted, aes(x = x)) +
            geom_line(aes(x = x, y = mean), color = "black") +
            facet_wrap(~ parameter, scales = "free") +
            geom_ribbon(aes(x = x, ymin = ci_overall_lower, ymax = ci_overall_upper, fill = "Overall"), alpha = 0.3) +
            geom_ribbon(aes(x = x, ymin = ci_e_lower, ymax = ci_e_upper, fill = "Epistemic"), alpha = 0.3) +
            xlab("Observed") + ylab("Predicted") +
            scale_fill_manual(name = "Uncertainty", values = cols) +
            theme_bw()
        } else {
          if (plot_type == "errorbar") {
            ggplot2::ggplot(data = df_predicted, aes(x = x)) +
              facet_wrap(~ parameter, scales = "free") +
              geom_errorbar(aes(x = x, ymin = ci_overall_lower, ymax = ci_overall_upper, colour = "Overall"), alpha = 0.5) +
              geom_errorbar(aes(x = x, ymin = ci_e_lower, ymax = ci_e_upper, colour = "Epistemic"), alpha = 0.5) +
              geom_point(aes(x = x, y = mean), color = "black") +
              xlab("Observed") + ylab("Predicted") +
              scale_colour_manual(name = "Uncertainty", values = cols) +
              theme_bw()
          }
        }

      }

    },

    #' @description
    #' Plot the distributions of estimates and predictions
    #'
    #' @param sample Index of the sample to plot
    #' @param prior logical, whether to plot the prior underneath the posterior and prediction
    #' @param uncertainty_type The type of uncertainty to plot, whether `conformal` credible intervals (default),
    #' the `uncertainty` estimated (square root of the variance) or the `posterior quantile`, that are credible intervals
    #' computed on the distribution of posteriors.
    #'
    plot_posterior = function(sample = 1,
                              prior = TRUE,
                              uncertainty_type = "conformal") {
      # Dim 1 is number of MC samples (predictions)
      # Dim 2 is number of observations
      # Dim 3 is parameters (mu + sigma)

      if (is.null(self$fitted)) {
        warning("The model has not been fitted.")
      } else {
        tidy_predictions = self$predictions()
        pal = RColorBrewer::brewer.pal(8, "Dark2")
        cols = c("Epistemic" = pal[3],"Overall" = pal[2])

        if (uncertainty_type == "uncertainty") {
          tidy_predictions$ci_upper = tidy_predictions$predictive_mean + tidy_predictions$overall_uncertainty
          tidy_predictions$ci_lower = tidy_predictions$predictive_mean - tidy_predictions$overall_uncertainty

          tidy_predictions$ci_e_upper = tidy_predictions$predictive_mean + tidy_predictions$epistemic_uncertainty
          tidy_predictions$ci_e_lower = tidy_predictions$predictive_mean - tidy_predictions$epistemic_uncertainty
        }

        if (uncertainty_type == "conformal") {
          tidy_predictions$ci_upper = tidy_predictions$predictive_mean + tidy_predictions$overall_conformal_credible_interval
          tidy_predictions$ci_lower = tidy_predictions$predictive_mean - tidy_predictions$overall_conformal_credible_interval

          tidy_predictions$ci_e_upper = tidy_predictions$predictive_mean + tidy_predictions$epistemic_conformal_credible_interval
          tidy_predictions$ci_e_lower = tidy_predictions$predictive_mean - tidy_predictions$epistemic_conformal_credible_interval
        }

        if (uncertainty_type == "posterior quantile") {
          tidy_predictions$ci_upper = tidy_predictions$posterior_upper_ci
          tidy_predictions$ci_lower = tidy_predictions$posterior_lower_ci

          tidy_predictions$ci_e_upper = as.numeric(NA)
          tidy_predictions$ci_e_lower = as.numeric(NA)
        }

        tidy_predictions = tidy_predictions[tidy_predictions$sample == sample,]

        if (prior) {
          tidy_priors = data.frame(param = rep(colnames(self$theta), each = nrow(self$theta)),
                                   prior = as.numeric(unlist(self$theta)))
          colnames(tidy_priors)[1] = "parameter"

          p = ggplot() +
            ggplot2::geom_histogram(data = tidy_priors, aes(x = prior), color = "darkgrey", fill = "grey", alpha = 0.1) +
            facet_wrap(~ parameter)
        } else {
          p = ggplot2::ggplot()
        }

        if (self$method %in% c("tabnet-abc", "monte carlo dropout", "concrete dropout")) {
          posteriors = self$posterior_samples[,sample,]
          posteriors = as.data.frame(posteriors)
          posteriors = posteriors[,1:self$output_dim, drop = FALSE]
          colnames(posteriors) = self$theta_names

          # Back-transform predictions to original scale
          cat("Back-transform scaled posteriors with method:", self$scale_target,"\n")
          posteriors = scaler(posteriors,
                              self$target_summary,
                              method = self$scale_target,
                              type = "backward")
          posteriors$mc_sample = as.character(c(1:nrow(posteriors)))

          # output_names = unlist(lapply(c("mu", "sigma"), function(x) paste(colnames(theta), x, sep = "_")))

          tidy_df = posteriors %>% tidyr::gather(param, prediction, any_of(colnames(self$theta)))
          colnames(tidy_df)[2] = "parameter"

          p = p + ggplot2::geom_histogram(data = tidy_df, aes(x = prediction)) +
            facet_wrap(~ parameter)
        }

        p = p +
          geom_vline(data = tidy_predictions, aes(xintercept = predictive_mean, colour = "black")) +
          geom_rect(data = tidy_predictions, aes(xmin = ci_e_lower, xmax = ci_e_upper, ymin = -Inf, ymax = Inf, colour = "Epistemic", fill = "Epistemic"), alpha = 0.1) +
          geom_vline(data = tidy_predictions, aes(xintercept = ci_e_lower, colour = "Epistemic")) +
          geom_vline(data = tidy_predictions, aes(xintercept = ci_e_upper, colour = "Epistemic")) +
          geom_vline(data = tidy_predictions, aes(xintercept = ci_lower, colour = "Overall")) +
          geom_vline(data = tidy_predictions, aes(xintercept = ci_upper, colour = "Overall")) +
          # geom_vline(data = tidy_predictions, aes(xintercept = ci_conformal_lower, colour = "Overall conformal")) +
          # geom_vline(data = tidy_predictions, aes(xintercept = ci_conformal_upper, colour = "Overall conformal")) +
          # geom_vline(data = tidy_predictions, aes(xintercept = ci_conformal_e_lower, colour = "Epistemic conformal")) +
          # geom_vline(data = tidy_predictions, aes(xintercept = ci_conformal_e_upper, colour = "Epistemic conformal")) +
          geom_rect(data = tidy_predictions, aes(xmin = ci_lower, xmax = ci_upper, ymin = -Inf, ymax = Inf, colour = "Overall", fill = "Overall"), alpha = 0.1) +
          # geom_rect(data = tidy_predictions, aes(xmin = ci_conformal_lower, xmax = ci_conformal_upper, ymin = -Inf, ymax = Inf, colour = "Overall conformal", fill = "Overall conformal"), alpha = 0.1) +
          # geom_rect(data = tidy_predictions, aes(xmin = ci_conformal_e_lower, xmax = ci_conformal_e_upper, ymin = -Inf, ymax = Inf, colour = "Epistemic conformal", fill = "Epistemic conformal"), alpha = 0.1) +
          facet_wrap(~ parameter, scales = "free") +
          scale_colour_manual(name = "Uncertainty", values = cols) +
          scale_fill_manual(name = "Uncertainty", values = cols) +
          xlab("Value") + ylab("Count") +
          ggtitle(uncertainty_type) +
          theme_bw() +
          theme(legend.position = "right")

        p
      }

    }
  )
)





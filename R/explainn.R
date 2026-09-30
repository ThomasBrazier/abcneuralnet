#' An `explainn` object for feature attribution
#' A R6 class object
#'
#' @description
#'
#' This module function allows to apply a diverse set of Feature attribtion methods on a fitted `abcnn` neural network
#' and a given observed dataset, in order to compute the weight of each summary statistic on predictions.
#' Summary statistics with the higher weight (or importance) are those contributing the most to the prediction.
#'
#' This feature importance method is useful to perform feature selection (removing summary statistics that don't explain well the output)
#' and to interpret properly the output of the model.
#'
#'
#' @param abcnn An `abcnn` object
#' @param method A feature attribution method, as named in the `ìnnsight` R package
#' including 'cw' (default), 'grad', 'smoothgrad', 'intgrad', 'expgrad', 'lrp', 'deeplift',
#' 'deepshap', 'shap', 'lime.' No method required for tabnet-ABC.
#' @param ensemble_num_model index of the model when the network is a deep ensemble (default = 1).
#' Attributions are computed for this single ensemble member only; they are not averaged across
#' ensemble members. To assess consistency across the ensemble, create separate `explainn`
#' objects with different `ensemble_num_model` values and compare results.
#'
#'
#' @details
#'
#' All the methods used in explain are implemented in the `innsight` R package, that is part of the R torch ecosystem. These methods are:
#'
#' - Vanilla Gradient and GradientxInput
#' - SmoothGrand and SmoothGradxInput
#' - Integrated Gradients
#' - Expected Gradients
#' - Layer-Wise Relevance Propagation (LRP)
#' - Deep learning important features (DeepLift)
#' - Deep Shapley additive explanations (DeepSHAP)
#' - Connection weights method
#' - Local interpretable model-agnostic explanations (LIME)
#' - Shapley values (SHAP)
#'
#' See `https://bips-hb.github.io/innsight/` for details.
#'
#' For `monte carlo dropout`, `gaussian monte carlo dropout`, `concrete dropout` and
#' `deep ensemble`, attributions explain the mean (mu) prediction head only. The aleatoric
#' log-variance head is never explained: a summary statistic with a low importance score for
#' the mean can still be the one driving a high aleatoric uncertainty for that sample, and this
#' method will not surface that.
#'
#' The network is trained on scaled targets (see `scale_target` in `abcnn`), so raw attributions
#' are expressed in units of the scaled target, which differ between parameters and between scaling
#' methods. By default, `run()` rescales them to a common scale, the fraction of each parameter's
#' prior range, so that they are comparable across parameters. See the `output_scale` argument of `run()`.
#'
#'
#' @slot converter Stores the `innsight::converter` object
#'
#' @slot result stores results of the `explainn$run()` method.
#' @slot model_method method of the trained neural network (e.g. "concrete dropout")
#' @slot variables names of the variables (summary statistics)
#' @slot parameters names of the parameter to infer
#' @slot ensemble_num_model index of the (single) ensemble member explained when the network is a deep ensemble; results are not averaged across members
#' @slot scale_input the `abcnn$scale_input` slot from the `abcnn` input object
#' @slot input_summary the `abcnn$input_summary` slot from the `abcnn` input object
#'
#' @import torch
#' @import luz
#' @import ggplot2
#' @import innsight
#' @import R6
#' @import RColorBrewer
#' @import janitor
#'
#' @importFrom Rdpack reprompt
#'
#' @return an `explainn` object
#' @export
#'
explainn = R6::R6Class("explainn",
                    public = list(
                      #' @field x an `abcnn` object
                      x = NULL,
                      #' @field method the `innsight` method to apply: `grad`, `cw`, `smoothgrad`, `intgrad`, `expgrad`, `lrp`, `deeplift`, `deepshap`, `shap`, `lime`
                      method = NULL,
                      #' @field converter the torch/luz model converted to an `innsight` object
                      converter = NULL,
                      #' @field result the result of the explainability method
                      result = NULL,
                      #' @field model_method the method in the `abcnn` object
                      model_method = NULL,
                      #' @field variables names of the variables (input dimensions)
                      variables = NULL,
                      #' @field parameters names of the parameters to estimate (output dimensions)
                      parameters = NULL,
                      #' @field ensemble_num_model index of the single ensemble member explained in Deep Ensemble (default is first model); results are not averaged across members
                      ensemble_num_model = NULL,
                      #' @field scale_input method used to scale input dimensions
                      scale_input = NULL,
                      #' @field input_summary summary statistics for the input scaling method
                      input_summary = NULL,
                      #' @field scale_target method used to scale the targets (parameters) in the `abcnn` object
                      scale_target = NULL,
                      #' @field target_summary summary statistics for the target scaling method (see `abcnn$target_summary`)
                      target_summary = NULL,
                      #' @field target_center median of the scaled training targets, the point at which `cw` attributions are rescaled under the non-linear `log` and `logit` target scalings
                      target_center = NULL,
                      #' @field model the explained network (mu head only) as a `torch` module on cpu, used to compute the scaled predictions needed to rescale attributions
                      model = NULL,
                      #' @field output_scale the scale of the attributions in `result` (`prior_range`, `relative` or `none`), see `run()`
                      output_scale = NULL,

                      #' @description
                      #' Create a new `explainn` object
                      #'
                      #' @param x an `abcnn` model
                      #' @param method the explainability method to use (see `innsight` for details) (defauls is `cw`)
                      #' @param ensemble_num_model index of the single ensemble member to explain in Deep Ensemble (default is first model); results are not averaged across members
                      #'
                      initialize = function(x,
                                            method = "cw",
                                            ensemble_num_model = 1) {
                        self$method = method
                        if (x$method == "tabnet-abc") {self$x = x}
                        self$model_method = x$method
                        self$variables = colnames(janitor::clean_names(x$sumstat))
                        self$parameters = colnames(janitor::clean_names(x$theta))
                        self$ensemble_num_model = ensemble_num_model
                        self$scale_input = x$scale_input
                        self$input_summary = x$input_summary
                        self$scale_target = x$scale_target
                        self$target_summary = x$target_summary
                        if (is.data.frame(x$theta_adj) || is.matrix(x$theta_adj)) {
                          self$target_center = apply(as.matrix(x$theta_adj), 2, function(z) median(z, na.rm = TRUE))
                        }

                        # Tabnet-ABC has its own set of methods
                        if (self$model_method == "tabnet-abc") {
                          print("Note that Tabnet-ABC has its own set of explainability methods.")
                        } else {
                          # Convert the abccn object passed as input
                          if (self$model_method == "monte carlo dropout") {
                            model = x$fitted$model
                            model_mc = model$mc_dropout

                            # `model_mc` interleaves linear / mc_dropout / leaky_relu modules
                            # (see `R/mc_dropout.R`). Dropout is deliberately excluded here, as
                            # is standard for deterministic attribution, but the activation
                            # after each linear layer must be kept, or the surrogate collapses
                            # to a single affine map.
                            model_sequential = torch::nn_sequential(model_mc[[1]], model_mc[[3]])

                            for (i in seq_len(x$num_hidden_layers - 1) + 1) {
                              linear_mod = model_mc[[(i - 1) * 3 + 1]]
                              relu_mod = model_mc[[(i - 1) * 3 + 3]]
                              model_sequential$add_module(name = paste0("linear_", i), module = linear_mod)
                              model_sequential$add_module(name = paste0("relu_", i), module = relu_mod)
                            }

                            mod = x$fitted$model$mc_dropout$output
                            model_sequential$add_module(name = "output_mu", module = mod)
                          }

                          if (self$model_method == "gaussian monte carlo dropout") {
                            # FOR A CONCRETE MODEL
                            model = x$fitted$model
                            # model$modules
                            model_gaussian_mc = model$modules[[1]]
                            # model_concrete
                            model_gaussian_mc = model_gaussian_mc$gaussian_mc_dropout
                            # model_concrete

                            # `model_gaussian_mc` interleaves linear / mc_dropout / leaky_relu
                            # modules, named "0"/"1"/"2" for the first block and
                            # "linear_i"/"dropout_i"/"relu_i" for subsequent ones (see
                            # `R/gaussian_mc_dropout.R`). Dropout is deliberately excluded here,
                            # as is standard for deterministic attribution, but the activation
                            # after each linear layer must be kept, or the surrogate collapses
                            # to a single affine map.
                            first_linear = model_gaussian_mc["0"][[1]]
                            first_relu = model_gaussian_mc["2"][[1]]
                            model_sequential = torch::nn_sequential(first_linear, first_relu)

                            for (i in seq_len(x$num_hidden_layers - 1) + 1) {
                              linear_mod = model_gaussian_mc[paste0("linear_", i)][[1]]
                              relu_mod = model_gaussian_mc[paste0("relu_", i)][[1]]
                              model_sequential$add_module(name = paste0("linear_", i), module = linear_mod)
                              model_sequential$add_module(name = paste0("relu_", i), module = relu_mod)
                            }

                            mod = x$fitted$model$linear_mu
                            model_sequential$add_module(name = "output_mu", module = mod)

                          }

                          if (self$model_method == "concrete dropout") {
                            # FOR A CONCRETE MODEL
                            model = x$fitted$model
                            # model$modules
                            model_concrete = model$modules[[1]]
                            # model_concrete
                            model_concrete = model_concrete$concrete_dropout
                            # model_concrete

                            # Each `nn_concrete_linear` applies concrete dropout, then its own
                            # linear layer, then a leaky ReLU (see `R/concrete_dropout.R`).
                            # Dropout is deliberately excluded here, as is standard for
                            # deterministic attribution, but the activation must be kept, or the
                            # surrogate collapses to a single affine map.
                            model_sequential = torch::nn_sequential(model_concrete[[1]]$linear, model_concrete[[1]]$relu)

                            for (i in seq_len(x$num_hidden_layers - 1) + 1) {
                              model_sequential$add_module(name = paste0("linear_", i), module = model_concrete[[i]]$linear)
                              model_sequential$add_module(name = paste0("relu_", i), module = model_concrete[[i]]$relu)
                            }

                            mod = x$fitted$model$linear_mu
                            model_sequential$add_module(name = "output_mu", module = mod)

                          }

                          if (self$model_method == "deep ensemble") {
                            # FOR A DEEP ENSEMBLE MODEL
                            model = x$fitted$model
                            # Extract one model. Attributions are computed for this single
                            # ensemble member only, not averaged across the ensemble.
                            single_model = model$model_list[[self$ensemble_num_model]]
                            # Clone the container before appending the mu head below:
                            # `single_model$mlp` is the SAME live module used by
                            # `single_model$forward()`. Calling `add_module()` directly on it
                            # would permanently splice the mu head into the trained network's
                            # forward pass, corrupting every later prediction from this `abcnn`
                            # object.
                            model_sequential = single_model$mlp$clone()

                            model_sequential$add_module(name = "output_mu", module = single_model$mu)
                          }

                          # move tensors to a common device on cpu
                          # Avoid errors when training on CUDA
                          model_sequential$to(device = 'cpu')
                          self$model = model_sequential
                          model_input_dim = dim(x$sumstat)[2]
                          converter = innsight::convert(model_sequential,
                                                        input_dim = model_input_dim,
                                                        input_names = self$variables,
                                                        output_names = self$parameters)

                          self$converter = converter

                          self$print()
                        }

                        # END ON INIT
                      },

                      #' @description
                      #' Print the converter
                      #'
                      print = function() {
                        if (self$model_method == "tabnet-abc") {
                          warning("No converter for the Tabnet-ABC method.")
                        } else {
                          self$converter$print()
                        }
                      },

                      #' @description
                      #' Apply the `method` to the passed `data` to be explained
                      #'
                      #' The method is run on a `data` object (see `innsight` manual)
                      #' @param data (array, data.frame, torch_tensor or list)
                      #' The data to which the method is to be applied. These must have the same format as the input data of the passed model to the converter object. This means either
                      #' an array, data.frame, torch_tensor or array-like format of size (batch_size, dim_in), if e.g., the model has only one input layer, or
                      #' a list with the corresponding input data (according to the upper point) for each of the input layers.
                      #'
                      #' Note: For the model-agnostic methods, only models with a single input and output layer is allowed!
                      #'
                      #' @param data_ref (array, data.frame or torch_tensor)
                      #' The dataset to which the method is to be applied. These must have the same format as the input data of the passed model and has to be either matrix, an array, a data.frame or a torch_tensor.
                      #' Note: For the model-agnostic methods, only models with a single input and output layer is allowed!
                      #' @param method The method to run. Change the method specified in `new()`
                      #' @param output_scale The scale on which attributions are reported, so that they are
                      #' comparable across parameters whatever their `scale_target` (ignored for Tabnet-ABC):
                      #' - `prior_range` (default): the change in each parameter, expressed as a fraction of the
                      #'   width of its prior support (`target_summary$max - target_summary$min`). This is the
                      #'   scale `minmax` already trains on, so `minmax` attributions are left unchanged.
                      #' - `relative`: per sample and per parameter, attributions are divided by the sum of
                      #'   their absolute values over all summary statistics, giving signed shares whose absolute
                      #'   values sum to 1. Magnitude across parameters is lost, but only relative importance is kept.
                      #' - `none`: raw attributions of the network output, i.e. on the *scaled* target space.
                      #'   These are not comparable across parameters with different `scale_target` or ranges.
                      #'
                      #' @details
                      #' The network predicts the scaled target `z = f(x)`, and the parameter is `theta = g(z)`,
                      #' with `g` the backward transform of `scaler()`. By the chain rule
                      #' `d theta / dx = g'(z) * dz / dx`, so `prior_range` multiplies the attributions of each
                      #' parameter by `|g'(z)| / (max - min)` (see `scaler_grad()`). The factor is constant, and the
                      #' rescaling exact, for the affine scalings (`none`, `minmax`, `robustscaler`,
                      #' `normalization`). For `log` and `logit` it is a local linearisation evaluated at each
                      #' sample's prediction; for `cw`, which uses no data, it is evaluated at the median of the
                      #' scaled training targets instead. With `relative`, the factor `g'(z)` is shared by all
                      #' summary statistics of a sample and cancels out, so gradient-based results are invariant
                      #' to `scale_target` without linearisation.
                      #'
                      run = function(data,
                                     data_ref = NULL,
                                     method = NULL,
                                     output_scale = "prior_range") {
                        output_scale = match.arg(output_scale, c("prior_range", "relative", "none"))

                        # TODO Scale the new input data to the same scale as training data
                        data = scaler(data,
                                      self$input_summary,
                                      method = self$scale_input,
                                      type = "forward")

                        if (!is.null(data_ref)) {
                          data_ref = scaler(data_ref,
                                            self$input_summary,
                                            method = self$scale_input,
                                            type = "forward")
                        }

                        if (self$model_method == "tabnet-abc") {
                          sumstat = as.matrix(data)
                          colnames(sumstat) = colnames(self$x$sumstat_adj)
                          result = tabnet::tabnet_explain(self$x$fitted, sumstat)
                        } else {
                          # change the method if specified as argument
                          if (!is.null(method)) {self$method = method}

                          # Methods: cw (default), grad, smoothgrad, intgrad, expgrad, lrp, deeplift,
                          # deepshap, shap, lime
                          if (self$method == "cw") {
                            result = innsight::run_cw(self$converter) # no data argument is needed
                          }
                          if (self$method == "grad") {
                            result = innsight::run_grad(self$converter, data)
                          }
                          if (self$method == "smoothgrad") {
                            result = innsight::run_smoothgrad(self$converter, data)
                          }
                          if (self$method == "intgrad") {
                            result = innsight::run_intgrad(self$converter, data)
                          }
                          if (self$method == "expgrad") {
                            result = innsight::run_expgrad(self$converter, data)
                          }
                          if (self$method == "lrp") {
                            result = innsight::run_lrp(self$converter, data)
                          }
                          if (self$method == "deeplift") {
                            result = innsight::run_deeplift(self$converter, data)
                          }
                          if (self$method == "deepshap") {
                            result = innsight::run_deepshap(self$converter, data)
                          }
                          if (self$method == "shap") {
                            result = innsight::run_shap(self$converter, data, data_ref)
                          }
                          if (self$method == "lime") {
                            result = innsight::run_lime(self$converter, data, data_ref)
                          }
                        }

                        self$result = result

                        if (self$model_method != "tabnet-abc") {
                          private$rescale_result(data, output_scale)
                          self$output_scale = output_scale
                        }

                        # return(result)

                      },

                      #' @description
                      #' Get the results of the Feature Attribution method
                      #'
                      #' @param type the results can be returned as an `array`, `data.frame`, or `torch_tensor`
                      #'
                      #' @details
                      #' Note that when the `abcnn` model is `tabnet-abc`, `get_result()` returns importances weigths of the fitted model.
                      #'
                      #'
                      get_result = function(type = "array") {
                        if (self$model_method == "tabnet-abc") {
                          self$x$fitted$fit$importances
                        } else {
                          innsight::get_result(self$result, type = type)
                        }
                      },

                      #' @description
                      #' Plot the results of the Feature Attribution method
                      #' for single data points
                      #'
                      #' @param as_plotly If `TRUE`, plot the figure as a plotly object (default = `FALSE`)
                      #' @param type a character value. The type of plot for `Tabnet`,
                      #' passed to the Tabnet autoplot method.
                      #' Either `barplot` for importance scores averaged across masks,
                      #' `mask_agg`, for a single heatmap of aggregated mask importance per predictor along the dataset,
                      #' or `steps` for one heatmap at each mask step.
                      #' @param output_label character, the names of the variables to plot (if NULL, all variables are plotted)
                      #'
                      #' @details
                      #' Note that when the `abcnn` model is `tabnet-abc`, `plot()` returns the `autoplot()` function on the results of the `tabnet` model.
                      #'
                      plot = function(as_plotly = FALSE,
                                      type = "barplot",
                                      output_label = NULL) {
                        if (self$model_method == "tabnet-abc") {
                          if (type == "barplot") {
                            mask_importance = lapply(self$result$masks, colMeans)
                            mask_importance = dplyr::bind_rows(mask_importance)
                            mask_importance = as.data.frame(colMeans(mask_importance))

                            colnames(mask_importance) = "Importance"
                            mask_importance$variable = rownames(mask_importance)

                            p = ggplot2::ggplot(mask_importance, aes(x = variable, y = Importance)) +
                              geom_col(aes(fill = Importance)) +
                              xlab("Feature") + ylab("Importance") +
                              scale_fill_viridis_c()

                            return(p)

                          } else {
                            autoplot(self$result, type = type)
                          }
                        } else {
                          if (is.null(output_label)) {
                            output_label = self$parameters
                          } else {
                            output_label = output_label
                          }
                          # Plot individual results
                          # Interactive plots can also be created for both methods
                          result = self$get_result()
                          plot(self$result,
                               output_label = output_label,
                               as_plotly = as_plotly) +
                            theme_bw()
                        }
                      },

                      #' @description
                      #' Plot the results of the Feature Attribution method for the global dataset
                      #'
                      #' @param as_plotly If `TRUE`, plot the figure as a plotly object (default = `FALSE`)
                      #' @param output_label character, the names of the variables to plot (if NULL, all variables are plotted)
                      #'
                      plot_global = function(as_plotly = FALSE,
                                             output_label = NULL) {
                        if (self$model_method == "tabnet-abc") {
                          mask_importance = lapply(self$result$masks, colMeans)
                          mask_importance = dplyr::bind_rows(mask_importance)
                          mask_importance = as.data.frame(colMeans(mask_importance))

                          colnames(mask_importance) = "Importance"
                          mask_importance$variable = rownames(mask_importance)

                          p = ggplot2::ggplot(mask_importance, aes(x = variable, y = Importance)) +
                            geom_col(aes(fill = importance)) +
                            xlab("Feature") + ylab("Importance") +
                            scale_fill_viridis_c()

                          return(p)
                        } else {

                          if (is.null(output_label)) {
                            output_label = self$parameters
                          } else {
                            output_label = output_label
                          }

                          result = self$result
                          # Plot a aggregated plot of all given data points in argument 'data'
                          # Interactive plots can also be created for both methods
                          innsight::plot_global(result,
                                                output_label = output_label,
                                                as_plotly = as_plotly) +
                            theme_bw()
                        }
                      },

                      #' @description
                      #' Alias for `plot_global` for tabular and signal data
                      #'
                      #' @param as_plotly If `TRUE`, plot the figure as a plotly object (default = `FALSE`)
                      #'
                      boxplot = function(as_plotly = FALSE) {
                        if (self$model_method == "tabnet-abc") {
                          warning("'boxplot' not applicable to Tabnet-ABC.")
                        } else {
                          result = self$result
                          # Plot a aggregated plot of all given data points in argument 'data'
                          # Interactive plots can also be created for both methods
                          innsight::boxplot(result, as_plotly = as_plotly)
                        }
                      }
                    ),

                    private = list(
                      # The factor |g'(z)| / (max - min) that carries the attributions of each parameter
                      # from the scaled target space to fractions of its prior range (see `run()`).
                      # Returns a matrix with one column per parameter, and either one row per sample
                      # or a single row when the factor does not depend on the sample.
                      prior_range_factor = function(data) {
                        n_param = length(self$parameters)
                        method = if (length(self$scale_target) == 1) rep(self$scale_target, n_param) else self$scale_target
                        nonlinear = any(method %in% c("log", "logit"))

                        if (!nonlinear) {
                          # Affine scalings: the gradient does not depend on z
                          z = as.data.frame(matrix(0, nrow = 1, ncol = n_param))
                        } else if (self$method == "cw") {
                          if (is.null(self$target_center)) {
                            stop("The scaled training targets are required to rescale 'cw' attributions under 'log' or 'logit' scaling. Fit the `abcnn` model first, or use output_scale = 'relative'.")
                          }
                          warning("'cw' uses no data, so under 'log' or 'logit' target scaling its attributions are rescaled at the median of the scaled training targets.")
                          z = as.data.frame(matrix(self$target_center, nrow = 1))
                        } else {
                          self$model$eval()
                          z = torch::with_no_grad({
                            self$model(torch::torch_tensor(as.matrix(data), dtype = torch::torch_float()))
                          })
                          z = as.data.frame(torch::as_array(z))
                        }

                        grad = as.matrix(scaler_grad(z, self$target_summary, method))
                        prior_range = self$target_summary$max - self$target_summary$min
                        sweep(grad, 2, prior_range, "/")
                      },

                      # Rescale the attributions stored in `self$result` in place, so that `get_result()`,
                      # `plot()`, `plot_global()` and `boxplot()` all report them on `output_scale`.
                      rescale_result = function(data, output_scale) {
                        if (output_scale == "none") {return(invisible(NULL))}

                        # A single input and a single output layer: the attributions are an
                        # array of dim (batch, number of summary statistics, number of explained outputs)
                        res = self$result$result[[1]][[1]]
                        is_tensor = inherits(res, "torch_tensor")
                        r = if (is_tensor) torch::as_array(res$to(device = "cpu")) else res
                        if (length(dim(r)) != 3) {
                          stop("Unexpected layout of the innsight result, attributions could not be rescaled. Use output_scale = 'none'.")
                        }
                        batch_size = dim(r)[1]

                        # innsight explains a subset of the outputs (`output_idx`), in that order
                        out_idx = self$result$output_idx
                        out_idx = if (is.list(out_idx)) out_idx[[1]] else out_idx
                        if (is.null(out_idx)) {out_idx = seq_len(dim(r)[3])}

                        if (output_scale == "prior_range") {
                          factor = private$prior_range_factor(data)
                          if (nrow(factor) == 1) {
                            factor = factor[rep(1, batch_size), , drop = FALSE]
                          }
                          if (nrow(factor) != batch_size) {
                            stop("The number of samples in `data` does not match the number of attributions.")
                          }
                        }

                        for (k in seq_along(out_idx)) {
                          r_k = r[, , k, drop = FALSE]
                          if (output_scale == "prior_range") {
                            # Recycled along the first (batch) dimension
                            r[, , k] = r_k * factor[, out_idx[k]]
                          }
                          if (output_scale == "relative") {
                            total = apply(abs(r_k), 1, sum)
                            r[, , k] = r_k / pmax(total, .Machine$double.eps)
                          }
                        }

                        self$result$result[[1]][[1]] = if (is_tensor) torch::torch_tensor(r, dtype = res$dtype) else r
                        invisible(NULL)
                      }
                    )
)


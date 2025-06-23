library(innsight)
library(plotly)

#' An `explain` object for feature attribution
#' A R6 class object
#'
#' See `https://bips-hb.github.io/innsight/articles/innsight.html` for details.
#'
#' @param abcnn An `abcnn` object
#' @param method A feature attribution method, as named in the `ìnnsight` R package
#' including 'cw' (default), 'grad', 'smoothgrad', 'intgrad', 'expgrad', 'lrp', 'deeplift',
#' 'deepshap', 'shap', 'lime.' No method required for tabnet-ABC.
#' @param ensemble_num_model index of the model when the network is a deep ensemble (default = 1)
#'
#'
#' @slot converter Stores the `innsight::converter` object
#'
#' @slot result stores results of the `explain$run()` method.
#' @slot model_method method of the trained neural network (e.g. "concrete dropout")
#' @slot variables names of the variables (summary statistics)
#' @slot parameters names of the parameter to infer
#' @slot ensemble_num_model index of the model when the network is a deep ensemble
#' @slot scale_input the `abcnn$scale_input` slot from the `abcnn` input object
#' @slot input_summary the `abcnn$input_summary` slot from the `abcnn` input object
#'
#' @import torch
#' @import luz
#' @import ggplot2
#' @import innsight
#' @import plotly
#' @import R6Class
#' @import RColorBrewer
#' @import janitor
#'
#' @return an `explain` object
#' @export
#'
explain = R6::R6Class("explain",
                    public = list(
                      x = NULL,
                      method = NULL,
                      converter = NULL,
                      result = NULL,
                      model_method = NULL,
                      variables = NULL,
                      parameters = NULL,
                      ensemble_num_model = NULL,
                      scale_input = NULL,
                      input_summary = NULL,

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

                        # Tabnet-ABC has its own set of methods
                        if (self$model_method == "tabnet-abc") {
                          print("Note that Tabnet-ABC has its own set of explainability methods.")
                        } else {
                          # Convert the abccn object passed as input
                          if (self$model_method == "monte carlo dropout") {
                            model = x$fitted$model
                            model_mc = model$mc_dropout

                            model_sequential = torch::nn_sequential(model_mc[[1]])

                            for (i in 2:x$num_hidden_layers) {
                              mod = model_mc[[(i -1) * 3 + 1]]
                              model_sequential$add_module(name = i - 1, module = mod)
                            }

                            mod = x$fitted$model$mc_dropout$output
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

                            model_sequential = torch::nn_sequential(model_concrete[[1]]$linear)

                            for (i in 2:x$num_hidden_layers) {
                              mod = model_concrete[[i]]$linear
                              model_sequential$add_module(name = i - 1, module = mod)
                            }

                            mod = x$fitted$model$linear_mu
                            model_sequential$add_module(name = "output_mu", module = mod)

                          }

                          if (self$model_method == "deep ensemble") {
                            # FOR A DEEP ENSEMBLE MODEL
                            model = x$fitted$model
                            # Extract one model
                            single_model = model$model_list[[self$ensemble_num_model]]
                            # Extract nn_sequential
                            model_sequential = single_model$mlp

                            model_sequential$add_module(name = "output_mu", module = single_model$mu)
                          }

                          # move tensors to a common device on cpu
                          # Avoid errors when training on CUDA
                          model_sequential$to(device = 'cpu')
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

                      # Print the converter
                      print = function() {
                        if (self$model_method == "tabnet-abc") {
                          warning("No converter for the Tabnet-ABC method.")
                        } else {
                          self$converter$print()
                        }
                      },

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
                      #'
                      run = function(data,
                                     data_ref = NULL,
                                     method = NULL) {
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

                        # return(result)

                      },

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

                      #' Plot the results of the Feature Attribution method
                      #' for single data points
                      #'
                      #' @param as_plotly If `TRUE`, plot the figure as a plotly object (default = `FALSE`)
                      #' @param type a character value. Passed to the Tabnet autoplot method. Either `mask_agg` the default, for a single heatmap of aggregated mask importance per predictor along the dataset, or `steps` for one heatmap at each mask step.
                      #'
                      #' @details
                      #' Note that when the `abcnn` model is `tabnet-abc`, `plot()` returns the `autoplot()` function on the results of the `tabnet` model.
                      #'
                      plot = function(as_plotly = FALSE,
                                      type = "mask_agg") {
                        if (self$model_method == "tabnet-abc") {
                          autoplot(self$result, type = type)
                        } else {
                          # Plot individual results
                          # Interactive plots can also be created for both methods
                          result = self$get_result()
                          plot(self$result, as_plotly = as_plotly) +
                            theme_bw()
                        }
                      },

                      #' Plot the results of the Feature Attribution method
                      #' for the global dataset
                      #'
                      #' @param as_plotly If `TRUE`, plot the figure as a plotly object (default = `FALSE`)
                      #'
                      plot_global = function(as_plotly = FALSE) {
                        if (self$model_method == "tabnet-abc") {
                          warning("'plot_global' not applicable to Tabnet-ABC.")
                        } else {
                          result = self$result
                          # Plot a aggregated plot of all given data points in argument 'data'
                          # Interactive plots can also be created for both methods
                          innsight::plot_global(result, as_plotly = as_plotly) +
                            theme_bw()
                        }
                      },

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
                    )
)


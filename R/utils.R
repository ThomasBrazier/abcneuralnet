#' Save the `abcnn` object and the serialized luz fitted model
#'
#' The function will save a `_luz.Rds`, a `_model.Rds` and a `_abcnn.Rds`,
#' which will contain the `luz` fitted model, the original `torch` model
#' and the `abcnn` model. The `abcnn` model will be reconstructed with `load_abcnn()`.
#'
#' @param object an `abcnn` object with a `luz` fitted model
#' @param prefix character, the prefix with path of the saved .Rds object
#'
#' @import bundle
#' @import torch
#' @import luz
#'
#' @export
#'
save_abcnn = function(object, prefix = "") {

  saveRDS(object, paste0(prefix, "_abcnn.Rds"))

  if (object$method == "tabnet-abc") {
    # See https://github.com/mlverse/tabnet/issues/34
    # Serialization
    torch::torch_save(object$fitted$fit$network, paste0(prefix, "_model"))
  } else {
    # Save the luz fitted object
    luz::luz_save(object$fitted, paste0(prefix, "_luz.Rds"))
    # mod = bundle::bundle(object$model)
  }
}


#' Load an `abcnn` object and the serialized luz fitted model
#'
#' The function reconstructs an `abcnn` object from the `_luz.Rds`, `_model.Rds` and `_abcnn.Rds` files.
#'
#' @param prefix character, the prefix with path of the saved .Rds object
#'
#' @import bundle
#' @import torch
#' @import luz
#'
#' @export
#'
#' @return an `abcnn` object
#'
load_abcnn = function(prefix = "") {

  object = readRDS(paste0(prefix, "_abcnn.Rds"))

  if (object$method == "tabnet-abc") {
    # Loading
    tabnet_network = torch::torch_load(paste0(prefix, "_model"))
    object$fitted$fit$network = tabnet_network
  } else {
    object$fitted = luz::luz_load(paste0(prefix, "_luz.Rds"))
    # mod = readRDS(paste0(prefix, "_model.Rds"))
    # object$model = bundle::unbundle(mod)
  }

  dev = ifelse(torch::cuda_is_available(), "cuda", "cpu")
  object$device = torch::torch_device(dev)
  if (object$method == "tabnet-abc") {
    object$fitted$fit$network$to(device = dev)
    object$fitted$fit$config$device = dev
  } else {
    object$fitted$model$to(device = dev)
  }
  

  return(object)
}


#' Compute the log1pexp trick
#'
#' @param x a tensor
#' @param threshold the threshold value under which the trick is applied to avoid `Inf` values
#'
#' @description
#' This is a more stable version of log(1 + exp(x)). Note that log(1 + exp(x)) is approximately equal to x when x is large enough.
#' See https://stackoverflow.com/questions/60903821/how-to-prevent-inf-while-working-with-exponential for details
#'
#' @return a tensor with values corrected with the log1pexp trick
#'
log1pexp = function(x, threshold = 10) {
  torch::torch_where(x < threshold, torch::torch_log1p(torch::torch_exp(x)) + 1e-6, x)
}



# A numerical method to avoid Inf when qlogis(0)
squeeze = function(p, n) (p * (n - 1) + 0.5) / n
unsqueeze = function(p, n) (p * n - 0.5) / (n - 1)

# A logit transformation
#' @param z a vector of numerical values to transform
#' @param a the min value of the training set
#' @param b the max value of the training set
#' @param n the number of samples in the training set
logit = function(y, a, b, n) {
  p = (y - a) / (b - a)
  # Trick to avoid Inf for qlogis(0)
  p = squeeze(p, n)
  return(qlogis(p))
}

#' The backward logit transform
#' @param z a vector of numerical values to transform
#' @param a the min value of the training set
#' @param b the max value of the training set
#' @param n the number of samples in the training set
inv_logit = function(z, a, b, n) {
  p = plogis(z)
  p = unsqueeze(p, n)
  # `unsqueeze()` undoes the `squeeze()` applied on the forward pass, but it
  # overshoots [0, 1] by up to 0.5/(n - 1) once `plogis(z)` saturates. Clamp, as
  # a parameter cannot fall outside the range learned on the training set.
  # p = pmin(pmax(p, 0), 1)
  # Keep it this way, I want to know when the CI is wrong (outside training range)
  a + (b - a) * p
}

#' The derivative of the backward logit transform
#'
#' @description
#' The derivative of `inv_logit()` with respect to `z`. As `unsqueeze()` is
#' affine in `p` with slope `n / (n - 1)`, and `d plogis(z) / dz = p * (1 - p)`,
#' the chain rule gives `(b - a) * n / (n - 1) * p * (1 - p)`.
#'
#' @param z a vector of numerical values
#' @param a the min value of the training set
#' @param b the max value of the training set
#' @param n the number of samples in the training set
inv_logit_grad = function(z, a, b, n) {
  p = plogis(z)
  (b - a) * (n / (n - 1)) * p * (1 - p)
}



#' A scaling function for targets and inputs
#'
#' @description
#'
#' The function allows to back-transform the numerical values to their original scale.
#' For this, it requires a list of summary statistics learned on the training set.
#'
#' @param x a data frame to scale, each column is scaled separately
#' @param sum_stats list, summary statistics learned on the data to back-transform
#' @param method the scaling method, either `minmax`, `robustscaler`, `normalization`, `log`, `logit` or `none`. Can be a single character (same transformation applied to all columns) or a vector of characters with one transformation per column.
#' @param type is `forward` when scaling inputs or targets and `backward` when back-transforming targets at prediction time
#'
#' @return a data frame with scaled values
#' 
#' @export
#'
scaler = function(x, sum_stats, method = "minmax", type = "forward") {

  x = as.data.frame(x)

  # Raise an error if the method is not provided
  l = lapply(method, function(x) (x %in% c("none", "minmax", "robustscaler", "log", "logit", "normalization")))
  if (!all(unlist(l))) {
    stop("The scaling method must be provided.")
  }
  
  method = if(length(method) == 1) {rep(method, ncol(x))} else {method}
  
  # Process each column one by one
  for (i in 1:ncol(x)) {
    if (method[i] == "none") {
      # Do nothing
    }
    else {
      if (type == "forward") {
        if (method[i] == "minmax") {
          x[,i] = (x[,i] - sum_stats$min[i]) / (sum_stats$max[i] - sum_stats$min[i])
        }
        if (method[i] == "normalization") {
          x[,i] = (x[,i] - sum_stats$mean[i]) / (sum_stats$sd[i])
        }
        if (method[i] == "robustscaler") {
          x[,i] = (x[,i] - sum_stats$quantile_25[i]) / (sum_stats$quantile_75[i] - sum_stats$quantile_25[i])
        }
        if (method[i] == "log") {
          x[,i] = log(x[,i])
        }
        if (method[i] == "logit") {
          # x[,i] = logit(x[,i], sum_stats$min[i], sum_stats$max[i], sum_stats$n_training[i])
          x[,i] = logit(x[,i], sum_stats$min[i], sum_stats$max[i], 100000)
        }
      }
      if (type == "backward") {
        if (method[i] == "minmax") {
          x[,i] = (x[,i] * (sum_stats$max[i] - sum_stats$min[i])) + sum_stats$min[i]
        }
        if (method[i] == "robustscaler") {
          x[,i] = (x[,i] * (sum_stats$quantile_75[i] - sum_stats$quantile_25[i])) + sum_stats$quantile_25[i]
        }
        if (method[i] == "normalization") {
          x[,i] = (x[,i] * (sum_stats$sd[i])) + sum_stats$mean[i]
        }
        if (method[i] == "log") {
          x[,i] = exp(x[,i])
        }
        if (method[i] == "logit") {
          # x[,i] = inv_logit(x[,i], sum_stats$min[i], sum_stats$max[i], sum_stats$n_training[i])
          x[,i] = inv_logit(x[,i], sum_stats$min[i], sum_stats$max[i], 100000)
        }
      }
    }
  }
  
  return(x)
  # if (method == "none") {
  #   # Do nothing
  #   return(x)
  # }
  # else {
  #   if (type == "forward") {
  #     if (method == "minmax") {
  #       x_scaled = data.frame(lapply(1:ncol(x), function(i) {(x[,i,drop=F] - sum_stats$min[i]) / (sum_stats$max[i] - sum_stats$min[i])}))
  #       return(x_scaled)
  #     }
  #     if (method == "normalization") {
  #       x_scaled = data.frame(lapply(1:ncol(x), function(i) {(x[,i,drop=F] - sum_stats$mean[i]) / (sum_stats$sd[i])}))
  #       return(x_scaled)
  #     }
  #     if (method == "robustscaler") {
  #       x_scaled = data.frame(lapply(1:ncol(x), function(i) {(x[,i,drop=F] - sum_stats$quantile_25[i]) / (sum_stats$quantile_75[i] - sum_stats$quantile_25[i])}))
  #       return(x_scaled)
  #     }
  #     if (method == "log") {
  #       x_scaled = data.frame(lapply(1:ncol(x), function(i) {log(x[,i,drop=F])}))
  #       return(x_scaled)
  #     }
  #     if (method == "logit") {
  #       x_scaled = data.frame(lapply(1:ncol(x), function(i) {logit(x[,i,drop=F], sum_stats$min[i], sum_stats$max[i])}))
  #       return(x_scaled)
  #     }
  #   }
  #   if (type == "backward") {
  #     if (method == "minmax") {
  #       x_scaled = data.frame(lapply(1:ncol(x), function(i) {(x[,i,drop=F] * (sum_stats$max[i] - sum_stats$min[i])) + sum_stats$min[i]}))
  #       return(x_scaled)
  #     }
  #     if (method == "robustscaler") {
  #       x_scaled = data.frame(lapply(1:ncol(x), function(i) {(x[,i,drop=F] * (sum_stats$quantile_75[i] - sum_stats$quantile_25[i])) + sum_stats$quantile_25[i]}))
  #       return(x_scaled)
  #     }
  #     if (method == "normalization") {
  #       x_scaled = data.frame(lapply(1:ncol(x), function(i) {(x[,i,drop=F] * (sum_stats$sd[i])) + sum_stats$mean[i]}))
  #       return(x_scaled)
  #     }
  #     if (method == "log") {
  #       x_scaled = data.frame(lapply(1:ncol(x), function(i) {exp(x[,i,drop=F])}))
  #       return(x_scaled)
  #     }
  #     if (method == "logit") {
  #       x_scaled = data.frame(lapply(1:ncol(x), function(i) {inv_logit(x[,i,drop=F], sum_stats$min[i], sum_stats$max[i])}))
  #       return(x_scaled)
  #     }
  #   }
  # }
}


#' The gradient of the backward scaling transform
#'
#' @description
#'
#' Returns `|g'(z)|`, the absolute derivative of the backward transform applied
#' by `scaler(type = "backward")`, evaluated at the scaled values `z`.
#'
#' This is the Jacobian factor used to carry a standard deviation from the
#' scaled space, where the neural network is trained, to the original parameter
#' scale with the delta method: `sd_original ~ |g'(z)| * sd_scaled`.
#'
#' The factor is a constant, and the delta method therefore exact, for the
#' affine methods (`none`, `minmax`, `robustscaler` and `normalization`).
#' For `log` and `logit` the backward transform is non-linear, so the result is
#' only a local linearisation around `z` and the resulting symmetric interval
#' may fall outside the support of the parameter. Prefer the conformal or
#' posterior-quantile intervals returned by `abcnn$predictions()`, which are
#' built by transforming interval endpoints and are exact under any monotone
#' scaling.
#'
#' @param z a data frame of scaled values at which to evaluate the gradient,
#' typically the scaled predictive mean. Each column is treated separately.
#' @param sum_stats list, summary statistics learned on the training data. See `scaler()`.
#' @param method the scaling method, either `minmax`, `robustscaler`, `normalization`, `log`, `logit` or `none`.
#' Can be a single character (same transformation applied to all columns) or a vector of characters with one transformation per column.
#'
#' @return a data frame of gradients with the same dimensions as `z`
#'
scaler_grad = function(z, sum_stats, method = "minmax") {

  z = as.data.frame(z)

  # Raise an error if the method is not provided
  l = lapply(method, function(x) (x %in% c("none", "minmax", "robustscaler", "log", "logit", "normalization")))
  if (!all(unlist(l))) {
    stop("The scaling method must be provided.")
  }

  method = if(length(method) == 1) {rep(method, ncol(z))} else {method}

  grad = z

  # Process each column one by one, as in scaler()
  for (i in 1:ncol(z)) {
    if (method[i] == "none") {
      grad[,i] = 1
    }
    if (method[i] == "minmax") {
      grad[,i] = sum_stats$max[i] - sum_stats$min[i]
    }
    if (method[i] == "robustscaler") {
      grad[,i] = sum_stats$quantile_75[i] - sum_stats$quantile_25[i]
    }
    if (method[i] == "normalization") {
      grad[,i] = sum_stats$sd[i]
    }
    if (method[i] == "log") {
      grad[,i] = exp(z[,i])
    }
    if (method[i] == "logit") {
      # Same hardcoded n as the logit branch of scaler()
      grad[,i] = inv_logit_grad(z[,i], sum_stats$min[i], sum_stats$max[i], 100000)
    }
  }

  return(abs(grad))
}


#' Cross-validation metrics between ground truth and predictions
#'
#' @description
#'
#' The function computes cross-validation metrics between ground truth and predictions
#'
#' @param cross_validation_param a tidy data frame with ground truth simulated parameters
#' @param cross_validation_predictions a tidy  data frame with predictions
#' 
#' @importFrom stats cor
#' @importFrom stats cov
#'
#' @return a data frame with cross-validation metrics
#'
cross_val = function(cross_validation_param,
                     cross_validation_predictions) {

  cross_validation_predictions$true_value = cross_validation_param$true_value

  res = cross_validation_predictions %>%
    dplyr::group_by(.data$parameter) %>%
    dplyr::summarise(n = n(),
              mae = rminer::mmetric(.data$predictive_mean, .data$true_value, metric = "MAE"),
              mse = rminer::mmetric(.data$predictive_mean, .data$true_value, metric = "MSE"),
              rmse = rminer::mmetric(.data$predictive_mean, .data$true_value, metric = "RMSE"),
              nmae = rminer::mmetric(.data$predictive_mean, .data$true_value, metric = "NMAE"),
              cor = stats::cor(.data$predictive_mean, .data$true_value, method = "spearman"),
              cov = stats::cov(.data$predictive_mean, .data$true_value),
              mean_epistemic_interval = mean(.data$epistemic_conformal_upper - .data$epistemic_conformal_lower),
              mean_overall_interval = mean(.data$overall_conformal_upper - .data$overall_conformal_lower))

  return(res)
}


# TODO Sample table

#' Return a data frame with sample sizes of an `abcnn` object
#'
#' The function returns all the sample sizes of training, testing, evaluation, conformal prediction and observed for an `abcnn` object.
#'
#' @param object an `abcnn` R6 class object
#'
#' @import torch
#'
#' @export
#'
#' @return a `data.frame` with sample sizes
#'
samples_abcnn = function(object) {

  # The same arithmetic as `abcnn$dataloader()`: the evaluation and testing
  # sets are proportions of all the simulations, and the training set is what
  # is left once they and the conformal calibration set have been taken out.
  n_total = nrow(object$sumstat)
  n_evaluation = round(n_total * object$validation_split, digits = 0)
  n_testing = round(n_total * object$test_split, digits = 0)
  n_training = n_total - n_evaluation - n_testing - object$num_conformal

  samples = data.frame(Sample = c("Simulations",
                                  "Training",
                                  "Testing split",
                                  "Testing",
                                  "Evaluation split",
                                  "Evaluation",
                                  "Conformal",
                                  "Observed"),
                       Size = c(n_total,
                                n_training,
                                object$test_split,
                                n_testing,
                                object$validation_split,
                                n_evaluation,
                                object$num_conformal,
                                nrow(object$observed)))

  return(samples)
}




# TODO Hyperparam table

#' Return a data frame with hyperparameters of an `abcnn` object
#'
#' The function returns all the hyperparameters of the neural network in an `abcnn` object.
#'
#' @param object an `abcnn` R6 class object
#'
#' @import torch
#'
#' @export
#'
#' @return a `data.frame` with hyperparameters
#'
hyperparams_abcnn = function(object) {

  hyperparams = data.frame(Hyperparameter = c("Method",
                                      "Scaling for inputs (summary statistics)",
                                      "Scaling for targets (theta)",
                                      "Number hidden layers",
                                      "Number hidden dimensions",
                                      "Batch size",
                                      "Epochs",
                                      "Early stopping callback",
                                      "Patience for early stopping",
                                      "Learning rate",
                                      "L2 weight decay",
                                      "Method for ABC",
                                      "Tolerance rate (ABC)",
                                      "Number of posterior samples (mc dropout and concrete dropout)",
                                      "Dropout rate",
                                      "Number of networks (deep ensemble)"),
                       Value = c(object$method,
                                 paste(object$scale_input, collapse = ", "),
                                 paste(object$scale_target, collapse = ", "),
                                 object$num_hidden_layers,
                                 object$num_hidden_dim,
                                 object$batch_size,
                                 object$epochs,
                                 as.character(object$early_stopping),
                                 object$patience,
                                 object$learning_rate,
                                 object$l2_weight_decay,
                                 object$abc_method,
                                 ifelse(is.null(object$tol), "NULL", object$tol),
                                 object$num_posterior_samples,
                                 object$dropout,
                                 object$num_networks))

  return(hyperparams)
}





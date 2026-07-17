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
squeeze = function(p, n = length(p)) (p * (n - 1) + 0.5) / n

# A logit transformation
logit = function(y, a, b) {
  p = (y - a) / (b - a)
  p = squeeze(p)
  return(qlogis(p))
}

# The backward logit transform
inv_logit = function(z, a, b, n = length(z)) {
  p = plogis(z)
  p = (p * n - 0.5) / (n - 1)
  a + (b - a) * p
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
          x[,i] = logit(x[,i], sum_stats$min[i], sum_stats$max[i])
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
          x[,i] = inv_logit(x[,i], sum_stats$min[i], sum_stats$max[i])
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
              mean_epistemic_interval = 2 * mean(.data$epistemic_conformal_credible_interval),
              mean_overall_interval = 2 * mean(.data$overall_conformal_credible_interval))

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

  samples = data.frame(Sample = c("Training",
                                  "Testing split",
                                  "Testing",
                                  "Evaluation split",
                                  "Evaluation",
                                  "Conformal",
                                  "Observed"),
                       Size = c(round(nrow(object$sumstat) * (1 - object$validation_split), digits = 0),
                                object$test_split,
                                round(nrow(object$sumstat) * (1 - object$validation_split) * object$test_split, digits = 0),
                                object$validation_split,
                                round(nrow(object$sumstat) * (object$validation_split), digits = 0),
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





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
  # Save the torch module used as model
  # torch_save(object$model, paste0(prefix, "_torch.Rds"))

  # Save the luz fitted object
  luz::luz_save(object$fitted, paste0(prefix, "_luz.Rds"))
  # bun = bundle::bundle(object$fitted)
  # saveRDS(bun, paste0(prefix, "_bundle.Rds"))

  mod = bundle::bundle(object$model)
  saveRDS(mod, paste0(prefix, "_model.Rds"))

  # Save the abcnn object
  # Remove torch module and luz fitted to avoid serialization issues
  # object$model = NULL
  # object$fitted = NULL

  # export = list(object = object,
  #               metrics = object$fitted$records$metrics)

  saveRDS(object, paste0(prefix, "_abcnn.Rds"))
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

  # object = import$object

  # object$fitted = luz::luz_load(paste0(prefix, "_luz.Rds"))
  # bun = readRDS(paste0(prefix, "_bundle.Rds"))
  # object$fitted = bundle::unbundle(bun)
  object$fitted = luz::luz_load(paste0(prefix, "_luz.Rds"))

  # object$fitted$records$metrics = import$metrics

  mod = readRDS(paste0(prefix, "_model.Rds"))
  object$model = bundle::unbundle(mod)

  object$device = torch::torch_device(if (torch::cuda_is_available()) {"cuda"} else {"cpu"})

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




#' A scaling function for targets and inputs
#'
#' @description
#'
#' The function allows to back-transform the numerical values to their original scale.
#' For this, it requires a list of summary statistics learned on the training set.
#'
#' @param x a data frame to scale, each column is scaled separately
#' @param sum_stats list, summary statistics learned on the data to back-transform
#' @param method the scaling method, either `minmax`, `robustscaler`, `normalization` or `none`
#' @param type is `forward` when scaling inputs or targets and `backward` when back-transforming targets at prediction time
#'
#' @return a data frame with scaled values
#'
scaler = function(x, sum_stats, method = "minmax", type = "forward") {

  x = as.data.frame(x)

  if (method == "none") {
    # Do nothing
    return(x)
  }
  else {
    if (type == "forward") {
      if (method == "minmax") {
        x_scaled = data.frame(lapply(1:ncol(x), function(i) {(x[,i,drop=F] - sum_stats$min[i]) / (sum_stats$max[i] - sum_stats$min[i])}))
        return(x_scaled)
      }
      if (method == "normalization") {
        x_scaled = data.frame(lapply(1:ncol(x), function(i) {(x[,i,drop=F] - sum_stats$mean[i]) / (sum_stats$sd[i])}))
        return(x_scaled)
      }
      if (method == "robustscaler") {
        x_scaled = data.frame(lapply(1:ncol(x), function(i) {(x[,i,drop=F] - sum_stats$quantile_25[i]) / (sum_stats$quantile_75[i] - sum_stats$quantile_25[i])}))
        return(x_scaled)
      }
    }
    if (type == "backward") {
      if (method == "minmax") {
        x_scaled = data.frame(lapply(1:ncol(x), function(i) {(x[,i,drop=F] * (sum_stats$max[i] - sum_stats$min[i])) + sum_stats$min[i]}))
        return(x_scaled)
      }
      if (method == "robustscaler") {
        x_scaled = data.frame(lapply(1:ncol(x), function(i) {(x[,i,drop=F] * (sum_stats$quantile_75[i] - sum_stats$quantile_25[i])) + sum_stats$quantile_25[i]}))
        return(x_scaled)
      }
      if (method == "normalization") {
        x_scaled = data.frame(lapply(1:ncol(x), function(i) {(x[,i,drop=F] * (sum_stats$sd[i])) + sum_stats$mean[i]}))
        return(x_scaled)
      }
    }
  }
}

log1pexp = function(x, threshold = 10) {
  # more stable version of log(1 + exp(x))
  #  Notice that log(1 + exp(x)) is approximately equal to x when x is large enough.
  # https://stackoverflow.com/questions/60903821/how-to-prevent-inf-while-working-with-exponential
  torch::torch_where(x < threshold, torch::torch_log1p(torch::torch_exp(x)) + 1e-6, x)
}




# A scaling function
# x, a data frame to scale
# method is either "minmax", "robustscaler", "normalization" or "none"
# It requires a list of summary statistics learned on the training set
# type is "forward" when scaling inputs or targets
# and "backward" when back-transforming targets at prediction time
scaler = function(x, sum_stats, method = "minmax", type = "forward") {

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

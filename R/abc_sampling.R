#' Compute the Epanechnikov kernel between two points or sets of points.
epanechnikov_kernel = function(x1, x2, bandwidth=bandwidth) {
  # Eculidean distance
  dist = unlist(lapply(1:nrow(x1), function(x) {dist(rbind(x1[x,], x2))}))
  # Normalize by bandwidth
  if (bandwidth == "max") {
    bandwidth = max(dist)
  }
  dist = dist / bandwidth
  # Weights of the kernel smoothing function
  weights = ifelse(abs(dist) <= 1, 3 / 4 * (1 - dist^2), 0)
  return(weights)
}


#' Compute the RBF kernel between two points or sets of points.
rbf_kernel = function(x1, x2, length_scale=1.0) {
  # Euclidean distance
  sq_dist = unlist(lapply(1:nrow(x1), function(x) {dist(rbind(x1[x,], x2))^2}))
  # RBF kernel
  weights = exp(-sq_dist / (2 * length_scale^2))
  return(weights)
}



# ABC sampling functions
#' Rejection sampling
rejection_sampling = function(kernel_values, theta, tol=0.01) {
  # Sample posterior based on the weighted kernel values
  # Keep values above the tolerance threshold
  sampled_indices = which(kernel_values > quantile(kernel_values, 1-tol))
  # sampled_indices = sample(c(1:nrow(theta)), size=size, p=kernel_values)
  # return(theta[sampled_indices,])
  return(sampled_indices)
}

#' Importance sampling
importance_sampling = function(kernel_values, theta, tol=0.01) {
  # Randomly sample indices based on the weights
  size = tol * length(kernel_values)
  sampled_indices = sample(1:nrow(theta), size=size, p=kernel_values)
  # return(theta[sampled_indices,])
  return(sampled_indices)
}

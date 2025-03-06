#------------------------------------------
# KERNELS
#------------------------------------------
# See ?density
# kernel = c("gaussian", "epanechnikov", "rectangular",
#            "triangular", "biweight",
#            "cosine", "optcosine")
library(pbmcapply)

#' Generic density kernel
density_kernel = function(x1, x2,
                          kernel = "epanechnikov",
                          bandwidth = 1.0,
                          length_scale = 1.0,
                          ncores = 1) {
  if (kernel == "epanechnikov") {
    # One measure per observation
    w = pbmclapply(1:nrow(x2), function(i) {epanechnikov_kernel(x1, x2[i,,drop=F], bandwidth = bandwidth)},
                   mc.cores = ncores)
  }
  if (kernel == "rbf") {
    w = pbmclapply(1:nrow(x2), function(i) {rbf_kernel(x1, x2[i,,drop=F], length_scale)},
                   mc.cores = ncores)
  }
  if (kernel == "gaussian") {
    w = pbmclapply(1:nrow(x2), function(i) {gaussian_kernel(x1, x2[i,,drop=F])},
                   mc.cores = ncores)
  }
  # Do the mean distance across observations
  # cols are training samples
  # rows are observations
  t = do.call(rbind, w)
  t = colMeans(t)

  return(t)
}


#' Compute the Epanechnikov kernel between two points or sets of points.
epanechnikov_kernel = function(x1, x2, bandwidth = 1.0) {
  # Euclidean distance, one distance per sumstat (col)
  # dist_obs = rbind(lapply(1:nrow(x2), function(y) rbind(lapply(1:nrow(x1), function(x) {dist(rbind(x1[x,], x2[y,]))}))))

  dist_obs = unlist(lapply(1:nrow(x1), function(x) {dist(rbind(x1[x,], x2))}))

  # dist_obs = as.data.frame(matrix(NA, nrow = nrow(x2), ncol = nrow(x1)))
  # for (j in 1:nrow(x2)) {
  #   # cat(j, "\n")
  #   tmp = unlist(lapply(1:nrow(x1), function(x) {dist(rbind(x1[x,], x2[j,]))}))
  #   dist_obs[j,] = tmp
  #   rm(tmp)
  # }

  # Normalize by bandwidth
  if (bandwidth == "max") {
    bandwidth = max(dist_obs)
  }
  weights = dist_obs / bandwidth
  # Weights of the kernel smoothing function
  # weights = data.frame(apply(weights, c(1,2), function(x) {ifelse(abs(x) <= 1, 3 / 4 * (1 - x^2), 0)}))
  weights = unlist(lapply(1:length(dist_obs), function(x) {ifelse(abs(dist_obs[x]) <= 1, 3 / 4 * (1 - dist_obs[x]^2), 0)}))
  # weights = ifelse(abs(dist_obs) <= 1, 3 / 4 * (1 - dist_obs^2), 0)
  return(weights)
}


#' Compute the RBF kernel between two points or sets of points.
rbf_kernel = function(x1, x2, length_scale = 1.0) {
  # Euclidean distance
  sq_dist = unlist(lapply(1:nrow(x1), function(x) {dist(rbind(x1[x,], x2))^2}))
  # RBF kernel
  weights = exp(-sq_dist / (2 * length_scale^2))
  return(weights)
}


#' Compute a Gaussian kernel between two points or sets of points.
gaussian_kernel = function(x1, x2, length_scale = 1.0) {
  # Euclidean distance, one distance per sumstat (col)
  dist_obs = unlist(lapply(1:nrow(x1), function(x) {dist(rbind(x1[x,], x2))}))
  # Gaussian kernel
  weights = dnorm(dist_obs)
  return(weights)
}

#------------------------------------------
# SAMPLING
#------------------------------------------
# ABC sampling functions
abc_sampling = function(kernel_values,
                        theta,
                        method = "importance",
                        tol = 0.01) {
  if (method == "rejection") {
    idx = rejection_sampling(kernel_values, theta, tol = tol)
  }
  if (method == "importance") {
    idx = importance_sampling(kernel_values, theta, tol = tol)
  }

  return(idx)
}



#' Rejection sampling
rejection_sampling = function(kernel_values, theta, tol = 0.1) {
  # Sample posterior based on the weighted kernel values
  # Keep values above the tolerance threshold
  sampled_indices = which(kernel_values > quantile(kernel_values, 1 - tol))
  # sampled_indices = sample(c(1:nrow(theta)), size=size, p=kernel_values)
  # return(theta[sampled_indices,])
  return(sampled_indices)
}

#' Importance sampling
importance_sampling = function(kernel_values, theta, tol = 0.1) {
  # Randomly sample indices based on the weights
  size = round(tol * length(kernel_values), digits = 0)
  sampled_indices = sample(1:nrow(theta), size = size, p = kernel_values)
  # return(theta[sampled_indices,])
  return(sampled_indices)
}

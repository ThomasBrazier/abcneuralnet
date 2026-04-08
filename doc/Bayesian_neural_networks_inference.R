## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.align = "center",
  fig.width = 8,
  fig.height = 4,
  warning = FALSE,
  message = FALSE
)

## ----setup, echo = F, eval = F------------------------------------------------
# # Install ABCNeuralNet
# devtools::install_github("ThomasBrazier/abcneuralnet")
# 
# # Install C++ dependencies of torch (e.g. CUDA, lantern)
# torch::install_torch()
# # After installing torch CUDA dependencies, the R session must be reloaded

## ----setup3, echo = F---------------------------------------------------------
library(abcneuralnet)
library(ggplot2)
library(torch)
library(tidyverse)
library(kableExtra)

## ----toy_data_1, echo = F, eval=T---------------------------------------------
n_train = 100000 # Number of data points
n_obs = 1000 # Validation size

gen_data_1d = function(n) {
  sigma = 1
  X = matrix(rnorm(n))
  w = 2
  b = 8
  Y = matrix(X %*% w + b + sigma * rnorm(n))
  list(X = X, Y = Y)
}

XY = gen_data_1d(n_train + n_obs)

X_train = XY$X[1:n_train]
Y_train = XY$Y[1:n_train]

X_obs = XY$X[(n_train + 1):(n_train + n_obs)]
Y_obs = XY$Y[(n_train + 1):(n_train + n_obs)]

# Predict Y when X is observed
theta = data.frame(y1 = Y_train)
sumstats = data.frame(x1 = X_train)
observed = data.frame(x1 = X_obs)

df_concrete = list(X_train = X_train,
                   Y_train = Y_train,
                   X_obs = X_obs,
                   Y_obs = Y_obs)

## ----save_toy1, echo=F, eval = F----------------------------------------------
# # Save the dataset
# saveRDS(df_concrete, "../inst/extdata/df_concrete.rds")

## ----load_toy1, echo=F--------------------------------------------------------
# Load it back
df_concrete = readRDS("../inst/extdata/df_concrete.rds")

## ----dataset_toy1, echo=F-----------------------------------------------------
X_train = df_concrete$X_train
Y_train = df_concrete$Y_train

X_obs = df_concrete$X_obs
Y_obs = df_concrete$Y_obs

theta = data.frame(y1 = Y_train)
sumstats = data.frame(x1 = X_train)
observed = data.frame(x1 = X_obs)

## ----echo = T-----------------------------------------------------------------
# Init an abcnn object with inputs and targets
abc = abcnn$new(theta,
            sumstats,
            observed,
            method = 'concrete dropout',
            scale_input = "none",
            scale_target = "none",
            num_hidden_layers = 3,
            num_hidden_dim = 128,
            epochs = 20,
            batch_size = 32,
            l2_weight_decay = 1e-5,
            learning_rate = 0.001,
            seed = 6295)

## ----echo = T-----------------------------------------------------------------
abc$summary()

## ----echo = T, eval = F-------------------------------------------------------
# # Use the fit() method to train the neural network
# abc$fit()

## ----echo=F, eval= T----------------------------------------------------------
# save_abcnn(abc, prefix = "../inst/extdata/abc_concrete")

abc = load_abcnn(prefix = "../inst/extdata/abc_concrete")

## ----echo = T-----------------------------------------------------------------
# The torch model
# abc$fitted$model

# The luz fitted model
abc$fitted

# The number of dimensions (i.e. neurons) and layers
abc$num_hidden_dim
abc$num_hidden_layers

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center", fig.cap="The training curve of the neural network across 30 epochs, computed on training data split in three partitions: training, validation (also called testing) and evaluation. Training and validation are computed at the end of each epoch. The black horizontal line is the loss computed on the evaluation dataset at the end of the training procedure."----
abc$plot_training()

## ----echo = T, eval = F-------------------------------------------------------
# save_abcnn(abc, prefix = "../path/abc_concrete")
# 
# abc = load_abcnn(prefix = "../path/abc_concrete")

## ----toy_data_2, echo = F, eval = F-------------------------------------------
# n_crossval = 1000 # Validation size
# 
# crossval = gen_data_1d(n_crossval)
# 
# theta_crossval = data.frame(y1 = crossval$Y)
# sumstats_crossval = data.frame(x1 = crossval$X)

## ----echo = T, eval = F-------------------------------------------------------
# abc$cross_validation(theta_crossval,
#                      sumstats_crossval)

## ----echo = T-----------------------------------------------------------------
abc$cross_validation()

## ----echo = T-----------------------------------------------------------------
abc$plot_cross_validation()

## ----echo = T, eval = F-------------------------------------------------------
# abc$predict()

## ----echo=F, eval=F-----------------------------------------------------------
# save_abcnn(abc, prefix = "../inst/extdata/abc_concrete")

## ----echo = T, message=F------------------------------------------------------
head(abc$predictions())

## ----echo = F-----------------------------------------------------------------
df_predicted = abc$predictions()

df_predicted$uncertainty_a_upper = df_predicted$predictive_mean + df_predicted$aleatoric_uncertainty
df_predicted$uncertainty_a_lower = df_predicted$predictive_mean - df_predicted$aleatoric_uncertainty

df_predicted$uncertainty_e_upper = df_predicted$predictive_mean + df_predicted$epistemic_uncertainty
df_predicted$uncertainty_e_lower = df_predicted$predictive_mean - df_predicted$epistemic_uncertainty

df_predicted$ci_conformal_upper = df_predicted$predictive_mean + df_predicted$overall_conformal_credible_interval
df_predicted$ci_conformal_lower = df_predicted$predictive_mean - df_predicted$overall_conformal_credible_interval

df_predicted$ci_conformal_e_upper = df_predicted$predictive_mean + df_predicted$epistemic_conformal_credible_interval
df_predicted$ci_conformal_e_lower = df_predicted$predictive_mean - df_predicted$epistemic_conformal_credible_interval

df_predicted$x = X_obs
df_predicted$y_true = Y_obs

df_training = data.frame(x = X_train,
                         y = Y_train)

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center", fig.cap="Predictions as a function of simulated parameter. The purple ribbon is the Conformal Credible Interval based on the epistemic unvertainty alone. The green ribbon is the Conformal Credible Interval based on the overall unvertainty."----
ggplot(data = df_training[1:1000,], aes(x = x, y = y)) +
  geom_point(color = "blue", alpha = 0.3) +
  # geom_point(data = df_predicted, aes(x = x, y = y_true), color = "green", alpha = 0.3) +
  geom_line(data = df_predicted, aes(x = x, y = predictive_mean), color = "Red") +
  geom_point(data = df_predicted, aes(x = x, y = predictive_mean), color = "Red") +
  facet_wrap(~ parameter, scales = "free") +
  geom_ribbon(data = df_predicted, aes(x = x, y = predictive_mean, ymin = ci_conformal_e_upper, ymax = ci_conformal_e_lower), alpha = 0.4, fill = "purple") +
  geom_ribbon(data = df_predicted, aes(x = x, y = predictive_mean, ymin = ci_conformal_upper, ymax = ci_conformal_lower), alpha = 0.3, fill = "green") +
  theme_bw()

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center", fig.cap="Predictions as a function of simulated parameters. The epistemic uncertainty (red) and aleatoric uncertainty (blue) were estimated with Concrete Dropout (Gal et 2017)."----
ggplot(data = df_training[1:1000,], aes(x = x, y = y)) +
  geom_point(color = "grey", alpha = 0.3) +
  # geom_point(data = df_predicted, aes(x = x, y = y_true), color = "green", alpha = 0.3) +
  geom_line(data = df_predicted, aes(x = x, y = predictive_mean), color = "Red") +
  geom_point(data = df_predicted, aes(x = x, y = predictive_mean), color = "Red") +
  facet_wrap(~ parameter, scales = "free") +
  geom_ribbon(data = df_predicted, aes(x = x, y = predictive_mean, ymin = uncertainty_a_lower, ymax = uncertainty_a_upper), alpha = 0.3, fill = "blue") +
  geom_ribbon(data = df_predicted, aes(x = x, y = predictive_mean, ymin = uncertainty_e_lower, ymax = uncertainty_e_upper), alpha = 0.3, fill = "red") +
  theme_bw()

## ----concrete_prediction, echo = T, fig.height = 4, fig.width = 8, fig.cap="Prediction results with uncertainty quantification. Red line shows mean predictions, blue ribbon indicates aleatoric uncertainty, and red ribbon shows epistemic uncertainty."----
abc$plot_prediction(uncertainty_type = "uncertainty")

## ----concrete_conformal, echo = T, fig.height = 4, fig.width = 8, fig.cap="Conformal credible intervals (green) compared to epistemic uncertainty intervals (purple). Conformal intervals provide calibrated coverage guarantees."----
abc$plot_prediction(uncertainty_type = "conformal")

## ----echo = T-----------------------------------------------------------------
abc$plot_cross_validation()

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center"--------------
abc$plot_prediction(uncertainty_type = "uncertainty", plot_type = "errorbar")

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center"--------------
abc$plot_posterior(sample = 700, prior = TRUE, uncertainty_type = "uncertainty") +
  geom_vline(xintercept = Y_obs[700], color = "red", size = 1.5)
Y_obs[700]
abc$predictive_mean$y1[700]

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center", fig.cap="Distribution of approximate posterior estimates with predictive means and credible intervals. The prior distribution is plotted underneath (white bars) to compare priors and posteriors."----
abc$plot_posterior(sample = 501, prior = TRUE, uncertainty_type = "conformal") +
  geom_vline(xintercept = Y_obs[501], color = "red", size = 1.5)
Y_obs[501]
abc$predictive_mean$y1[501]

## ----echo = T, eval = F-------------------------------------------------------
# # Parameters of simulated input x
# data_range = 7
# data_step = 0.0005
# 
# # Boundaries of the gap in the data range
# bound1 = -2
# bound2 = 2
# 
# # Random noise applied on y
# data_sigma1a = 0.1
# data_sigma2a = 0.5
# 
# data_sigma1b = 0.2
# data_sigma2b = 0.1
# 
# # Number of simulated data points
# # num_data = 10000
# 
# # Simulate x1
# data_x1a = seq(-data_range, bound1 + data_step, by = data_step)
# data_x1b = seq(bound2, data_range + data_step, by = data_step)
# # Simulate targets y
# data_y1a = sin(data_x1a) + rnorm(length(data_x1a), 0, data_sigma1a)
# data_y1b = sin(data_x1b) + rnorm(length(data_x1b), 0, data_sigma2a)
# 
# # Shift X1 to get X2
# data_x2a = data_x1a + 7
# data_x2b = data_x1b + 7
# # Simulate targets y
# data_y2a = cos(data_x2a) + rnorm(length(data_x2a), 0, data_sigma1b)
# data_y2b = cos(data_x2b) + rnorm(length(data_x2b), 0, data_sigma2b)
# 
# df = data.frame(x1 = c(data_x1a, data_x1b),
#                     x2 = c(data_x2a, data_x2b),
#                     y1 = c(data_y1a, data_y1b),
#                     y2 = c(data_y2a, data_y2b))
# 
# # Shuffle data
# shuffle_idx = sample(1:(nrow(df)), nrow(df), replace = FALSE)
# df_train = df[shuffle_idx,]
# 
# # Train/Test datasets
# # test_ratio = 0.1
# # num_train_data = round(nrow(df) * (1 - test_ratio), digits = 0)
# # num_test_data  = nrow(df) - num_train_data
# #
# # train_x = df[1:num_train_data, c("x1", "x2")]
# # train_y = df[1:num_train_data, c("y1", "y2")]
# # test_x = df[num_train_data:nrow(df_train), c("x1", "x2")]
# # test_y = df[num_train_data:nrow(df_train), c("y1", "y2")]
# train_x = df_train[, c("x1", "x2")]
# train_y = df_train[, c("y1", "y2")]
# 
# 
# # Make a pseudo-obseerved dataset with out of distribution data points
# # Simulate x1
# data_x1 = seq(-data_range, data_range, length.out = 1000)
# # Simulate true targets y
# data_y1 = sin(data_x1)
# 
# # Shift X1 to get X2
# data_x2 = data_x1 + 7
# # Simulate targets y
# data_y2 = cos(data_x2)
# 
# df_observed = data.frame(x1 = data_x1,
#                 x2 = data_x2,
#                 y1 = data_y1,
#                 y2 = data_y2)
# 
# 
# observed_x  = df_observed[, c("x1", "x2")]
# observed_y  = df_observed[, c("y1", "y2")]
# 
# df_deepensemble = list(df_train = df_train,
#                        df_observed = df_observed)

## ----echo = F, eval = F-------------------------------------------------------
# # Save the dataset
# saveRDS(df_deepensemble, "../inst/extdata/df_deepensemble.rds")

## ----echo = F-----------------------------------------------------------------
# Load it back
df_deepensemble = readRDS("../inst/extdata/df_deepensemble.rds")

df_train = df_deepensemble$df_train
df_observed = df_deepensemble$df_observed

train_x = df_train[, c("x1", "x2")]
train_y = df_train[, c("y1", "y2")]
observed_x  = df_observed[, c("x1", "x2")]
observed_y  = df_observed[, c("y1", "y2")]

## ----echo = F, fig.height = 4, fig.width = 8, fig.align="center"--------------
# Plot the simulated data
p1 = ggplot(data = df_train, aes(x = x1, y = y1)) +
  geom_point(color = "Blue", alpha = 0.2) +
  geom_line(data = df_observed, aes(x = x1, y = y1), color = "red") +
  theme_bw()

p2 = ggplot(data = df_train, aes(x = x2, y = y2)) +
  geom_point(color = "Green", alpha = 0.2) +
  geom_line(data = df_observed, aes(x = x2, y = y2), color = "red") +
  theme_bw()

ggpubr::ggarrange(p1, p2, ncol = 2)

## ----echo = T-----------------------------------------------------------------
# devtools::load_all()
# Predict Y when X is observed
theta = train_y
sumstats = train_x
observed = observed_x
true_param = observed_y

abc_ensemble = abcnn$new(theta,
            sumstats,
            observed,
            method = 'deep ensemble',
            num_networks = 5,
            scale_input = "minmax",
            scale_target = "none",
            epochs = 20,
            num_hidden_layers = 3,
            num_hidden_dim = 512,
            batch_size = 128,
            seed = 4722)

## ----echo = T, eval = F-------------------------------------------------------
# abc_ensemble$fit()

## ----echo = F-----------------------------------------------------------------
# save_abcnn(abc_ensemble, prefix = "../inst/extdata/abc_ensemble")

abc_ensemble = load_abcnn(prefix = "../inst/extdata/abc_ensemble")

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center"--------------
abc_ensemble$plot_training()

## ----echo = T, eval = F-------------------------------------------------------
# abc_ensemble$predict()

## ----echo = F, eval = F-------------------------------------------------------
# save_abcnn(abc_ensemble, prefix = "../inst/extdata/abc_ensemble")

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center"--------------
abc_ensemble$plot_prediction(uncertainty_type = "uncertainty")

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center"--------------
abc_ensemble$plot_prediction(uncertainty_type = "conformal")

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center"--------------
# Print a sample with -5 < x1 < -4 (within the distribution with a low noise)
# which(abc$observed < -4 & abc$observed > -5)
abc_ensemble$plot_posterior(sample = 155, prior = TRUE)

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center"--------------
# Print a sample with 4 < x1 < 5 (within the distribution with a high noise)
# which(abc$observed < 5 & abc$observed > 4)
abc_ensemble$plot_posterior(sample = 800, prior = TRUE)

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center"--------------
# Print a sample with -1 < x1 < 1 (out of training distribution)
# which(abc$observed < 1 & abc$observed > -1)
abc_ensemble$plot_posterior(sample = 520, prior = TRUE)

## ----eval=F, echo=F-----------------------------------------------------------
# ########################################################
# #               Simulation of                          #
# #           a Gaussian toy example                     #
# ########################################################
# 
# library(mvtnorm) # for multivariate normal distribution
# library(spatstat) # for weighted.quantile function
# library(MCMCpack) # for the inverse gamma distribution
# 
# #====================================
# # Homoscedastic Normal toy model
# #====================================
# # Sample size of y
# n = 10
# # Inverse gamma parameters
# alpha = 4
# beta = 3
# # Training set sample size
# N = 100000
# # Test set sample size
# p = 1000
# # offset for the out-of-dist data
# offset = 0
# 
# homoscedastic_normal = function(alpha = 4,
#                                 beta = 3,
#                                 n = 10,
#                                 N = 10000,
#                                 p = 1000,
#                                 offset = 0) {
#   # Function to compute quantiles from
#   # student distribution
#   qnst = function(p, deg, loca, scale) {
#     return(loca + scale * qt(p, df = deg))
#   }
# 
#   # Simulation of the ABC reference table
# 
#   set.seed(1) # for reproducibility
# 
#   # X parameter
#   theta1.train = rep(NA, N)
#   theta2.train = 1 / rgamma(N, shape = alpha, rate = beta)
#   for (i in 1:N) {
#     theta1.train[i] = rnorm(1, 0, sqrt(theta2.train[i]))
#   }
# 
#   # Add offset for out of dist
#   theta2.train = theta2.train + offset
#   theta1.train = theta1.train + offset
# 
#   # Y parameter
#   # n sample cols
#   # n training set rows
#   y.ref = matrix(NA, N, n)
#   for (i in 1:N) {
#     y.ref[i, ] = rnorm(n, theta1.train[i], sqrt(theta2.train[i]))
#   }
# 
#   # Compute some summary statistics
#   sumstats.train = matrix(NA, N, 3)
# 
#   for (i in 1:N) {
#     sumstats.train[i, ] = c(mean(y.ref[i, ]), var(y.ref[i, ]), mad(y.ref[i, ]))
#   }
# 
#   ref.training = cbind(theta1.train, theta2.train, sumstats.train)
#   colnames(ref.training) = c("theta1", "theta2", "expectation", "variance", "mad")
# 
#   # Simulation of the ABC test table
#   theta1.test = rep(NA, p)
# 
#   theta2.test = 1 / rgamma(p, shape = alpha, rate = beta)
#   for (i in 1:p) {
#     theta1.test[i] = rnorm(1, 0, sqrt(theta2.test[i]))
#   }
# 
#   theta2.test = theta2.test + offset
#   theta1.test = theta1.test + offset
# 
#   y.test = matrix(NA, p, n)
# 
#   for (i in 1:p) {
#     y.test[i, ] = rnorm(n, theta1.test[i], sqrt(theta2.test[i]))
#   }
# 
#   # Compute some summary statistics
#   sumstats.test = matrix(NA, p, 3)
# 
#   for (i in 1:p) {
#     sumstats.test[i, ] =
#       c(mean(y.test[i, ]), var(y.test[i, ]), mad(y.test[i, ]))
#   }
# 
#   ref.testing = cbind(theta1.test, theta2.test, sumstats.test)
#   colnames(ref.testing) = c("theta1", "theta2", "expectation", "variance", "mad")
# 
# 
#   # Compute the exact posterior expectations, variances and quantiles
#   # for parameters theta1 and theta2
#   theta1.test.exact = rep(NA, p)
#   theta2.test.exact = rep(NA, p)
#   var1.test.exact = rep(NA, p)
#   var2.test.exact = rep(NA, p)
#   quant.theta1.test.freq = matrix(NA, p, 2)
#   quant.theta2.test.freq = matrix(NA, p, 2)
# 
#   for (i in 1:p) {
#     theta1.test.exact[i] = sum(y.test[i, ]) / (n + 1)
#     var1.test.exact[i] =
#       (beta + sum((y.test[i, ] - mean(y.test[i, ])) ^ 2)/2 + n*(mean(y.test[i, ]))^2 / (2*(n+1))  ) /
#       ( (n + 1) * (alpha - 1 + n / 2) )
#     theta2.test.exact[i] =
#       (beta + sum((y.test[i, ] - mean(y.test[i, ])) ^ 2)/2 + n*(mean(y.test[i, ]))^2 / (2*(n+1))  ) / (alpha - 1 + n / 2)
#     var2.test.exact[i] =
#       (beta + sum((y.test[i, ] - mean(y.test[i, ])) ^ 2)/2 + n*(mean(y.test[i, ]))^2 / (2*(n+1))  ) ^ 2 / ((alpha - 1 + n / 2) ^ 2 * (alpha - 2 + n / 2))
#     quant.theta1.test.freq[i, ] =
#       c(qnst(0.025, n + 2 * alpha, sum(y.test[i, ]) / (n + 1), sqrt(2 * (beta + sum((y.test[i, ] - mean(y.test[i, ])) ^ 2)/2 + n*(mean(y.test[i, ]))^2 / (2*(n+1))  ) / ((n + 1) * (n + 2 * alpha) ))),
#         qnst(0.975, n + 2 * alpha, sum(y.test[i, ]) / (n + 1), sqrt(2 * (beta + sum((y.test[i, ] - mean(y.test[i, ])) ^ 2)/2 + n*(mean(y.test[i, ]))^2 / (2*(n+1))  ) / ((n + 1) * (n + 2 * alpha) ))))
#     quant.theta2.test.freq[i, ] =
#       c(1 / qgamma(0.975, shape = (n + 2 * alpha) / 2, rate = (beta + sum((y.test[i, ] - mean(y.test[i, ])) ^ 2)/2 + n*(mean(y.test[i, ]))^2 / (2*(n+1))  ) ),
#         1 / qgamma(0.025, shape = (n + 2 * alpha) / 2, rate = (beta + sum((y.test[i, ] - mean(y.test[i, ])) ^ 2)/2 + n*(mean(y.test[i, ]))^2 / (2*(n+1))  )) )
#   }
# 
#   test.exact = data.frame(mean.theta1 = theta1.test.exact,
#                           var.theta1 = var1.test.exact,
#                           lower.theta1 = quant.theta1.test.freq[,1],
#                           upper.theta1 = quant.theta1.test.freq[,2],
#                           mean.theta2 = theta2.test.exact,
#                           var.theta2 = var2.test.exact,
#                           lower.theta2 = quant.theta2.test.freq[,1],
#                           upper.theta2 = quant.theta2.test.freq[,2])
# 
#   # Add noise to summary statistics simulated according
#   # to a uniform(0,1) distribution
#   nNoise = 50 # or 500
# 
#   set.seed(3)  # for reproducibility
# 
#   sumstats.noise = matrix(runif((N+p) * nNoise), N+p, nNoise)
#   ref.training = cbind(ref.training, sumstats.noise[1:N, ])
#   ref.testing = cbind(ref.testing, sumstats.noise[(N+1):(N+p), ])
# 
#   colnames(ref.training) =
#     c("theta1", "theta2", "expectation", "variance", "mad", c(1:nNoise))
#   colnames(ref.testing) =
#     c("theta1", "theta2", "expectation", "variance", "mad", c(1:nNoise))
# 
#   # Add some others summary statistics
#   y = ref.training[, 1:2]
#   x = ref.training[, -c(1:2)]
# 
#   x =cbind(x, x[, 1] + x[, 2], x[, 1] + x[, 3], x[, 2] + x[, 3], x[, 1] + x[, 2] +
#              x[, 3], x[, 1] * x[, 2], x[, 1] * x[, 3], x[, 2] * x[, 3], x[, 1] * x[, 2] *
#              x[, 3])
#   colnames(x) =
#     c("expectation",
#       "variance",
#       "mad",
#       c(1:nNoise),
#       "sum_esp_var",
#       "sum_esp_mad" ,
#       "sum_var_mad",
#       "sum_esp_var_mad",
#       "prod_esp_var",
#       "prod_esp_mad",
#       "prod_var_mad" ,
#       "prod_esp_var_mad")
# 
#   ytest = ref.testing[, 1:2]
#   xtest = ref.testing[, -c(1:2)]
# 
#   xtest =
#     cbind(
#       xtest,
#       xtest[, 1] + xtest[, 2],
#       xtest[, 1] + xtest[, 3],
#       xtest[, 2] + xtest[, 3],
#       xtest[, 1] + xtest[, 2] + xtest[, 3],
#       xtest[, 1] * xtest[, 2],
#       xtest[, 1] * xtest[, 3],
#       xtest[, 2] * xtest[, 3],
#       xtest[, 1] * xtest[, 2] * xtest[, 3]
#     )
#   colnames(xtest) =
#     c("expectation",
#       "variance",
#       "mad",
#       c(1:nNoise),
#       "sum_esp_var",
#       "sum_esp_mad" ,
#       "sum_var_mad",
#       "sum_esp_var_mad",
#       "prod_esp_var",
#       "prod_esp_mad",
#       "prod_var_mad" ,
#       "prod_esp_var_mad")
# 
#   data.theta1 = data.frame(theta1 = y[,1], x)
#   data.theta2 = data.frame(theta2 = y[,2], x)
# 
#   param.Test = data.frame(ytest)
# 
#   colnames(param.Test) = c("theta1", "theta2")
# 
#   stats.Test = data.frame(xtest)
# 
#   colnames(stats.Test) = colnames(x)
#   colnames(param.Test) = colnames(y)
# 
#   return(list(x.train = x,
#               y.train = y,
#               x.test = stats.Test,
#               y.test = param.Test,
#               y.exact = test.exact))
# }
# 
# 
# dataset = homoscedastic_normal(alpha, beta, n, N, p, offset)
# 
# saveRDS(dataset, "../inst/extdata/normal_toy_model.Rds")

## ----echo = F-----------------------------------------------------------------
dataset = readRDS("../inst/extdata/normal_toy_model.Rds")

## ----echo = T-----------------------------------------------------------------
# For training
sumstats.train = as.data.frame(dataset$x.train)
theta.train = as.data.frame(dataset$y.train)

# Testing
sumstats.test = dataset$x.test
theta.test = dataset$y.test

# The exact value to find
theta.exact = dataset$y.exact

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center"--------------
ggplot(theta.train, aes(x = theta1, y = theta2)) +
  geom_point() +
  geom_point(data = theta.test, aes(x = theta1, y = theta2), color = "red", alpha = 0.5)


## -----------------------------------------------------------------------------
deepensemble_highdim = abcnn$new(theta.train,
            sumstats.train,
            sumstats.test[1:1000,],
            method = 'deep ensemble',
            scale_input = "minmax",
            scale_target = "minmax",
            num_hidden_layers = 3,
            num_hidden_dim = 256,
            epochs = 20,
            batch_size = 128,
            l2_weight_decay = 1e-4,
            epsilon_adversarial = 0.001,
            seed = 42)


deepensemble_highdim$summary()

## ----eval = F-----------------------------------------------------------------
# deepensemble_highdim$fit()

## ----echo = F, eval=T---------------------------------------------------------
deepensemble_highdim = load_abcnn(prefix = "../inst/extdata/deepensemble_highdim")

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center"--------------
deepensemble_highdim$plot_training()

## ----echo = F, eval=F---------------------------------------------------------
# save_abcnn(deepensemble_highdim, prefix = "../inst/extdata/deepensemble_highdim")

## ----echo = T, eval = T-------------------------------------------------------
exp = explainn$new(deepensemble_highdim)

exp$run(data = sumstats.test[1:1000,],
        method = "deeplift")

## ----explainn_plot, echo = T, fig.height = 4, fig.width = 8, fig.cap="Feature importance visualization using DeepLIFT attribution methods, for the first sample output prediction. Colors indicate the contribution of each feature to the final prediction."----
exp$plot()

## ----echo = T, fig.height = 4, fig.width = 8, fig.cap="Feature importance visualization using DeepLIFT attribution methods, summarized across the 1,000 smaples. Colors indicate the contribution of each feature to the final prediction."----
exp$plot_global()

## ----echo = T-----------------------------------------------------------------
# Three dimension outpout:
# - First sample
# - 10 first summary statistics
# - the two output parameters
exp$get_result()[1,1:10,1:2]

## ----echo = T, eval = T-------------------------------------------------------
exp = explainn$new(deepensemble_highdim)

exp$run(data = sumstats.test[1:1000,],
        method = "smoothgrad")

## ----echo = T, fig.height = 4, fig.width = 8, fig.cap="Feature importance visualization using DeepLIFT attribution methods. Colors indicate the contribution of each feature to the final prediction."----
exp$plot()

## ----echo = T, fig.height = 4, fig.width = 8, fig.cap="Feature importance visualization using DeepLIFT attribution methods. Colors indicate the contribution of each feature to the final prediction."----
exp$plot_global()

## ----echo = T-----------------------------------------------------------------
exp$get_result()[1:5,1:10,1]

## -----------------------------------------------------------------------------
tabnetabc = abcnn$new(theta.train[1:10000,],
            sumstats.train[1:10000,],
            sumstats.test[1:1000,],
            method = 'tabnet-abc',
            scale_input = "none",
            scale_target = "none",
            epochs = 30,
            batch_size = 64,
            tol = 0.1,
            abc_keep_original_sumstats = 10,
            abc_method = "loclinear",
            seed = 4567)


tabnetabc$summary()

## ----eval = F-----------------------------------------------------------------
# tabnetabc$fit()

## ----echo = F, eval=T---------------------------------------------------------
tabnetabc = load_abcnn(prefix = "../inst/extdata/tabnetabc")

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center"--------------
tabnetabc$plot_training()

## ----eval = F, message=FALSE, results = FALSE, include=FALSE------------------
# tabnetabc$predict()

## ----echo = F, eval=F---------------------------------------------------------
# save_abcnn(tabnetabc, prefix = "../inst/extdata/tabnetabc")

## ----tabnet_performance, echo = T, fig.height = 4, fig.width = 8, fig.cap="TabNet-ABC performance showing posterior quantile predictions compared to true parameter values."----
tabnetabc$plot_prediction(uncertainty_type = "posterior quantile",
                          plot_type = "errorbar")

## ----tabnet_accuracy, echo = T, fig.height = 4, fig.width = 8, fig.cap="Scatter plot comparing TabNet-ABC predictions to exact posterior means. The shaded region represents 95% credible intervals."----
df = tabnetabc$predictions() %>%
  filter(parameter == "theta1")

df$true.theta = theta.exact[1:1000,"mean.theta1"]

ggplot(df, aes(x = true.theta, y = predictive_mean)) +
  geom_ribbon(aes(ymin = posterior_lower_ci, ymax = posterior_upper_ci), 
              fill = "lightgrey", alpha = 0.5) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(x = "True Posterior Mean", y = "Predicted Mean",
       title = "TabNet-ABC Predictive check") +
  theme_bw()

## ----echo = T, eval = T-------------------------------------------------------
exp = explainn$new(tabnetabc)

exp$run(data = sumstats.test)

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center"--------------
exp$plot()

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center"--------------
exp$plot(type = "mask_agg")

## ----echo = T, fig.height = 4, fig.width = 8, fig.align="center"--------------
exp$plot(type = "steps")

## ----echo=F-------------------------------------------------------------------
hyperparam = data.frame(Hyperparameter = c("Number of hidden dimensions (i.e. neurons) in one layer",
                                           "Number of hidden layers",
                                           "Epochs",
                                           "Batch size",
                                           "Learning rate",
                                           "L2 weight decay",
                                           "Dropout (only for Monte Carlo Dropout)"),
                        Values = c("100-500 is generally a good range. You can try higher values if you have more than 200 summary statistics.",
                                   "3-6, deeper networks tend to be harder to train",
                                   "30-60 are generally enough to avoid overfitting",
                                   "32-256, depending mostly on the variance in your training set. Higher batch sizes have a smoothing effect on excess of variance during training.",
                                   "1e-3 is generally fine for 10,000-100,000 training samples. 1e-4 tend to be better for sample sizes >= 1,000,000.",
                                   "1e-5, minimal impact.",
                                   "0.1-0.5. Smaller values should produce smaller Credible Intervals, but at the risk of less precise or biased estimates. Higher values are more conservative."))

  
knitr::kable(hyperparam, caption = "Hyperparameters available in abcnn and standard range of values.")

## ----session_info-------------------------------------------------------------
sessionInfo()


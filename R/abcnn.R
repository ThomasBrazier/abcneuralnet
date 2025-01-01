#' Create an `abcnn` R6 class object
#'
#' @param observed a vector of summary statistics computed on the data
#' @param theta a vector, matrix or data frame of the simulated theta_parameter values
#' @param sumstat a vector, matrix or data frame of the simulated summary statistics
#' @param method either `monte carlo dropout`, `concrete dropout` or `ensemble`, See details
#' @param scale whether to scale summary statistics before training
#' @param num_hidden_layers the number of hidden layers in the Neural Network
#' @param num_hidden_dim the dimension of hidden layers (i.e. number of neurons) in each layer of the Neural Network
#' @param model the type of Neural Network to use if you want a custom model. See `details`
#' @param dropout the dropout rate for `MC dropout`, i.e. the proportion of neurons dropped (must be between 0.1 and 0.5)
#' @param validation_split the proportion of samples retained for validation
#' @param test_split the proportion of samples retained for evaluation
#' @param batch_size the batch size
#' @param learning_rate the learning rate
#' @param epochs the number of epochs
#' @param early_stopping boolean, whether to use early stopping or not
#' @param patience the patience (number of iterations) before early stopping
#' @param optimizer
#' @param loss
#' @param test_metrics
#' @param tol the tolerance rate in ABC sampling, i.e. the proportion of simulated theta retained to approximate the posterior. Set NULL if you want to keep all simulations for training
#' @param num_posterior_samples the number of Approximate posterior samples to predict with the `monte carlo dropout` method
#' @param kernel the kernel function, either `rbf` or `epanechnikov`
#' @param sampling the ABC sampling function, either `rejection` or `importance`
#' @param length_scale the length scale parameter of the `rbf` kernel
#' @param bandwith the bandwith hyperparameter for the kernel smoothing function, either a numeric value or set to "max" if the bandwidth must be estimated as the maximum distance value
#' @param prior_length_scale the prior length scale hyperparameter for the concrete dropout method
#' @param num_val the number of duplicate observed sample to provide for each iteration of `concrete dropout` approximate posterior prediction
#' @param num_net the number of iterations in the `regression adjustment` approach
#' @param epsilon_adversarial The multiplying factor for perturbed inputs in adversarial training (Deep Ensemble). Set NULL to disable adverarial training (default) or 0.01 is a good value to start with.
#'
#' @import torch
#' @import luz
#' @import ggplot2
#' @import tidyr
#' @import R6Class
#'
#' @return an `abcnn` object
#' @export
abcnn = R6::R6Class("abcnn",
  public = list(
    theta = NULL,
    sumstat = NULL,
    observed = NULL,
    model=NULL,
    method='concrete dropout',
    scale=FALSE,
    num_hidden_layers=3,
    num_hidden_dim=128,
    validation_split=0.1,
    test_split=0.1,
    num_conformal=0,
    dropout=0.5,
    dropout_input=0,
    batch_size=32,
    epochs=20,
    early_stopping=TRUE,
    callbacks=NULL,
    patience=4,
    optimizer=NULL,
    learning_rate=0.001,
    l2_weight_decay=1e-5,
    loss=NULL,
    tol=NULL,
    num_posterior_samples=1000,
    kernel='rbf',
    sampling='importance',
    length_scale=1.0,
    bandwith=1.0,
    prior_length_scale=1e-4,
    wr=NA,
    dr=NA,
    num_val=100,
    num_networks=5,
    num_iter=10000,
    num_print=1000,
    epsilon_adversarial=NULL,
    inference=TRUE,
    device="cpu",
    input_dim = NA,
    output_dim = NA,
    n_train = NA,
    sumstat_names = NA,
    output_names = NA,
    n_obs = NA,
    prior_lower = NA,
    prior_upper = NA,
    fitted = NULL,
    evaluation=NULL,
    eval_metrics=NA,
    posterior_samples = NA,
    predictive_mean = NA,
    CI_aleatoric_lower = NA,
    CI_aleatoric_upper = NA,
    CI_epistemic_lower = NA,
    CI_epistemic_upper = NA,
    CI_overall_lower = NA,
    CI_overall_upper = NA,
    epistemic_uncertainty = NA,
    aleatoric_uncertainty = NA,
    dropout_rates = NA,
    sumstat_mean = NA,
    sumstat_sd = NA,

    initialize = function(theta,
                          sumstat,
                          observed,
                          model=NULL,
                          method='concrete dropout',
                          scale=FALSE,
                          num_hidden_layers=3,
                          num_hidden_dim=128,
                          validation_split=0.1,
                          test_split=0.1,
                          num_conformal=0,
                          dropout=0.5,
                          dropout_input=0,
                          batch_size=32,
                          epochs=20,
                          early_stopping=TRUE,
                          patience=4,
                          optimizer=optim_adam,
                          learning_rate=0.001,
                          l2_weight_decay=1e-5,
                          loss=nn_mse_loss,
                          tol=NULL,
                          num_posterior_samples=1000,
                          kernel='rbf',
                          sampling='importance',
                          length_scale=1.0,
                          bandwith=1.0,
                          prior_length_scale=1e-4,
                          num_val=100,
                          num_networks=5,
                          num_iter=10000,
                          num_print=1000,
                          epsilon_adversarial=NULL,
                          inference=TRUE,
                          device="cpu") {
      #-----------------------------------
      # CHECK INPUTS
      #-----------------------------------
      # Input data validation
      if(missing(observed)) stop("'observed' is missing")
      if(missing(theta)) stop("'theta' is missing")
      if(missing(sumstat)) stop("'sumstat' is missing")
      if(!is.data.frame(theta)) stop("'theta' has to be a data.frame with column names.")
      if(!is.data.frame(sumstat)) stop("'sumstat' has to be a data.frame with column names.")
      if(!is.data.frame(observed)) stop("'observed' has to be a data.frame with column names.")
      if(dropout < 0.1 | dropout > 0.5) stop("The 'dropout' rate must be between 0.1 and 0.5.")

      # Device. Use CUDA if available
      self$device = torch_device(if (cuda_is_available()) {"cuda"} else {"cpu"})
      # "When fitting, luz will use the fastest possible accelerator;
      # if a CUDA-capable GPU is available it will be used, otherwise we fall back to the CPU.
      # It also automatically moves data, optimizers,
      # and models to the selected device so you don’t need to handle it manually."

      # TODO Check input dim and type
      # Coerce data to array
      self$observed = as.data.frame(observed)
      self$theta = as.data.frame(theta)
      self$sumstat = as.data.frame(sumstat)

      # TODO Message to user: num params, num observation, training/test/validation sizes
      # TODO Print dimensions and model

      #-----------------------------------
      # INIT ATTRIBUTES
      #-----------------------------------
      self$method=method
      self$scale=scale
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
      self$num_posterior_samples=num_posterior_samples
      self$kernel=kernel
      self$sampling=sampling
      self$length_scale=length_scale
      self$bandwith=bandwith
      self$l2_weight_decay
      self$epsilon_adversarial = epsilon_adversarial

      self$input_dim = ncol(sumstat)
      self$output_dim = ncol(theta)

      self$n_train = nrow(theta)
      self$sumstat_names = colnames(observed)
      colnames(self$sumstat) = self$sumstat_names
      self$n_obs = nrow(self$observed)
      self$output_names = NA

      # Keep metadata of prior boundaries for plotting
      self$prior_lower = apply(as.matrix(self$theta), 1, min)
      self$prior_upper = apply(as.matrix(self$theta), 1, max)

      # Init output slots
      self$fitted = NULL
      self$posterior_samples = NA # Raw posterior samples predicted by the NN
      self$predictive_mean = NA # The mean predicted from MC samples (MC dropout) or pointwise estimate by concrete dropout
      # 95% confidence intervals based on the estimated epist. and aleat. uncertainty in concrete dropout
      # = 2*sd
      self$CI_aleatoric_lower = NA
      self$CI_aleatoric_upper = NA
      self$CI_epistemic_lower = NA
      self$CI_epistemic_upper = NA
      self$CI_overall_lower = NA
      self$CI_overall_upper = NA
      self$epistemic_uncertainty = NA # Raw epistemic uncertainty, expressed as sd
      self$aleatoric_uncertainty = NA # Raw aleatoric uncertainty, expressed as sd

      self$dropout_rates = NA # The dropout rates hyperparameter estimated by concrete dropout

      #-----------------------------------
      # CALLBACKS
      #-----------------------------------
      self$patience = patience
      if (early_stopping) {
        early_stopping_callback = luz_callback_early_stopping(patience = self$patience)
        self$callbacks = list(early_stopping_callback)
      } else {
        self$callbacks = list()
      }

      #-----------------------------------
      # PRE-PROCESSING
      #-----------------------------------
      # Scale summary statistics (optional)
      # Save mean and sd of training data to apply on observed data at prediction time
      self$sumstat_mean = apply(self$sumstat, 2, function(x) {mean(x, na.rm = TRUE)})
      self$sumstat_sd = apply(self$sumstat, 2, function(x) {sd(x, na.rm = TRUE)})
      if (scale) {
        self$sumstat = (self$sumstat - self$sumstat_mean) / self$sumstat_sd
        self$observed = (self$observed - self$sumstat_mean) / self$sumstat_sd
      }

    },

    # Train the neural network
    fit = function() {
      if (self$method == 'monte carlo dropout') {
        # Load data
        dl = self$dataloader()

        # Fit
        model = mc_dropout_model %>%
          setup(optimizer = self$optimizer,
                loss = self$loss) %>%
          set_hparams(num_input_dim = self$input_dim,
                      num_hidden_dim = self$num_hidden_dim,
                      num_output_dim = self$output_dim,
                      num_hidden_layers = self$num_hidden_layers,
                      dropout_hidden = self$dropout,
                      dropout_input = self$dropout_input) %>%
          set_opt_hparams(lr = self$learning_rate, weight_decay = self$l2_weight_decay)

        self$fitted = model %>%
          fit(dl$train,
              epochs = self$epochs,
              valid_data = dl$valid,
              callbacks = self$callbacks)
        self$model = self$fitted$model

      }

      if (self$method == 'concrete dropout') {
        # Load data
        dl = self$dataloader()

        # Fit

        # TODO Make utility functions for wr and dr
        l = self$prior_length_scale
        N = self$n_train
        self$wr = l^2 / N
        self$dr = 2 / N

        # print(self$optimizer)
        # print(self$loss)

        # print(dl$train)

        # iter = dl$train$.iter()
        # print(iter$.next())

        # print(self$callbacks)

        model = concrete_model %>%
          setup(optimizer = self$optimizer) %>%
          set_hparams(num_input_dim = self$input_dim,
                      num_hidden_dim = self$num_hidden_dim,
                      num_output_dim = self$output_dim,
                      num_hidden_layers = self$num_hidden_layers,
                      weight_regularizer = self$wr,
                      dropout_regularizer = self$dr,
                      dropout_input = self$dropout_input) %>%
          set_opt_hparams(lr = self$learning_rate, weight_decay = self$l2_weight_decay)

        self$fitted = model %>%
          fit(dl$train,
              epochs = self$epochs,
              valid_data = dl$valid,
              callbacks = self$callbacks)
        self$model = self$fitted$model
      }

      if (self$method == 'deep ensemble') {
        # Load data
        dl = self$dataloader()

        # The range of noise to add to perturbed inputs in adversarial training
        if (!is.null(self$epsilon_adversarial)) {
          epsilon = self$epsilon_adversarial * 2 * self$sumstat_sd
        } else {
          epsilon = NULL
        }

        # Fit
        model = nn_ensemble %>%
          setup() %>%
          set_hparams (model = single_model,
                       learning_rate = self$learning_rate,
                       weight_decay = self$l2_weight_decay,
                       num_models = self$num_networks,
                       num_input_dim = self$input_dim,
                       num_output_dim = self$output_dim,
                       num_hidden_layers = self$num_hidden_layers,
                       num_hidden_dim = self$num_hidden_dim,
                       epsilon = epsilon)

        self$fitted = model %>%
          fit(dl$train,
              epochs = self$epochs,
              valid_data = dl$valid)
        self$model = self$fitted$model
      }

      # Evaluation
      # Plot training
      plot(self$fitted)

      # Monitor loss and metrics
      metrics = get_metrics(self$fitted)
      print(metrics)

      # TODO luz::evaluate currently not working with Deep Ensemble
      # fitted returns n values named value.x instead of a single value
      if (self$method != "deep ensemble") {
        self$evaluation = self$fitted %>% luz::evaluate(data = dl$test)
        self$eval_metrics = get_metrics(self$evaluation)
        print(self$eval_metrics)
        print(self$evaluation)
      }
    },

    # Predict parameters from a vector/array of observed summary statistics
    predict = function() {

      observed = torch_tensor(as.matrix(self$observed))


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
          epistemic_uncertainty = apply(means, 3, function(x) apply(x, 2, var))
        } else {
          predictive_mean = apply(means, 2, mean)
          epistemic_uncertainty = apply(means, 2, var)
        }

        predictive_mean = as.data.frame(array(predictive_mean, dim = c(observed$shape[1], self$output_dim)))
        colnames(predictive_mean) = colnames(self$theta)
        self$predictive_mean = predictive_mean

        epistemic_uncertainty = as.data.frame(array(epistemic_uncertainty, dim = c(observed$shape[1], self$output_dim)))
        colnames(epistemic_uncertainty) = colnames(self$theta)
        self$epistemic_uncertainty = epistemic_uncertainty

        self$CI_epistemic_lower = predictive_mean - 2 * sqrt(epistemic_uncertainty)
        self$CI_epistemic_upper = predictive_mean + 2 * sqrt(epistemic_uncertainty)

        # Only the epistemic error is estimated
        self$CI_overall_lower = CI_epistemic_lower
        self$CI_overall_upper = CI_epistemic_upper
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
        } else {
          predictive_mean = apply(means, 2, mean)
          epistemic_uncertainty = apply(means, 2, var)
          aleatoric_uncertainty = exp(colMeans(logvar))
        }
        predictive_mean = as.data.frame(array(predictive_mean, dim = c(observed$shape[1], self$output_dim)))
        colnames(predictive_mean) = colnames(self$theta)
        self$predictive_mean = predictive_mean

        epistemic_uncertainty = as.data.frame(array(epistemic_uncertainty, dim = c(observed$shape[1], self$output_dim)))
        colnames(epistemic_uncertainty) = colnames(self$theta)
        self$epistemic_uncertainty = epistemic_uncertainty

        aleatoric_uncertainty = as.data.frame(array(aleatoric_uncertainty, dim = c(observed$shape[1], self$output_dim)))
        colnames(aleatoric_uncertainty) = colnames(self$theta)
        self$aleatoric_uncertainty = aleatoric_uncertainty

        self$CI_aleatoric_lower = self$predictive_mean -
          2 * sqrt(self$aleatoric_uncertainty)
        self$CI_aleatoric_upper = self$predictive_mean +
          2 * sqrt(self$aleatoric_uncertainty)

        self$CI_epistemic_lower = self$predictive_mean -
          2 * sqrt(self$epistemic_uncertainty)
        self$CI_epistemic_upper = self$predictive_mean +
          2 * sqrt(self$epistemic_uncertainty)

        self$CI_overall_lower = self$predictive_mean -
          2 * sqrt(self$epistemic_uncertainty) -
          2 * sqrt(self$aleatoric_uncertainty)
        self$CI_overall_upper = self$predictive_mean +
          2 * sqrt(self$epistemic_uncertainty) +
          2 * sqrt(self$aleatoric_uncertainty)

        # Store dropout rates inferred
        params = self$fitted$model$named_parameters()
        p_logit = params[grepl("p_logit", names(params))]
        p = lapply(p_logit, function(x) torch_sigmoid(x))
        p = unlist(lapply(p, function(x) as.numeric(x)))
        self$dropout_rates = p
      }

      if (self$method == 'deep ensemble') {
        # Use forward to get mean prediction + variance

        print("Predictions")

        # Infer epistemic + aleatoric uncertainty
        out_mu_sample  = torch_zeros(c(self$n_obs, self$output_dim, self$num_networks))
        out_sig_sample = torch_zeros(c(self$n_obs, self$output_dim, self$num_networks))

        for (i in 1:self$num_networks) {
          pb = txtProgressBar(min = 1, max = self$num_networks, style = 3)

          print(paste("Network", i))

          preds = self$fitted$model$model_list[[i]](observed)

          print(preds)
          mu_sample = preds[1,,]
          print(mu_sample)

          sig_sample = preds[2,,]
          # sig_sample = torch_logsumexp(sig_sample, 1, keepdim = TRUE) + 1e-6
          sig_sample = log1pexp(sig_sample)
          print(sig_sample)

          out_mu_sample[,,i]  = mu_sample
          out_sig_sample[,,i] = sig_sample

          setTxtProgressBar(pb, i)
        }
        close(pb)

        print("Compute predictive mean")

        self$predictive_mean  = torch_mean(out_mu_sample, dim = 3)

        print(self$predictive_mean)

        print("Compute uncertainty")

        # out_sig_sample_final = torch_sqrt(torch_mean(out_sig_sample, dim = 3) + torch_mean(torch_square(out_mu_sample), dim = 3)
        #                                   - torch_square(out_mu_sample_final))
        self$aleatoric_uncertainty = torch_sqrt(torch_mean(out_sig_sample, dim = 3))
        self$epistemic_uncertainty = torch_sqrt(torch_mean(torch_square(out_mu_sample), dim = 3) - torch_square(self$predictive_mean))

        self$CI_aleatoric_lower = self$predictive_mean -
          2 * sqrt(self$aleatoric_uncertainty)
        self$CI_aleatoric_upper = self$predictive_mean +
          2 * sqrt(self$aleatoric_uncertainty)

        self$CI_epistemic_lower = self$predictive_mean -
          2 * sqrt(self$epistemic_uncertainty)
        self$CI_epistemic_upper = self$predictive_mean +
          2 * sqrt(self$epistemic_uncertainty)

        self$CI_overall_lower = self$predictive_mean -
          2 * sqrt(self$epistemic_uncertainty) -
          2 * sqrt(self$aleatoric_uncertainty)
        self$CI_overall_upper = self$predictive_mean +
          2 * sqrt(self$epistemic_uncertainty) +
          2 * sqrt(self$aleatoric_uncertainty)
      }

    },

    # Prepare the torch dataloader from sumstat/theta (input/target)
    # Return a dataloader object
    dataloader = function() {

      if (!is.null(self$tol)) {
        abc = self$abc_sampling()
        theta = as.matrix(abc$theta)
        sumstat = as.matrix(abc$sumstat)
      } else {
        theta = as.matrix(self$theta)
        sumstat = as.matrix(self$sumstat)
      }

      if (self$method == 'monte carlo dropout' | self$method == 'concrete dropout') {
        # Make Torch dataloader
        # train-validation-test sets

        # Randomly sample indexes
        n_val = round(self$n_train * self$validation_split, digits=0)
        n_test = round(self$n_train * self$test_split, digits=0)
        n_train = self$n_train - n_val -n_test - self$num_conformal

        random_idx = sample(1:nrow(theta), replace = FALSE)
        train_idx = random_idx[1:n_train]
        valid_idx = random_idx[(n_train + 1):(n_train + n_val)]
        test_idx = random_idx[(n_train + n_val + 1):(n_train + n_val + n_test)]
        # conformal_idx = random_idx[(n_train + n_val + n_test + 1):(n_train + n_val + n_test + num_conformal)]

        sumstat_tensor = torch_tensor(as.matrix(sumstat))
        theta_tensor = torch_tensor(as.matrix(theta))
        # sumstat_tensor = sumstat_tensor$unsqueeze(length(dim(sumstat_tensor)) + 1)
        # theta_tensor = theta_tensor$unsqueeze(length(dim(theta_tensor)) + 1)

        ds = tensor_dataset(sumstat_tensor, theta_tensor)

        # Make train-val-test tensors (for Ensemble method)

        # Data loader (MC dropout and Concrete dropout)
        train_ds = dataset_subset(ds, train_idx)
        train_dl = dataloader(train_ds, batch_size = self$batch_size, shuffle = TRUE, drop_last = TRUE)

        valid_ds = dataset_subset(ds, valid_idx)
        valid_dl = dataloader(valid_ds, batch_size = self$batch_size, shuffle = TRUE, drop_last = TRUE)

        test_ds = dataset_subset(ds, test_idx)
        test_dl = dataloader(test_ds, batch_size = self$batch_size, shuffle = TRUE, drop_last = TRUE)

        # conformal_ds = dataset_subset(ds, conformal_idx)
        # conformal_dl = dataloader(conformal_ds, batch_size = batch_size, shuffle = TRUE, drop_last = TRUE)

        # observed = torch_tensor(as.matrix(observed))
        # observed = observed$unsqueeze(length(dim(observed)) + 1)

        return(list(train = train_dl, valid = valid_dl, test = test_dl))
      }

      if (self$method == 'deep ensemble') {
        # Randomly sample indexes
        n_val = round(self$n_train * self$validation_split, digits = 0)
        n_test = round(self$n_train * self$test_split, digits = 0)
        n_train = self$n_train - n_val -n_test - self$num_conformal

        random_idx = sample(1:nrow(theta), replace = FALSE)
        train_idx = random_idx[1:n_train]
        valid_idx = random_idx[(n_train + 1):(n_train + n_val)]
        test_idx = random_idx[(n_train + n_val + 1):(n_train + n_val + n_test)]
        # conformal_idx = random_idx[(n_train + n_val + n_test + 1):(n_train + n_val + n_test + num_conformal)]

        sumstat_tensor = torch_tensor(as.matrix(sumstat))
        theta_tensor = torch_tensor(as.matrix(theta))
        # sumstat_tensor = sumstat_tensor$unsqueeze(length(dim(sumstat_tensor)) + 1)
        # theta_tensor = theta_tensor$unsqueeze(length(dim(theta_tensor)) + 1)

        ds = tensor_dataset(sumstat_tensor, theta_tensor)

        valid_ds = dataset_subset(ds, valid_idx)
        valid_dl = dataloader(valid_ds, batch_size = self$batch_size, shuffle = TRUE, drop_last = TRUE)

        test_ds = dataset_subset(ds, test_idx)
        test_dl = dataloader(test_ds, batch_size = self$batch_size, shuffle = TRUE, drop_last = TRUE)

        # Make Ensemble dataset
        train_x = sumstat_tensor[train_idx,]
        train_y = theta_tensor[train_idx,]
        train_ds_list = ensemble_dataset(train_x, train_y, self$num_networks, randomize = TRUE)
        train_dl_list = train_ds_list %>% dataloader(batch_size = self$batch_size, shuffle = TRUE, drop_last = TRUE)

        return(list(train = train_dl_list, valid = valid_dl, test = test_dl))
      }

    },

    # ABC sampling of the simulations closest to the observed summary statistics
    abc_sampling = function() {
      # Apply kernel weighting to subset the simulations closest to observed
      # Select the theta values that best match the observed data
      # Improved Kernel sampling
      if (self$kernel == 'rbf') {
        kernel_values = rbf_kernel(self$sumstat, self$observed, length_scale = self$length_scale)
      }
      if (self$kernel == 'epanechnikov') {
        kernel_values = epanechnikov_kernel(self$sumstat, self$observed, bandwidth = self$bandwidth)
      }

      # Normalize the kernel values to form a proper weighting
      kernel_values = kernel_values/sum(kernel_values)
      # Sampling
      # Select sumstats and priors (parameters) in the given sampled region
      # = sumstats and unadjusted values
      if (self$sampling == 'rejection') {
        idx = rejection_sampling(self$kernel_values, self$theta, tol = self$tol)
      }
      if (self$sampling == 'importance') {
        idx = importance_sampling(self$kernel_values, self$theta, tol = self$tol)
      }

      theta = as.matrix(self$theta[idx,])
      sumstat = as.matrix(self$sumstat[idx,])

      return(list(theta = theta, sumstat = sumstat))
    },

    # Print a summary of the abcnn object
    summary = function() {
      # TODO Print number of samples and basic information on methods

      # CUDA is installed?
      cuda_is_available()

      print(self$device)
    },

    # Plot LDA projection of simulations with the observed points
    plot_lda = function() {

    },

    # Plot the training curves (training/validation)
    plot_training = function(discard_first = FALSE) {
      if (self$method != "deep ensemble") {
        train_metric = as.numeric(unlist(self$fitted$records$metrics$train))
        valid_metric = as.numeric(unlist(self$fitted$records$metrics$valid))
        eval = ifelse(self$method != "deep ensemble", self$eval_metrics$value, NA)

        if (discard_first) {
          train_metric[1] = NA
        }

        train_eval = data.frame(Epoch = rep(1:length(train_metric), self$output_dim),
                                Metric = c(train_metric, valid_metric),
                                Mode = c(rep("train", length(train_metric)), rep("validation", length(valid_metric))))

        ggplot(train_eval, aes(x = Epoch, y = Metric, color = Mode, fill = Mode)) +
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

        ggplot(train_eval, aes(x = Epoch, y = Metric, color = Mode, fill = Mode)) +
          geom_point() +
          geom_line() +
          xlab("Epoch") + ylab("Loss") +
          facet_wrap(~ Model) +
          theme_bw()
      }
    },

    # Plot predicted values (predicted ~ observed)
    plot_predicted = function() {
      # If same number of parameters x and y, infer pairwise relationship between each x and y (i.e. x1 ~ y1, x2 ~ y2...)
      # Otherwise order x axis by index
      if (ncol(self$theta) == ncol(self$sumstat)) {
        x_pos = as.numeric(unlist(self$observed))
      } else {
        x_pos = seq(1, nrow(self$observed))
      }
      df_predicted = data.frame(param = rep(colnames(self$theta), each = nrow(self$observed)),
                                x = x_pos,
                                mean = as.numeric(unlist(self$predictive_mean)),
                                ci_overall_upper = as.numeric(unlist(self$CI_overall_upper)),
                                ci_overall_lower = as.numeric(unlist(self$CI_overall_lower)),
                                ci_e_upper = as.numeric(unlist(self$CI_epistemic_upper)),
                                ci_e_lower = as.numeric(unlist(self$CI_epistemic_lower)))


      ggplot(data = df_predicted, aes(x = x)) +
        geom_line(aes(x = x, y = mean), color = "black") +
        facet_wrap(~ param, scales = "free") +
        geom_ribbon(aes(x = x, ymin = ci_overall_lower, ymax = ci_overall_upper), alpha = 0.3, fill = "black") +
        geom_ribbon(aes(x = x, ymin = ci_e_lower, ymax = ci_e_upper), alpha = 0.3, fill = "black") +
        xlab("Observed") + ylab("Predicted") +
        theme_bw()
    },

    # Plot the distributions of estimates and predictions
    plot_posterior = function(sample = 1, prior = TRUE) {
      # Dim 1 is number of MC samples (predictions)
      # Dim 2 is number of observations
      # Dim 3 is parameters (mu + sigma)
      tidy_predictions = data.frame(param = colnames(self$theta),
                                    mean = as.numeric(self$predictive_mean[sample,]),
                                    ci_lower = as.numeric(self$CI_overall_lower[sample,]),
                                    ci_upper = as.numeric(self$CI_overall_upper[sample,]),
                                    ci_e_lower = as.numeric(self$CI_epistemic_lower[sample,]),
                                    ci_e_upper = as.numeric(self$CI_epistemic_upper[sample,]))

      cols = c("Epistemic"="#D55E00","Overall"="#0072B2")

      if (prior) {
        tidy_priors = data.frame(param = rep(colnames(self$theta), each = nrow(self$theta)),
                                 prior = as.numeric(unlist(self$theta)))

        p = ggplot() +
          geom_histogram(data = tidy_priors, aes(x = prior), color = "darkgrey", fill = "grey", alpha = 0.1)
      } else {
        p = ggplot()
      }

      if (self$method %in% c("monte carlo dropout", "concrete dropout")) {
        posteriors = self$posterior_samples[,sample,]
        # output_names = unlist(lapply(c("mu", "sigma"), function(x) paste(colnames(theta), x, sep = "_")))
        posteriors = as.data.frame(posteriors)
        posteriors = posteriors[,1:self$output_dim]
        colnames(posteriors) = colnames(self$theta)
        posteriors$mc_sample = as.character(c(1:nrow(posteriors)))

        tidy_df = posteriors %>% tidyr::gather(param, prediction, any_of(colnames(self$theta)))

        p = p + geom_histogram(data = tidy_df, aes(x = prediction))
      }

      p = p +
        geom_vline(data = tidy_predictions, aes(xintercept = mean, colour = "Epistemic")) +
        geom_rect(data = tidy_predictions, aes(xmin = ci_e_lower, xmax = ci_e_upper, ymin = -Inf, ymax = Inf, colour = "Epitstemic", fill = "Epistemic"), alpha = 0.2) +
        geom_vline(data = tidy_predictions, aes(xintercept = ci_e_lower, colour = "Epistemic")) +
        geom_vline(data = tidy_predictions, aes(xintercept = ci_e_upper, colour = "Epistemic")) +
        geom_vline(data = tidy_predictions, aes(xintercept = ci_lower, colour = "Overall")) +
        geom_vline(data = tidy_predictions, aes(xintercept = ci_upper, colour = "Overall")) +
        geom_rect(data = tidy_predictions, aes(xmin = ci_lower, xmax = ci_upper, ymin = -Inf, ymax = Inf, colour = "Overall", fill = "Overall"), alpha = 0.2) +
        facet_wrap(~ param, scales = "free") +
        scale_colour_manual(name = "Uncertainty", values = cols) +
        scale_fill_manual(name = "Uncertainty", values = cols) +
        xlab("Value") + ylab("Count") +
        theme_bw() +
        theme(legend.position = "right")

      p

    }
  )
)





# Save the abcnn object and internal torch model
save_abcnn = function(object, prefix = "") {
  # Save the torch module used as model
  # torch_save(object$model, paste0(prefix, "_torch.Rds"))

  # Save the luz fitted object
  luz_save(object$fitted, paste0(prefix, "_luz.Rds"))

  # Save the abcnn object
  # Remove torch module and luz fitted to avoid serialization issues
  # object$model = NULL
  # object$fitted = NULL

  saveRDS(object, paste0(prefix, "_abcnn.Rds"))
}


# Load an abcnn object and internal torch model
load_abcnn = function(prefix = "") {
  object = readRDS(paste0(prefix, "_abcnn.Rds"))

  object$fitted = luz_load(paste0(prefix, "_luz.Rds"))
  object$model = object$fitted$model

  return(object)

}

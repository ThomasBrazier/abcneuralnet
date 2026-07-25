# Integration tests for complete end-to-end workflows

methods = c("monte carlo dropout", "gaussian monte carlo dropout", "concrete dropout", "deep ensemble", "tabnet-abc")

for (method in methods) {
  test_that(paste0("Complete workflow works for ", method), {
    n_samples = 10000
    theta_training = data.frame(param1 = runif(n_samples, 0, 1))
    sumstats_training = data.frame(
      stat1 = theta_training$param1 + rnorm(n_samples, 0, 0.05),
      stat2 = theta_training$param1^2 + rnorm(n_samples, 0, 0.1)
    )
    sumstats_observed = data.frame(stat1 = 0.4, stat2 = 0.2)

    # Complete workflow
    abc = abcnn$new(
      theta_training,
      sumstats_training,
      sumstats_observed,
      method = method,
      dropout = 0.2,
      scale_input = "minmax",
      scale_target = "minmax",
      num_hidden_layers = 3,
      num_hidden_dim = 64,
      batch_size = 32,
      learning_rate = 0.001,
      num_networks = 5,
      # TODO epsilon_adversarial = 0.01,
      epochs = 10,
      tol = 0.1,
      validation_split = 0.2,
      early_stopping = FALSE,
      num_posterior_samples = 1000,
      verbose = FALSE
    )

    expect_equal(abc$method, method)

    # Step 1: Fit model
    expect_no_error(abc$fit())

    # Step 2: Make predictions
    expect_no_error(abc$predict())
    expect_equal(dim(abc$predictive_mean), c(1, 1))
    # TODO expect_equal(dim(abc$predictive_variance), c(1, 1))
    expect_equal(dim(abc$epistemic_uncertainty), c(1, 1))
    if (method %in% c("concrete dropout", "gaussian monte carlo dropout", "deep ensemble")) {
      expect_equal(dim(abc$aleatoric_uncertainty), c(1, 1))
    }

    # Step 3: Generate posterior samples
    expect_no_error(abc$predictions())
    if (method %in% c("concrete dropout", "gaussian monte carlo dropout")) {
      # Mean + variance are registered in posterior samples
      expect_equal(dim(abc$posterior_samples), c(1000, 1, 2))
    } else {
      if (method %in% c("monte carlo dropout", "tabnet-abc"))
      expect_equal(dim(abc$posterior_samples), c(1000, 1, 1))
    }


    # Step 4: Generate plots
    expect_no_error(abc$plot_training())
    expect_no_error(abc$plot_prediction())
    expect_no_error(abc$plot_posterior())

    # Step 6: Get summary
    expect_no_error(abc$summary())

    # Step 7: Save and load model
    if (method != "tabnet-abc") {
      temp_prefix = tempfile(paste0(gsub(" ", "_", method), "_integration_"))
      expect_no_error(save_abcnn(abc, prefix = temp_prefix))

      abc_loaded = load_abcnn(prefix = temp_prefix)
      testthat::expect_r6_class(abc_loaded, "abcnn")
      expect_equal(abc_loaded$method, method)

      # Step 8: Verify loaded model works
      abc_loaded = load_abcnn(prefix = temp_prefix)
      expect_no_error(abc_loaded$fit())
      expect_no_error(abc_loaded$predict())

      abc_loaded = load_abcnn(prefix = temp_prefix)
      expect_no_error(abc_loaded$predict())
      expect_equal(dim(abc_loaded$predictive_mean), c(1, 1))

      # Step 9: Clean up
      unlink(paste0(temp_prefix, "*"))
    }

    # Validate results
    expect_true(is.finite(abc$predictive_mean[1, 1]))
    # TODO expect_true(abc$predictive_variance[1, 1] > 0)
    if (method != "deep ensemble") {
      expect_true(all(is.finite(abc$posterior_samples)))
    }

    pred = abc$predictions()
    expect_true(pred$epistemic_uncertainty > 0)
    expect_true(pred$epistemic_conformal_lower < pred$epistemic_conformal_upper)
    if (method != "deep ensemble") {
      expect_true(pred$posterior_lower_ci < pred$posterior_upper_ci)
    }

    if (method %in% c("concrete dropout", "deep ensemble")) {
      expect_true(pred$aleatoric_uncertainty > 0)
      expect_true(pred$overall_uncertainty > 0)
      expect_true(pred$overall_conformal_lower < pred$overall_conformal_upper)
    }
  })
}


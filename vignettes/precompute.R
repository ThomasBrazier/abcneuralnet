# Pre-compute the vignette.
#
# The vignette loads large fitted models from `inst/extdata`, which is kept out
# of the package tarball by `.Rbuildignore` because it is far above the CRAN
# size limit. It also uses packages that are not dependencies of the package
# (MCMCpack, spatstat, mvtnorm, ...). Building it during `R CMD build` would
# therefore fail.
#
# The expensive document is kept as `Bayesian_neural_networks_inference.Rmd.orig`
# and knitted here into a plain `.Rmd` in which every result is already inlined.
# `knitr::knit()` turns each `{r}` chunk into a fenced `r` block followed by its
# output, so the generated vignette contains no code to evaluate: it builds in
# seconds, needs no data and no extra packages.
#
# Regenerate whenever the `.Rmd.orig` changes, from the package root:
#
#     Rscript vignettes/precompute.R
#
# then commit the regenerated `.Rmd` together with `vignettes/figure/`.
#
# Note that `knit()` has to run with `vignettes/` as the working directory, so
# that the `../inst/extdata/` paths in the document resolve and the figures are
# written to `vignettes/figure/`.

# The body is wrapped in a function so that `on.exit()` is tied to a frame and
# restores the working directory only once `knit()` has returned. At top level,
# under `source()`, it would fire straight away and `knit()` would look for the
# document in the wrong directory.
precompute = function() {
  # Works whether this is run from the package root or from `vignettes/`
  dir = if (dir.exists("vignettes")) "vignettes" else "."

  owd = setwd(dir)
  on.exit(setwd(owd), add = TRUE)

  knitr::knit(input = "Bayesian_neural_networks_inference.Rmd.orig",
              output = "Bayesian_neural_networks_inference.Rmd")
}

precompute()

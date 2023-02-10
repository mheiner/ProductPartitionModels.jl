# ProductPartitionModels.jl

Fit product partition (PPM; [Hartigan, 1990](https://www.tandfonline.com/doi/abs/10.1080/03610929008830345)) and covariate-dependent product partition (PPMx; [Muller, Quintana, and Rosner, 2011](https://www.tandfonline.com/doi/abs/10.1198/jcgs.2011.09066)) models using Markov chain Monte Carlo (MCMC). Covariates are currently assumed to be of continuous type. They may optionally contain `missing` values ([Page, Quintana, and Muller, 2022](https://www.tandfonline.com/doi/full/10.1080/10618600.2021.1999824); Heiner, Page, and Quintana, 2023).

## Workflow
First create a model object with `Model_PPMx`. Then run MCMC with `mcmc!` on the model object. For example, using a vector of responses `y` and covariate matrix `X`, fit the model with

    model = Model_PPMx(y, X, 0, similarity_type=:NN, sampling_model=:Reg)

    sims = mcmc!(model, 1000)

The object `sims` is a vector of dictionaries, with each dictionary indexed by symbols. Access a list of dictionary keys with

    keys(sims[1])

Posterior samples of parameter values or other quantities in the keys list can be extracted by collecting them into arrays

    sims_llik = [ sims[i][:llik] for i in 1:length(sims) ]

Posterior draws can also be used for in-sample prediction

    Ypred_is, Cpred_is, Mpred_is = postPred(model, sims)

or out-of-sample prediction

    Ypred_oos, Cpred_oos, Mpred_oos = postPred(Xpred, model, sims)

Model objects can be initialized with user-defined priors and similarity functions. They can also be modified prior to running MCMC, for example

    model.state.similarity = Similarity_NNiChisq_indep(0.0, 0.1, 10.0, 0.5^2)
    model.prior.baseline.mu0_mean = 1.0
    model.state.baseline.sig_upper = 1.5

See documentation on `Model_PPMx`, `mcmc!`, and `postPred` for additional options, or run

    model |> typeof |> fieldnames
    model.prior |> typeof |> fieldnames
    model.state |> typeof |> fieldnames

to explore the fields in each object.

## Citing this package
See `CITATION.bib`.
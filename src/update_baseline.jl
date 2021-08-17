# update_baseline.jl

# export ; 

function update_mu0!(base::Baseline_NormDLUnif, mu_vec::Vector{T}, prior::Prior_baseline_NormDLUnif) where T <: Real
    n = length(mu_vec)
    sum_mu = sum(mu_vec)

    prior_mu0_sig2 = prior.mu0_sd^2
    sig02_now = base.sig0^2

    v1 = 1.0 / (1.0 / prior_mu0_sig2 + n / sig02_now)
    m1 = v1 * (prior.mu0_mean / prior_mu0_sig2 + sum_mu / sig02_now )

    base.mu0 = randn()*sqrt(v1) + m1
    return nothing
end

mutable struct TargetArgs_NormSigUnif{T <: Real} <: TargetArgs
    y::Vector{T}
    mu::T
end

function logtarget(sig::Real, args::TargetArgs_NormSigUnif)
    ee = args.y .- args.mu
    ss = sum(ee.^2)
    out = -0.5 * ss / sig^2 - length(args.y)*log(sig)
    return out, ss
end

function update_sig0!(base::Baseline_NormDLUnif, mu_vec::Vector{T}, prior::Prior_baseline_NormDLUnif) where T <: Real
    sigout, lt = shrinkSlice(base.sig0, 0.0, prior.sig0_upper,
                    logtarget, TargetArgs_NormSigUnif(mu_vec, base.mu0))
    base.sig0 = sigout
    return nothing
end

function update_tau0!(base::Baseline_NormDLUnif, lik_params::Vector{LikParams_PPMxReg{TR}}, prior::Prior_baseline_NormDLUnif) where TR <: Real
    K = length(lik_params)
    beta_all = vcat([ deepcopy(lik_params[k].beta) for k in 1:K ]...)
    var_fixed_beta = vcat([ lik_params[k].beta_hypers.tau^2 .*
            lik_params[k].beta_hypers.phi.^2 .* 
            lik_params[k].beta_hypers.psi for k in 1:K ]...)
    sh1 = prior.tau02_sh + 0.5*length(beta_all)
    sc1 = prior.tau02_sc + 0.5*sum( beta_all.^2 ./ var_fixed_beta )

    base.tau0 = sqrt( rand( Distributions.InverseGamma(sh1, sc1) ) )
    return nothing
end

function update_baseline!(model::Model_PPMx, update_params::Vector{Symbol})

    K = length(model.state.lik_params)

    if :mu0 in update_params
        mu_vec = [ model.state.lik_params[k].mu for k in 1:K ]
        update_mu0!(model.state.baseline, mu_vec, model.prior.baseline)
    end

    if :sig0 in update_params
        mu_vec = [ model.state.lik_params[k].mu for k in 1:K ]
        update_sig0!(model.state.baseline, mu_vec, model.prior.baseline)
    end

    if :tau0 in update_params
        update_tau0!(model.state.baseline, model.state.lik_params, model.prior.baseline)
    end

    return nothing
end


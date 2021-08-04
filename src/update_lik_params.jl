# update_lik_params.jl

struct TargetArgs_EsliceBetas{T <: Real, TT <: LikParams_PPMxReg} <: TargetArgs
    y_k::Vector{T}
    X_k::Union{Matrix{T}, Matrix{Union{T, Missing}}}
    ObsXIndx_k::Vector{ObsXIndx}
    lik_params_k::TT
    Xstats_k::Vector{Similarity_NiG_indep_stats}
    similarity::Similarity_NiG_indep
end

struct TargetArgs_sliceSig{T <: Real} <: TargetArgs
    y_k::Vector{T}
    means::Vector{T}
    vars::Vector{T}
    sig_old::T
end

function llik_k_forEsliceBeta(beta_cand::Vector{T}, args::TargetArgs_EsliceBetas)

    lik_params_cand = deepcopy(args.lik_params_k)
    lik_params_cand.beta = beta_cand

    return llik_k(args.y_k, args.X_k, args.ObsXIndx_k, lik_params_cand, 
                  args.Xstats_k, args.similarity)
end

function llik_k_forSliceSig(sig_cand::Real, args::TargetArgs_sliceSig)
    return llik_k(args.y_k, args.means, args.vars, args.sig_old::T, sig_cand)
end

function update_lik_params!(model::Model_PPMx)

    K = maximum(model.state.C)
    prior_mean_beta = zeros(model.p)

    for k in 1:K

        indx_k = findall(model.state.C.==k)

        ## update betas, produces vectors of obs-specific means and variances
        prior_var_beta = model.state.lik_params[k].beta_hypers.τ.^2 .* 
            model.state.lik_params[k].beta_hypers.ϕ.^2 .* 
            model.state.lik_params[k].beta_hypers.ψ

        model.state.lik_params[k].beta, beta_upd_stats = ellipSlice(
            model.state.lik_params[k].beta, 
            prior_mean_beta, prior_var_beta,
            llik_k_forEsliceBeta, 
            TargetArgs_EsliceBetas(model.y[indx_k], model.X[indx_k,:], model.obsXIndx[indx_k], 
                model.state.llik_params[k], model.state.Xstats[k], model.state.similarity)
            )

        ## update beta hypers (could customize a function here to accommodate different shrinkage priors)
        model.state.lik_params[k].beta_hypers.ψ = update_ψ(model.state.lik_params[k].beta_hypers.ϕ, 
            model.state.lik_params[k].beta, 
            model.state.lik_params[k].beta_hypers.τ)
        
        model.state.lik_params[k].beta_hypers.τ = update_τ(model.state.lik_params[k].beta_hypers.ϕ, 
            model.state.lik_params[k].beta, 1.0/model.p)
        
        model.state.lik_params[k].beta_hypers.ϕ = update_ϕ(model.state.lik_params[k].beta, 1.0/model.p)

        ## update sig, which preserves means to be modified in the update for means
        model.state.lik_params[k].sig, sig_upd_stats = shrinkSlice(model.state.lik_params[k].sig, 
            0.0, model.state.baseline.upper_σ,
            llik_k_forSliceSig, 
            TargetArgs_sliceSig(model.y[indx_k], beta_upd_stats[2], beta_upd_stats[3], model.state.lik_params[k].sig)
            ) # sig_old doesn't need to be updated during slice sampler--this is just a computational trick

        ## update mu
        yy = model.y[indx_k] - sig_upd_stats[2] .+ model.state.lik_params[k].mu
        one_div_var = 1.0 ./ sig_upd_stats[3]
        yy_div_var = yy .* one_div_var
        v1 = 1.0 / (1.0/model.state.baseline.σ0^2 + sum(one_div_var))
        m1 = v1 * (model.state.baseline.μ0 / model.state.baseline.σ0^2 + sum(yy_div_var))
        model.state.lik_params[k].mu = randn()*sqrt(v1) + m1

    end

    return nothing
end


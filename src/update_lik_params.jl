# update_lik_params.jl

struct TargetArgs_EsliceBetas{T <: Real, TT <: LikParams_PPMxReg, TTT <: Similarity_PPMxStats, TTTT <: Similarity_PPMx} <: TargetArgs
    y_k::Vector{T}
    X_k::Union{Matrix{T}, Matrix{Union{T, Missing}}}
    ObsXIndx_k::Vector{ObsXIndx}
    lik_params_k::TT
    Xstats_k::Vector{TTT}
    similarity::TTTT
end

struct TargetArgs_sliceSig{T <: Real} <: TargetArgs
    y_k::Vector{T}
    means::Vector{T}
    vars::Vector{T}
    sig_old::T

    beta::Vector{T}
    tau0::T
    tau::T
    phi::Vector{T}
    psi::Vector{T}
end

function llik_k_forEsliceBeta(beta_cand::Vector{T}, args::TargetArgs_EsliceBetas) where T <: Real

    lik_params_cand = deepcopy(args.lik_params_k)
    lik_params_cand.beta = beta_cand

    return llik_k(args.y_k, args.X_k, args.ObsXIndx_k, lik_params_cand,
                  args.Xstats_k, args.similarity)
end

function llik_k_forSliceSig(sig_cand::Real, args::TargetArgs_sliceSig)

    llik_kk, means, vars = llik_k(args.y_k, args.means, args.vars, args.sig_old, sig_cand)

    prior_var_beta = args.tau^2 .* sig_cand^2 .*
        args.tau0^2 .* args.phi.^2 .* args.psi

    lpri_beta = 0.0
    for ell in 1:length(args.beta)
        vv = prior_var_beta[ell]
        lpri_beta += -0.5*log(2π) - 0.5*log(vv) - 0.5*(args.beta[ell])^2/vv # prior mean is 0
    end

    llik_kk += lpri_beta

    return llik_kk, means, vars
end

function update_lik_params!(model::Model_PPMx,
    update::Vector{Symbol}=[:mu, :sig, :beta],
    sliceiter::Int=5000)

    K = maximum(model.state.C)
    prior_mean_beta = zeros(model.p)

    for k in 1:K ## can parallelize; would need to pass rng through updates (including slice functions and hyper updates)

        indx_k = findall(model.state.C.==k)

        if (:beta in update)
            ## update betas, produces vectors of obs-specific means and variances
            prior_var_beta = model.state.lik_params[k].beta_hypers.tau^2 .* model.state.lik_params[k].sig^2 .*
                model.state.baseline.tau0^2 .*
                model.state.lik_params[k].beta_hypers.phi.^2 .*
                model.state.lik_params[k].beta_hypers.psi

            model.state.lik_params[k].beta, beta_upd_stats, iters_eslice = ellipSlice(
                model.state.lik_params[k].beta,
                prior_mean_beta, prior_var_beta,
                llik_k_forEsliceBeta,
                TargetArgs_EsliceBetas(model.y[indx_k], model.X[indx_k,:], model.obsXIndx[indx_k],
                    model.state.lik_params[k], model.state.Xstats[k], model.state.similarity),
                    sliceiter
                )

            ## update beta hypers (could customize a function here to accommodate different shrinkage priors)
            model.state.lik_params[k].beta_hypers.psi = update_ψ(model.state.lik_params[k].beta_hypers.phi,
                model.state.lik_params[k].beta ./ model.state.baseline.tau0 ./ model.state.lik_params[k].sig,
                model.state.lik_params[k].beta_hypers.tau
            )

            model.state.lik_params[k].beta_hypers.tau = update_τ(model.state.lik_params[k].beta_hypers.phi,
                model.state.lik_params[k].beta ./ model.state.baseline.tau0 ./ model.state.lik_params[k].sig,
                1.0/model.p
            )

            model.state.lik_params[k].beta_hypers.phi = update_ϕ(model.state.lik_params[k].beta ./ model.state.baseline.tau0 ./ model.state.lik_params[k].sig,
                1.0/model.p
            )
        else
            beta_upd_stats = llik_k(model.y[indx_k], model.X[indx_k,:], model.obsXIndx[indx_k],
                model.state.lik_params[k], model.state.Xstats[k], model.state.similarity)
        end

        ## update sig, which preserves means to be modified in the update for means
        if (:sig in update)
            model.state.lik_params[k].sig, sig_upd_stats, iters_sslice = shrinkSlice(model.state.lik_params[k].sig,
                0.0, model.state.baseline.sig_upper,
                llik_k_forSliceSig,
                TargetArgs_sliceSig(model.y[indx_k], beta_upd_stats[2], beta_upd_stats[3], model.state.lik_params[k].sig,
                                    model.state.lik_params[k].beta, model.state.baseline.tau0, model.state.lik_params[k].beta_hypers.tau,
                                    model.state.lik_params[k].beta_hypers.phi, model.state.lik_params[k].beta_hypers.psi),
                sliceiter
            ) # sig_old doesn't need to be updated during intermediate proposals of slice sampler--vars isn't updated either,
              # so each step uses the same (original) set of target args. This allows us to use the generic slice sampler code.
        else
            sig_upd_stats = deepcopy(beta_upd_stats) # means (indx 2) and vars (indx 3) that get used haven't changed
        end

        ## update mu
        if (:mu in update)
            yy = model.y[indx_k] - sig_upd_stats[2] .+ model.state.lik_params[k].mu
            one_div_var = 1.0 ./ sig_upd_stats[3]
            yy_div_var = yy .* one_div_var
            v1 = 1.0 / (1.0/model.state.baseline.sig0^2 + sum(one_div_var))
            m1 = v1 * (model.state.baseline.mu0 / model.state.baseline.sig0^2 + sum(yy_div_var))
            model.state.lik_params[k].mu = randn()*sqrt(v1) + m1
        end

    end

    return nothing
end

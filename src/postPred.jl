# postPred.jl

export postPred, postPredLogdens;


function predWeights(i::Int,
    Xpred::Union{Matrix{T},Matrix{Union{T,Missing}}} where {T<:Real},
    lcohesions::Vector{T} where {T<:Real},
    Xstats::Vector{Vector{TT}} where {TT<:Similarity_PPMxStats},
    lsimilarities::Vector{Vector{T}} where {T<:Real},
    K::Int, S::Vector{Int},
    model::Model_PPMx, lcohes1::T where {T<:Real})

    lw = Vector{Float64}(undef, K + 1)

    # cohesion with obs i added
    lcohes_cand = [log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[k] + 1, true)) for k in 1:K]

    # stats for similarity with obs i added (each X[i,:], 1:p); similarity with obs i added
    stats_cand = [[Similarity_stats(Xstats[k][j], Xpred[i, j], :add) for j in 1:model.p] for k in 1:K]

    lsimilar_cand = [[log_similarity(model.state.similarity, stats_cand[k][j], true) for j in 1:model.p] for k in 1:K]

    for k in 1:K
        lw[k] = lcohes_cand[k] + sum(lsimilar_cand[k]) - lcohesions[k] - sum(lsimilarities[k])
    end

    # weight for new singleton cluster
    stats_newclust = [Similarity_stats(model.state.similarity, [Xpred[i, j]]) for j = 1:model.p]

    lsimilar_newclust = [log_similarity(model.state.similarity, stats_newclust[j], true) for j in 1:model.p]
    lw[K+1] = lcohes1 + sum(lsimilar_newclust)

    return lw .- maximum(lw)
end

function postPred(Xpred::Union{Matrix{T},Matrix{Union{T,Missing}}},
    model::Model_PPMx,
    sims::Vector{Dict{Symbol,Any}},
    update_params::Vector{Symbol}=[:mu, :sig, :beta, :mu0, :sig0]) where {T<:Real}

    ## currently assumes cohesion and similarity parameters are fixed
    ## treats each input as the n+1th observation with no consideration of them clustering together
    ## does not update the likelihood centering with the prediction obs, stats

    n_pred, p_pred = size(Xpred)
    p_pred == model.p || throw("Xpred and original X have different numbers of predictors.")

    obsXIndx_pred = [ObsXIndx(Xpred[i, :]) for i in 1:n_pred]

    n_sim = length(sims)

    Cpred = Matrix{Int}(undef, n_sim, n_pred)
    Ypred = Matrix{typeof(model.y[1])}(undef, n_sim, n_pred)
    Mean_pred = Matrix{typeof(model.y[1])}(undef, n_sim, n_pred)

    lcohes1 = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, 1, true))
    x_mean_empty, x_sd_empty = aux_moments_empty(model.state.similarity)

    for ii in 1:n_sim

        lcohesions, Xstats, lsimilarities = get_lcohlsim(sims[ii][:C], model.X, model.state.cohesion, model.state.similarity)
        K = length(lcohesions)
        S = StatsBase.counts(sims[ii][:C], K)

        Xbars = Matrix{typeof(model.y[1])}(undef, K, model.p)
        Sds = Matrix{typeof(model.y[1])}(undef, K, model.p)

        for k in 1:K
            Xbars[k, :], Sds[k, :] = aux_moments_k(Xstats[k], model.state.similarity)
        end

        for i in 1:n_pred

            lw = predWeights(i, Xpred, lcohesions, Xstats, lsimilarities, K, S, model, lcohes1)

            # sample membership for obs i
            C_i = StatsBase.sample(StatsBase.Weights(exp.(lw)))
            if C_i > K
                C_i = 0
            end
            Cpred[ii, i] = C_i

            # draw y value
            if C_i > 0
                mean_now = deepcopy(sims[ii][:lik_params][C_i][:mu])
                sig2_now = sims[ii][:lik_params][C_i][:sig]^2

                if obsXIndx_pred[i].n_mis > 0
                    sig2_now += sum(sims[ii][:lik_params][C_i][:beta][obsXIndx_pred[i].indx_mis] .^ 2)
                end

                if obsXIndx_pred[i].n_obs > 0
                    z = (Xpred[i, obsXIndx_pred[i].indx_obs] - Xbars[C_i, obsXIndx_pred[i].indx_obs]) ./ Sds[C_i, obsXIndx_pred[i].indx_obs]
                    mean_now += z' * sims[ii][:lik_params][C_i][:beta][obsXIndx_pred[i].indx_obs]
                end
            else
                if (:mu0 in update_params) && (:sig0 in update_params)
                    lik_params_new = simpri_lik_params(
                        Baseline_NormDLUnif(sims[ii][:baseline][:mu0], sims[ii][:baseline][:sig0],
                            model.state.baseline.tau0, model.state.baseline.sig_upper),
                        model.p, model.state.lik_params[1], update_params
                    ) # this is a messy patch, needs attention
                else
                    lik_params_new = simpri_lik_params(model.state.baseline,
                        model.p, model.state.lik_params[1], update_params
                    )
                end

                mean_now = deepcopy(lik_params_new.mu)
                sig2_now = lik_params_new.sig^2

                if obsXIndx_pred[i].n_mis > 0
                    sig2_now += sum(lik_params_new.beta[obsXIndx_pred[i].indx_mis] .^ 2)
                end

                if obsXIndx_pred[i].n_obs > 0
                    z = (Xpred[i, obsXIndx_pred[i].indx_obs] .- x_mean_empty) ./ x_sd_empty
                    mean_now += z' * lik_params_new.beta[obsXIndx_pred[i].indx_obs]
                end
            end

            Mean_pred[ii, i] = deepcopy(mean_now)
            Ypred[ii, i] = randn() .* sqrt(sig2_now) + mean_now

        end
    end

    return Ypred, Cpred, Mean_pred
end
function postPred(model::Model_PPMx,
    sims::Vector{Dict{Symbol,Any}}) where {T<:Real}

    ## currently assumes cohesion and similarity parameters are fixed
    ## treats each input as the n+1th observation with no consideration of them clustering together
    ## does not update the likelihood centering with the prediction obs, stats

    n_sim = length(sims)

    Ypred = Matrix{typeof(model.y[1])}(undef, n_sim, model.n)
    Mean_pred = Matrix{typeof(model.y[1])}(undef, n_sim, model.n)

    for ii in 1:n_sim

        lcohesions, Xstats, lsimilarities = get_lcohlsim(sims[ii][:C], model.X, model.state.cohesion, model.state.similarity)
        K = length(lcohesions)
        S = StatsBase.counts(sims[ii][:C], K)

        Xbars = Matrix{typeof(model.y[1])}(undef, K, model.p)
        Sds = Matrix{typeof(model.y[1])}(undef, K, model.p)

        for k in 1:K
            Xbars[k, :], Sds[k, :] = aux_moments_k(Xstats[k], model.state.similarity)
        end

        for i in 1:model.n
            # draw y value
            C_i = deepcopy(sims[ii][:C][i])
            mean_now = deepcopy(sims[ii][:lik_params][C_i][:mu])
            sig2_now = sims[ii][:lik_params][C_i][:sig]^2

            if model.obsXIndx[i].n_mis > 0
                sig2_now += sum(sims[ii][:lik_params][C_i][:beta][model.obsXIndx[i].indx_mis] .^ 2)
            end

            if model.obsXIndx[i].n_obs > 0
                z = (model.X[i, model.obsXIndx[i].indx_obs] - Xbars[C_i, model.obsXIndx[i].indx_obs]) ./ Sds[C_i, model.obsXIndx[i].indx_obs]
                mean_now += z' * sims[ii][:lik_params][C_i][:beta][model.obsXIndx[i].indx_obs]
            end

            Mean_pred[ii, i] = deepcopy(mean_now)
            Ypred[ii, i] = randn() .* sqrt(sig2_now) + mean_now

        end
    end

    return Ypred, Mean_pred
end

function postPredLogdens(Xpred::Union{Matrix{T},Matrix{Union{T,Missing}}},
    ypred::Vector{T},
    model::Model_PPMx,
    sims::Vector{Dict{Symbol,Any}},
    update_params::Vector{Symbol}=[:mu, :sig, :beta, :mu0, :sig0]) where {T<:Real}

    ## currently assumes cohesion and similarity parameters are fixed
    ## treats each input as the n+1th observation with no consideration of them clustering together
    ## does not update the likelihood centering with the prediction obs, stats

    n_y = length(ypred)
    n_pred, p_pred = size(Xpred)
    p_pred == model.p || throw("Xpred and original X have different numbers of predictors.")

    obsXIndx_pred = [ObsXIndx(Xpred[i, :]) for i in 1:n_pred]

    n_sim = length(sims)

    # lDenspred = Matrix{typeof(model.y[1])}(undef, n_sim, n_pred, n_y)
    lDenspred = zeros(n_sim, n_pred, n_y)

    lcohes1 = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, 1, true))
    x_mean_empty, x_sd_empty = aux_moments_empty(model.state.similarity)

    halflog2pi = 0.5*log(2π)

    for ii in 1:n_sim

        lcohesions, Xstats, lsimilarities = get_lcohlsim(sims[ii][:C], model.X, model.state.cohesion, model.state.similarity)
        K = length(lcohesions)
        S = StatsBase.counts(sims[ii][:C], K)

        Xbars = Matrix{typeof(model.y[1])}(undef, K, model.p)
        Sds = Matrix{typeof(model.y[1])}(undef, K, model.p)

        for k in 1:K
            Xbars[k, :], Sds[k, :] = aux_moments_k(Xstats[k], model.state.similarity)
        end

        for i in 1:n_pred

            lw = predWeights(i, Xpred, lcohesions, Xstats, lsimilarities, K, S, model, lcohes1)
            lw = lw .- logsumexp(lw)

            # draw y value
            mean_now = Vector{typeof(model.y[1])}(undef, K + 1)
            sig2_now = Vector{typeof(model.y[1])}(undef, K + 1)

            for k in 1:K
                mean_now[k] = deepcopy(sims[ii][:lik_params][k][:mu])
                sig2_now[k] = sims[ii][:lik_params][k][:sig]^2

                if obsXIndx_pred[i].n_mis > 0
                    sig2_now[k] += sum(sims[ii][:lik_params][k][:beta][obsXIndx_pred[i].indx_mis] .^ 2)
                end

                if obsXIndx_pred[i].n_obs > 0
                    z = (Xpred[i, obsXIndx_pred[i].indx_obs] - Xbars[k, obsXIndx_pred[i].indx_obs]) ./ Sds[k, obsXIndx_pred[i].indx_obs]
                    mean_now[k] += z' * sims[ii][:lik_params][k][:beta][obsXIndx_pred[i].indx_obs]
                end
            end

            if (:mu0 in update_params) && (:sig0 in update_params)
                lik_params_new = simpri_lik_params(
                    Baseline_NormDLUnif(sims[ii][:baseline][:mu0], sims[ii][:baseline][:sig0],
                        model.state.baseline.tau0, model.state.baseline.sig_upper),
                    model.p, model.state.lik_params[1], update_params
                ) # this is a messy patch, needs attention
            else
                lik_params_new = simpri_lik_params(model.state.baseline,
                    model.p, model.state.lik_params[1], update_params
                )
            end

            mean_now[K+1] = deepcopy(lik_params_new.mu)
            sig2_now[K+1] = lik_params_new.sig^2

            if obsXIndx_pred[i].n_mis > 0
                sig2_now[K+1] += sum(lik_params_new.beta[obsXIndx_pred[i].indx_mis] .^ 2)
            end

            if obsXIndx_pred[i].n_obs > 0
                z = (Xpred[i, obsXIndx_pred[i].indx_obs] .- x_mean_empty) ./ x_sd_empty
                mean_now[K+1] += z' * lik_params_new.beta[obsXIndx_pred[i].indx_obs]
            end

            for k in (K+1)
                lDenspred[ii, i, :] .+= lw[k] .- halflog2pi .- 0.5*log(sig2_now[k]) .- (0.5.*(ypred .- mean_now[k]).^2 ./ sig2_now[k])
            end

        end
    end

    return lDenspred
end
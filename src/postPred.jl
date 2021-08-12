# postPred.jl

export postPred;

function postPred(Xpred::Union{Matrix{T}, Matrix{Union{T, Missing}}}, 
    model::Model_PPMx,
    sims::Vector{Dict{Symbol, Any}}) where T <: Real

    ## currently assumes cohesion and similarity parameters are fixed
    ## treats each input as the n+1th observation with no consideration of them clustering together
    ## does not update the likelihood centering with the prediction obs, stats

    n_pred, p_pred = size(Xpred)
    p_pred == model.p || throw("Xpred and original X have different numbers of predictors.")

    obsXIndx_pred = [ ObsXIndx(Xpred[i,:]) for i in 1:n_pred ]

    n_sim = length(sims)

    Cpred = Matrix{Int}(undef, n_sim, n_pred)
    Ypred = Matrix{typeof(model.y[1])}(undef, n_sim, n_pred)

    lcohes1 = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, 1, true))
    x_mean_empty = model.state.similarity.m0 # could do something else
    x_sd_empty = sqrt(model.state.similarity.b0 / (model.state.similarity.a0 + 1.0) / model.state.similarity.sc_div0) # could do something else

    for ii in 1:n_sim

        lcohesions, Xstats, lsimilarities = get_lcohlsim(sims[ii][:C], model.X, model.state.cohesion, model.state.similarity)
        K = length(lcohesions)
        S = StatsBase.counts(sims[ii][:C], K)

        Xbars = Matrix{typeof(model.y[1])}(undef, K, model.p)
        Sds = Matrix{typeof(model.y[1])}(undef, K, model.p)

        for k in 1:K

            Ck_indx = findall(sims[ii][:C] .== k)

            for j in 1:model.p

                n_fill = sum(ismissing.(X[Ck_indx, j]))
                
                if n_fill < S[k]
                    xbar_now = Xstats[k][j].sumx / Xstats[k][j].n
                    mean_now = xbar_now # could do something else
                    if Xstats[k][j].n > 1
                        s2_now = (Xstats[k][j].sumx2 - Xstats[k][j].n * xbar_now^2) / (Xstats[k][j].n - 1.0)
                        sd_now = sqrt(s2_now) # could do something else
                    else
                        sd_now = x_sd_empty
                    end 
                else
                    mean_now = x_mean_empty
                    sd_now = x_sd_empty
                end

                Xbars[k,j] = mean_now
                Sds[k,j] = sd_now
            end
        end

        for i in 1:n_pred
            # draw cluster membership
            lw = Vector{Float64}(undef, K + 1)

            # cohesion with obs i added
            lcohes_cand = [ log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[k] + 1, true)) for k in 1:K ]

            # stats for similarity with obs i added (each X[i,:], 1:p); similarity with obs i added
            stats_cand = [ [ Similarity_NiG_indep_stats(Xstats[k][j], Xpred[i,j], :add) for j in 1:model.p] for k in 1:K ]
            lsimilar_cand = [ [ log_similarity(model.state.similarity, stats_cand[k][j], true) for j in 1:model.p ] for k in 1:K  ]
            
            for k in 1:K
                lw[k] = lcohes_cand[k] + sum(lsimilar_cand[k]) - lcohesions[k] - sum(lsimilarities[k])
            end

            # weight for new singleton cluster
            stats_newclust = [ Similarity_NiG_indep_stats([Xpred[i,j]]) for j = 1:model.p ]
            lsimilar_newclust = [ log_similarity(model.state.similarity, stats_newclust[j], true) for j in 1:model.p ]
            lw[K + 1] = lcohes1 + sum(lsimilar_newclust)

            # sample membership for obs i
            lw = lw .- maximum(lw)
            C_i = StatsBase.sample(StatsBase.Weights(exp.(lw)))
            if C_i > K
                C_i = 0
            end
            Cpred[ii, i] = C_i

            # draw y value
            if C_i > 0
                mean_now = sims[ii][:lik_params][C_i][:mu]
                sig2_now = sims[ii][:lik_params][C_i][:sig]^2

                if obsXIndx_pred[i].n_mis > 0
                    sig2_now += sum( (sims[ii][:lik_params][C_i][:beta][obsXIndx_pred[i].indx_mis] .* Sds[C_i, obsXIndx_pred[i].indx_mis]).^2 )
                end

                if obsXIndx_pred[i].n_obs > 0
                    xc = Xpred[i, obsXIndx_pred[i].indx_obs] - Xbars[C_i, obsXIndx_pred[i].indx_obs]
                    mean_now += xc' * sims[ii][:lik_params][C_i][:beta][obsXIndx_pred[i].indx_obs]
                end
                
                Ypred[ii, i] = randn() .* sqrt(sig2_now) + mean_now
            else
                lik_params_new = simpri_lik_params(model.state.baseline, model.p)

                mean_now = lik_params_new.mu
                sig2_now = lik_params_new.sig^2

                if obsXIndx_pred[i].n_mis > 0
                    sig2_now += sum( ( lik_params_new.beta[obsXIndx_pred[i].indx_mis] .* x_sd_empty).^2 )
                end

                if obsXIndx_pred[i].n_obs > 0
                    xc = Xpred[i, obsXIndx_pred[i].indx_obs] .- x_mean_empty
                    mean_now += xc' * lik_params_new.beta[obsXIndx_pred[i].indx_obs]
                end
                
                Ypred[ii, i] = randn() .* sqrt(sig2_now) + mean_now
            end

        end
    end

    return Ypred, Cpred
end
# update_config.jl

function update_Ci!(model::Model_PPMx, i::Int)
    
    K = maximum(model.state.C)
    S = StatsBase.counts(model.state.C, K)

    ## remove obs i from current C and modify current cohesions, Xstats, and similarities without obs i
    ci_old = model.state.C[i]
    model.state.C[i] = 0

    if S[ci_old] > 1

        S[ci_old] -= 1
            
        model.state.lcohesions[ci_old] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[ci_old], true))

        for j in 1:model.p
            model.state.Xstats[ci_old][j] = Similarity_NiG_indep_stats(model.state.Xstats[ci_old][j], model.X[i,j], :subtract)
            model.state.lsimilarities[ci_old][j] = log_similarity(model.state.similarity, model.state.Xstats[ci_old][j], true)
        end

    else ## collapse if C[i] was a singleton

        deleteat!(model.state.lcohesions, ci_old)
        deleteat!(model.state.Xstats, ci_old)
        deleteat!(model.state.lsimilarities, ci_old)
        deleteat!(model.state.lik_params, ci_old)
        deleteat!(S, ci_old)

        model.state.C[findall(model.state.C .> ci_old)] .-= 1 # relabel all obs with label higher than the deleted one
        K -= 1

    end

    ## get cohesions, Xstats, and similarities each with obs i hypothetically added to each cluster
    lcohesions1 = [ log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[k] + 1, true)) for k in 1:K ]
    Xstats1 = [ [ Similarity_NiG_indep_stats(model.state.Xstats[k][j], model.X[i,j], :add) for j in 1:model.p ] for k in 1:K ]
    lsimilar1 = [ [ log_similarity(model.state.similarity, Xstats1[k][j], true) for j in 1:model.p ] for k in 1:K ]

    ## get cohesion and similarity for the extra cluster
    lcohes_newclust = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, 1, true))
    Xstats_newclust = [ Similarity_NiG_indep_stats([model.X[i,j]]) for j = 1:model.p ]
    lsimilar_newclust = [ log_similarity(model.state.similarity, Xstats_newclust[j], true) for j in 1:model.p ]

    ## calculate llik with obs i assigned to each cluster, including the extra
    llik0 = Vector{Float64}(undef, K)
    llik1 = Vector{Float64}(undef, K)
    for k in 1:K
        indx_k = findall(model.state.C.==k) ## currently, C[i] = 0, so obs i gets omitted from the calculation
        llik0[k] = llik_k(model.y[indx_k], model.X[indx_k,:], model.obsXIndx[indx_k], model.state.lik_params[k], model.state.Xstats[k], model.state.similarity)[1]
        indx_k_cand = vcat(indx_k, i)
        llik1[k] = llik_k(model.y[indx_k_cand], model.X[indx_k_cand,:], model.obsXIndx[indx_k_cand], model.state.lik_params[k], Xstats1[k], model.state.similarity)[1]
    end

    llik_wgts = Vector{Float64}(undef, K)
    sum_llik0 = sum(llik0)
    for k in 1:K
        llik_wgts[k] = sum_llik0 - llik0[k] + llik1[k]
    end

    lik_params_extra = simpri_lik_params(model.state.baseline, model.p)
    llik_wgts_newclust = sum_llik0 + llik_k(model.y[[i]], model.X[[i],:], [model.obsXIndx[i]], lik_params_extra, Xstats_newclust, model.state.similarity)[1]

    ## calculate weights
    lw = Vector{Float64}(undef, K + 1)
    for k in 1:K
        lw[k] = lcohesions1[k] + sum(lsimilar1[k]) - model.state.lcohesions[k] - sum(model.state.lsimilarities[k]) + llik_wgts[k]
    end
    
    ## weight for new singleton cluster
    lw[K + 1] = lcohes_newclust + sum(lsimilar_newclust) + llik_wgts_newclust

    ## sample membership
    lw = lw .- maximum(lw)
    C_out = StatsBase.sample(StatsBase.Weights(exp.(lw)))

    ## refresh model state to reflect update
    model.state.C[i] = C_out
    if C_out > K
        push!(model.state.lik_params, lik_params_extra)
        push!(model.state.lcohesions, lcohes_newclust)
        push!(model.state.Xstats, Xstats_newclust)
        push!(model.state.lsimilarities, lsimilar_newclust)
    else
        model.state.lcohesions[C_out] = deepcopy(lcohesions1[C_out])
        model.state.Xstats[C_out] = deepcopy(Xstats1[C_out])
        model.state.lsimilarities[C_out] = deepcopy(lsimilar1[C_out])
    end

    return nothing
end


function update_C!(model::Model_PPMx)
    for i in 1:model.n
        update_Ci!(model, i)
    end
    return nothing
end
# update_config.jl

function update_Ci!(model::Model_PPMx, i::Int, K::Int, S::Vector{Int}, llik_old::Vector{T}, update_lik_params::Vector{Symbol}=[:mu, :sig, :beta]) where T <: Real

    ## remove obs i from current C and modify current cohesions, Xstats, and similarities without obs i
    Ci_old = model.state.C[i]
    model.state.C[i] = 0
    wasnot_single = S[Ci_old] > 1

    if S[Ci_old] > 1

        S[Ci_old] -= 1

        model.state.lcohesions[Ci_old] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[Ci_old], true))

        for j in 1:model.p
            model.state.Xstats[Ci_old][j] = Similarity_stats(model.state.Xstats[Ci_old][j], model.X[i,j], :subtract)
            model.state.lsimilarities[Ci_old][j] = log_similarity(model.state.similarity, model.state.Xstats[Ci_old][j], true)
        end

    else ## collapse if C[i] was a singleton

        deleteat!(model.state.lcohesions, Ci_old)
        deleteat!(model.state.Xstats, Ci_old)
        deleteat!(model.state.lsimilarities, Ci_old)
        deleteat!(model.state.lik_params, Ci_old)
        deleteat!(S, Ci_old)
        deleteat!(llik_old, Ci_old)

        model.state.C[findall(model.state.C .> Ci_old)] .-= 1 # relabel all obs with label higher than the deleted one
        K -= 1

    end

    ## get cohesions, Xstats, and similarities each with obs i hypothetically added to each cluster
    lcohesions1 = [ log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[k] + 1, true)) for k in 1:K ]
    Xstats1 = [ [ Similarity_stats(model.state.Xstats[k][j], model.X[i,j], :add) for j in 1:model.p ] for k in 1:K ]
    lsimilar1 = [ [ log_similarity(model.state.similarity, Xstats1[k][j], true) for j in 1:model.p ] for k in 1:K ]

    ## get cohesion and similarity for the extra cluster
    lcohes_newclust = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, 1, true))
    Xstats_newclust = [ Similarity_stats(model.state.similarity, [model.X[i,j]]) for j = 1:model.p ]
    lsimilar_newclust = [ log_similarity(model.state.similarity, Xstats_newclust[j], true) for j in 1:model.p ]

    ## calculate llik with obs i assigned to each cluster, including the extra
    llik0 = deepcopy(llik_old)
    llik1 = Vector{Float64}(undef, K)
    for k in 1:K
        indx_k = findall(model.state.C.==k) ## currently, C[i] = 0, so obs i gets omitted from the calculation
        if wasnot_single && k == Ci_old
            llik0[k] = llik_k(model.y[indx_k], model.X[indx_k,:], model.obsXIndx[indx_k], model.state.lik_params[k], model.state.Xstats[k], model.state.similarity)[1]
            llik1[k] = deepcopy(llik_old[k])
        else
            indx_k_cand = vcat(indx_k, i)
            llik1[k] = llik_k(model.y[indx_k_cand], model.X[indx_k_cand,:], model.obsXIndx[indx_k_cand], model.state.lik_params[k], Xstats1[k], model.state.similarity)[1]
        end
    end

    llik_wgts = Vector{Float64}(undef, K)
    sum_llik0 = sum(llik0)
    for k in 1:K
        llik_wgts[k] = sum_llik0 - llik0[k] + llik1[k]
    end

    lik_params_extra = simpri_lik_params(model.state.baseline, model.p, model.state.lik_params[1], update_lik_params)
    llik_newclust = llik_k(model.y[[i]], model.X[[i],:], [model.obsXIndx[i]], lik_params_extra, Xstats_newclust, model.state.similarity)[1]
    llik_wgts_newclust = sum_llik0 + llik_newclust

    ## calculate weights
    lw = Vector{Float64}(undef, K + 1)
    for k in 1:K
        lw[k] = lcohesions1[k] + sum(lsimilar1[k]) - model.state.lcohesions[k] - sum(model.state.lsimilarities[k]) + llik_wgts[k]
    end

    ## weight for new singleton cluster
    lw[K + 1] = lcohes_newclust + sum(lsimilar_newclust) + llik_wgts_newclust

    ## sample membership
    lw = lw .- maximum(lw)
    Ci_out = StatsBase.sample(StatsBase.Weights(exp.(lw)))

    ## continuing log likelihood
    llik_out = deepcopy(llik0)

    ## refresh model state to reflect update
    model.state.C[i] = Ci_out
    if Ci_out > K
        push!(model.state.lik_params, lik_params_extra)
        push!(model.state.lcohesions, lcohes_newclust)
        push!(model.state.Xstats, Xstats_newclust)
        push!(model.state.lsimilarities, lsimilar_newclust)
        push!(llik_out, llik_newclust)
        K += 1
        push!(S, 1)
    else
        model.state.lcohesions[Ci_out] = deepcopy(lcohesions1[Ci_out])
        model.state.Xstats[Ci_out] = deepcopy(Xstats1[Ci_out])
        model.state.lsimilarities[Ci_out] = deepcopy(lsimilar1[Ci_out])
        llik_out[Ci_out] = deepcopy(llik1[Ci_out])
        S[Ci_out] += 1
    end

    return llik_out, K, S
end


function update_Ci_MH!(model::Model_PPMx, i::Int, K::Int, S::Vector{Int}, llik_vec::Vector{T},
    update_lik_params::Vector{Symbol}=[:mu, :sig, :beta], pswap = 0.5) where T <: Real

    ### Adaptation of Neal's (2000) Algorithm 7

    Ci_old = deepcopy(model.state.C[i])

    if rand() < pswap # single -> group or swap group -> single

        if S[Ci_old] == 1 # if i is a singleton, propose to move to one of the occupied groups

            ks_use = setdiff(1:K, Ci_old)

            ## get cohesions, Xstats, and similarities each with obs i hypothetically added to each cluster
            lcohesions1 = deepcopy(model.state.lcohesions)
            Xstats1 = deepcopy(model.state.Xstats)
            lsimilar1 = deepcopy(model.state.lsimilarities)

            lcg_ratios = Vector{Float64}(undef, K)
            for k in ks_use
                lcohesions1[k] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[k] + 1, true))
                Xstats1[k] = [ Similarity_stats(Xstats1[k][j], model.X[i,j], :add) for j in 1:model.p ]
                lsimilar1[k] = [ log_similarity(model.state.similarity, Xstats1[k][j], true) for j in 1:model.p ]
                lcg_ratios[k] = lcohesions1[k] + sum(lsimilar1[k]) - model.state.lcohesions[k] - sum(model.state.lsimilarities[k])
            end
            lsum_cgratios = logsumexp(lcg_ratios[ks_use])

            ## sample candidate membership
            lw = lcg_ratios .- maximum(lcg_ratios[ks_use])
            lw[Ci_old] = -Inf
            Ci_cand = StatsBase.sample(StatsBase.Weights(exp.(lw)))

            ## compute acceptance probability
            lcg_single = model.state.lcohesions[Ci_old] + sum(model.state.lsimilarities[Ci_old])

            indx_Ci_cand = findall(model.state.C .== Ci_cand)
            indx_Ci_cand = vcat(indx_Ci_cand, i)
            llik_Ci_cand = llik_k(model.y[indx_Ci_cand], model.X[indx_Ci_cand,:], model.obsXIndx[indx_Ci_cand], model.state.lik_params[Ci_cand], Xstats1[Ci_cand], model.state.similarity)[1]
            llik_as_single = llik_vec[Ci_cand] + llik_vec[Ci_old]

            lp_accpt = lsum_cgratios + llik_Ci_cand - lcg_single - llik_as_single

            ## determine whether to accept candidate and update
            accpt = log(rand()) < lp_accpt
            if accpt
                model.state.C[i] = deepcopy(Ci_cand)
                model.state.C[findall(model.state.C .> Ci_old)] .-= 1 # relabel all obs with label higher than the deleted one
                Ci_out = deepcopy(model.state.C[i])

                deleteat!(model.state.lcohesions, Ci_old)
                deleteat!(model.state.Xstats, Ci_old)
                deleteat!(model.state.lsimilarities, Ci_old)
                deleteat!(model.state.lik_params, Ci_old)
                deleteat!(llik_vec, Ci_old)

                model.state.lcohesions[Ci_out] = deepcopy(lcohesions1[Ci_cand]) # Ci_out and state indexed after singleton deleted; lcohesions1 and Ci_cand indexed before singleton deleted
                model.state.Xstats[Ci_out] = deepcopy(Xstats1[Ci_cand])
                model.state.lsimilarities[Ci_out] = deepcopy(lsimilar1[Ci_cand])
                llik_vec[Ci_out] = llik_Ci_cand
                K -= 1
                S[Ci_cand] += 1
                deleteat!(S, Ci_old)
                # length(S) == K || throw("Error in swap single to group, K=$(K) and S=$(S)")
            end

        else # if i has company in its cluster, propose a singleton

            ## get cohesion, similarity, llik for the extra cluster
            lcohes_newclust = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, 1, true))
            Xstats_newclust = [ Similarity_stats(model.state.similarity, [model.X[i,j]]) for j = 1:model.p ]
            lsimilar_newclust = [ log_similarity(model.state.similarity, Xstats_newclust[j], true) for j in 1:model.p ]

            lcg_single = lcohes_newclust + sum(lsimilar_newclust)

            lik_params_extra = simpri_lik_params(model.state.baseline, model.p, model.state.lik_params[1], update_lik_params)
            llik_newclust = llik_k(model.y[[i]], model.X[[i],:], [model.obsXIndx[i]], lik_params_extra, Xstats_newclust, model.state.similarity)[1]

            ## get cohesions, Xstats, and similarities each with obs i hypothetically in or out of each cluster
            lcohesions0 = deepcopy(model.state.lcohesions)
            Xstats0 = deepcopy(model.state.Xstats)
            lsimilar0 = deepcopy(model.state.lsimilarities)

            lcohesions0[Ci_old] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[Ci_old] - 1, true))
            for j in 1:model.p
                Xstats0[Ci_old][j] = Similarity_stats(Xstats0[Ci_old][j], model.X[i,j], :subtract)
                lsimilar0[Ci_old][j] = log_similarity(model.state.similarity, Xstats0[Ci_old][j], true)
            end

            ks_other = setdiff(1:K, Ci_old)

            lcohesions1 = deepcopy(model.state.lcohesions)
            Xstats1 = deepcopy(model.state.Xstats)
            lsimilar1 = deepcopy(model.state.lsimilarities)

            lcg_ratios = Vector{Float64}(undef, K)
            for k in ks_other
                lcohesions1[k] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[k] + 1, true))
                Xstats1[k] = [ Similarity_stats(Xstats1[k][j], model.X[i,j], :add) for j in 1:model.p ]
                lsimilar1[k] = [ log_similarity(model.state.similarity, Xstats1[k][j], true) for j in 1:model.p ]
                lcg_ratios[k] = lcohesions1[k] + sum(lsimilar1[k]) - lcohesions0[k] - sum(lsimilar0[k])
            end
            lcg_ratios[Ci_old] = lcohesions1[Ci_old] + sum(lsimilar1[Ci_old]) - lcohesions0[Ci_old] - sum(lsimilar0[Ci_old])

            lsum_cgratios = logsumexp(lcg_ratios)

            ## compute acceptance probability
            indx_Ci_wo_i = findall(model.state.C .== Ci_old)
            indx_Ci_wo_i = setdiff(indx_Ci_wo_i, i)
            # if length(indx_Ci_wo_i) == 0
            #     println("K=$(K), S=$(S), Ci_cand=new, Ci_old=$(Ci_old), length(S)=$(length(S)), sum(S)=$(sum(S))")
            #     S_now = StatsBase.counts(model.state.C, 1:maximum(model.state.C))
            #     println("Scounts_now=$(S_now) with length $(length(S_now)) and sum $(sum(S_now))")
            #     println("S and Scounts_now equal? $(all(S .== S_now))")
            #     println("S minus Scounts_now=$(S-S_now)")
            # end
            llik_Ci_wo_i = llik_k(model.y[indx_Ci_wo_i], model.X[indx_Ci_wo_i,:], model.obsXIndx[indx_Ci_wo_i],
                                  model.state.lik_params[Ci_old], Xstats0[Ci_old], model.state.similarity)[1]
            llik_as_single = llik_Ci_wo_i + llik_newclust

            lp_accpt = lcg_single + llik_as_single - lsum_cgratios - llik_vec[Ci_old]

            ## determine whether to accept candidate and update
            accpt = log(rand()) < lp_accpt
            if accpt
                model.state.C[i] = K + 1

                push!(model.state.lik_params, lik_params_extra)
                push!(model.state.lcohesions, lcohes_newclust)
                push!(model.state.Xstats, Xstats_newclust)
                push!(model.state.lsimilarities, lsimilar_newclust)
                push!(llik_vec, llik_newclust)

                model.state.lcohesions[Ci_old] = deepcopy(lcohesions0[Ci_old])
                model.state.Xstats[Ci_old] = deepcopy(Xstats0[Ci_old])
                model.state.lsimilarities[Ci_old] = deepcopy(lsimilar0[Ci_old])
                llik_vec[Ci_old] = llik_Ci_wo_i

                K += 1
                S[Ci_old] -= 1
                push!(S, 1)
                # length(S) == K || throw("Error in swap group to single, K=$(K) and S=$(S)")
            end

        end

    else # move between occupied clusters

        if S[Ci_old] > 1 && K > 1 # only propose a move if not a singleton

            ## get cohesions, Xstats, and similarities each with obs i hypothetically in or out of each cluster
            lcohesions0 = deepcopy(model.state.lcohesions)
            Xstats0 = deepcopy(model.state.Xstats)
            lsimilar0 = deepcopy(model.state.lsimilarities)

            lcohesions0[Ci_old] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[Ci_old] - 1, true))
            for j in 1:model.p
                Xstats0[Ci_old][j] = Similarity_stats(Xstats0[Ci_old][j], model.X[i,j], :subtract)
                lsimilar0[Ci_old][j] = log_similarity(model.state.similarity, Xstats0[Ci_old][j], true)
            end

            ks_notCi_old = setdiff(1:K, Ci_old)

            lcohesions1 = deepcopy(model.state.lcohesions)
            Xstats1 = deepcopy(model.state.Xstats)
            lsimilar1 = deepcopy(model.state.lsimilarities)

            lcg_ratios = Vector{Float64}(undef, K)
            for k in ks_notCi_old
                lcohesions1[k] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[k] + 1, true))
                Xstats1[k] = [ Similarity_stats(Xstats1[k][j], model.X[i,j], :add) for j in 1:model.p ]
                lsimilar1[k] = [ log_similarity(model.state.similarity, Xstats1[k][j], true) for j in 1:model.p ]
                lcg_ratios[k] = lcohesions1[k] + sum(lsimilar1[k]) - lcohesions0[k] - sum(lsimilar0[k])
            end
            lcg_ratios[Ci_old] = lcohesions1[Ci_old] + sum(lsimilar1[Ci_old]) - lcohesions0[Ci_old] - sum(lsimilar0[Ci_old])

            ## draw candidate
            lw_cand = deepcopy(lcg_ratios)
            lw_cand[Ci_old] = -Inf
            lw_cand = lw_cand .- maximum(lw_cand)

            Ci_cand = StatsBase.sample(StatsBase.Weights(exp.(lw_cand)))

            ## calculate acceptance prob
            ks_not_Ci_cand = setdiff(1:K, Ci_cand)

            lsum_cgratios_num = logsumexp(lcg_ratios[ks_notCi_old])
            lsum_cgratios_denom = logsumexp(lcg_ratios[ks_not_Ci_cand])


            llik_Ci_old_w = llik_vec[Ci_old]

            indx_Ci_old_wo = findall(model.state.C .== Ci_old)
            indx_Ci_old_wo = setdiff(indx_Ci_old_wo, i)
            # if length(indx_Ci_old_wo) == 0
            #     println("K=$(K), S=$(S), Ci_cand=$(Ci_cand), Ci_old=$(Ci_old), length(S)=$(length(S)), sum(S)=$(sum(S))")
            #     S_now = StatsBase.counts(model.state.C, 1:maximum(model.state.C))
            #     println("Scounts_now=$(S_now) with length $(length(S_now)) and sum $(sum(S_now))")
            #     println("S and Scounts_now equal? $(all(S .== S_now))")
            #     println("S minus Scounts_now=$(S - S_now)")
            # end
            llik_Ci_old_wo = llik_k(model.y[indx_Ci_old_wo], model.X[indx_Ci_old_wo,:], model.obsXIndx[indx_Ci_old_wo],
                                  model.state.lik_params[Ci_old], Xstats0[Ci_old], model.state.similarity)[1]

            llik_Ci_cand_wo = llik_vec[Ci_cand]

            indx_Ci_cand_w = findall(model.state.C .== Ci_cand)
            indx_Ci_cand_w = push!(indx_Ci_cand_w, i)
            llik_Ci_cand_w = llik_k(model.y[indx_Ci_cand_w], model.X[indx_Ci_cand_w,:], model.obsXIndx[indx_Ci_cand_w],
                                  model.state.lik_params[Ci_cand], Xstats1[Ci_cand], model.state.similarity)[1]

            lp_accpt = lsum_cgratios_num + llik_Ci_cand_w + llik_Ci_old_wo - lsum_cgratios_denom - llik_Ci_cand_wo - llik_Ci_old_w

            ## determine whether to accept candidate and update
            accpt = log(rand()) < lp_accpt
            if accpt
                model.state.C[i] = deepcopy(Ci_cand)

                model.state.lcohesions[Ci_cand] = deepcopy(lcohesions1[Ci_cand])
                model.state.Xstats[Ci_cand] = deepcopy(Xstats1[Ci_cand])
                model.state.lsimilarities[Ci_cand] = deepcopy(lsimilar1[Ci_cand])
                llik_vec[Ci_cand] = llik_Ci_cand_w

                model.state.lcohesions[Ci_old] = deepcopy(lcohesions0[Ci_old])
                model.state.Xstats[Ci_old] = deepcopy(Xstats0[Ci_old])
                model.state.lsimilarities[Ci_old] = deepcopy(lsimilar0[Ci_old])
                llik_vec[Ci_old] = llik_Ci_old_wo

                S[Ci_cand] += 1
                S[Ci_old] -= 1
                # length(S) == K || throw("Error in move between occupied clusters, K=$(K) and S=$(S)")
            end

        end

    end

    return llik_vec, K, S
end




function update_C!(model::Model_PPMx, update_lik_params::Vector{Symbol}=[:mu, :sig, :beta])

    K = length(model.state.lik_params)
    S = StatsBase.counts(model.state.C, K)

    llik_now = Vector{Float64}(undef, K)

    for k in 1:K
        indx_k = findall(model.state.C.==k)
        llik_now[k] = llik_k(model.y[indx_k], model.X[indx_k,:], model.obsXIndx[indx_k], model.state.lik_params[k], model.state.Xstats[k], model.state.similarity)[1]
    end

    for i in 1:model.n
        # llik_now, K, S = update_Ci!(model, i, K, S, llik_now, update_lik_params)  # full Gibbs, Algo 8
        llik_now, K, S = update_Ci_MH!(model, i, K, S, llik_now, update_lik_params)  # MH, Algo 7
    end

    return nothing
end

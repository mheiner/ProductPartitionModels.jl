# update_config.jl

function update_Ci!(model::Model_PPMx, i::Int, K::Int, S::Vector{Int}, 
    llik_old::Vector{T}, update_lik_params::Vector{Symbol}=[:mu, :sig, :beta],
    M_newclust::Int=10
    ) where T <: Real

    # FIXME This algorithm should remove obs i, but if it's a singleton, make the old params one of the extras (needs to be able to return to original state); 
    ## do not use until corrected.
    ## Currently implemented ONLY for the :Reg type model and not the :Mean type

    ## remove obs i from current C and modify current cohesions, Xstats, and similarities without obs i
    Ci_old = model.state.C[i]
    model.state.C[i] = 0
    wasnot_single = S[Ci_old] > 1
    # S[Ci_old] -= 1

    ## cohesions, Xstats, and similarities each with obs i hypothetically in (1) or not in (0) each cluster

    lcohesions0 = deepcopy(model.state.lcohesions)
    Xstats0 = deepcopy(model.state.Xstats)
    lsimilar0 = deepcopy(model.state.lsimilarities)

    lcohesions1 = deepcopy(model.state.lcohesions)
    Xstats1 = deepcopy(model.state.Xstats)
    lsimilar1 = deepcopy(model.state.lsimilarities)
    
    ## get cohesion and similarity for the extra cluster(s)
    lcohes_newclust = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, 1, true))
    Xstats_newclust = [ Similarity_stats(model.state.similarity, [model.X[i,j]]) for j = 1:model.p ]
    lsimilar_newclust = [ log_similarity(model.state.similarity, Xstats_newclust[j], true) for j in 1:model.p ]

    # do the above in the style of llik calculations below
    for k in 1:K
        if wasnot_single && (k == Ci_old)

            lcohesions0[k] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[k] - 1, true))
            # lcohesions1[k] is the old value that was copied

            for j in 1:model.p
                Xstats0[k][j] = Similarity_stats(model.state.Xstats[k][j], model.X[i,j], :subtract)
                lsimilar0[k][j] = log_similarity(model.state.similarity, Xstats0[k][j], true)

                # Xstats1[k][j] is the old value that was copied
                # lsimilar1[k][j] is the old value that was copied
            end

        elseif (k == Ci_old) # was single and k == Ci_old

            lcohesions0[k] = 0.0
            # lcohesions1[k] is the old value that was copied

            for j in 1:model.p
                # Xstats0[k][j] will go unmodified because it is not needed here
                lsimilar0[k][j] = 0.0

                # Xstats1[k][j] is the old value that was copied
                # lsimilar1[k][j] is the old value that was copied
            end

        else # k != Ci_old

            # lcohesions0[k] is the old value that was copied
            lcohesions1[k] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[k] + 1, true))

            for j in 1:model.p
                # Xstats0[k][j] is the old value that was copied
                # lsimilar0[k][j] is the old value that was copied

                Xstats1[k][j] = Similarity_stats(model.state.Xstats[k][j], model.X[i,j], :add)
                lsimilar1[k][j] = log_similarity(model.state.similarity, Xstats1[k][j], true)
            end

        end
    end

    ## calculate llik with obs i assigned to each cluster, including the extra
    llik0 = deepcopy(llik_old)
    llik1 = Vector{Float64}(undef, K)
    for k in 1:K
        indx_k = findall(model.state.C .== k) ## currently, C[i] = 0, so obs i gets omitted from the calculation
        if wasnot_single && (k == Ci_old)
            llik0[k] = llik_k(model.y[indx_k], model.X[indx_k,:], model.obsXIndx[indx_k], model.state.lik_params[k], Xstats0[k], model.state.similarity)[:llik]
            llik1[k] = deepcopy(llik_old[k])
        elseif (k == Ci_old)
            llik0[k] = 0.0 ## is this right?
            llik1[k] = deepcopy(llik_old[k])
        else # k != Ci_old
            indx_k_cand = vcat(indx_k, i)
            llik1[k] = llik_k(model.y[indx_k_cand], model.X[indx_k_cand,:], model.obsXIndx[indx_k_cand], model.state.lik_params[k], Xstats1[k], model.state.similarity)[:llik]
        end
    end

    llik_wgts = Vector{Float64}(undef, K)
    sum_llik0 = sum(llik0)
    for k in 1:K
        llik_wgts[k] = sum_llik0 - llik0[k] + llik1[k]
    end

    lik_params_extra = [ simpri_lik_params(model.state.baseline, model.p, model.state.lik_params[1], update_lik_params) for kk in 1:M_newclust ]
    llik_newclust = [ llik_k(model.y[[i]], model.X[[i],:], [model.obsXIndx[i]], lik_params_extra[kk], Xstats_newclust, model.state.similarity)[:llik] for kk in 1:M_newclust ]
    llik_wgts_newclust = sum_llik0 .+ llik_newclust

    ## calculate weights
    lw = Vector{Float64}(undef, K + M_newclust)
    for k in 1:K
        lw[k] = lcohesions1[k] + sum(lsimilar1[k]) - lcohesions0[k] - sum(lsimilar0[k]) + llik_wgts[k]
    end

    ## weight for new singleton cluster(s)
    logM_nc = log(M_newclust)
    for kk in 1:M_newclust
        lw[K + kk] = lcohes_newclust + sum(lsimilar_newclust) + llik_wgts_newclust[kk] - logM_nc
    end

    ## sample membership
    lw = lw .- maximum(lw)
    Ci_out = StatsBase.sample(StatsBase.Weights(exp.(lw)))
    
    if Ci_out > K
        which_newclust = Ci_out - K
        Ci_out = K + 1
    end

    ## continuing log likelihood
    llik_out = deepcopy(llik0)

    ## refresh model state to reflect update
    model.state.C[i] = Ci_out

    if (Ci_out != Ci_old) && (Ci_out <= K) # if moving within existing clusters, migrate to new cluster
        model.state.lcohesions[Ci_out] = deepcopy(lcohesions1[Ci_out])
        model.state.Xstats[Ci_out] = deepcopy(Xstats1[Ci_out])
        model.state.lsimilarities[Ci_out] = deepcopy(lsimilar1[Ci_out])
        llik_out[Ci_out] = deepcopy(llik1[Ci_out])
        S[Ci_out] += 1
    end

    if Ci_out > K # if moving to a newly created cluster, migrate there
        push!(model.state.lik_params, lik_params_extra[which_newclust])
        push!(model.state.lcohesions, lcohes_newclust)
        push!(model.state.Xstats, Xstats_newclust)
        push!(model.state.lsimilarities, lsimilar_newclust)
        push!(llik_out, llik_newclust[which_newclust])
        K += 1
        push!(S, 1)
    end

    if (!wasnot_single) && (Ci_out != Ci_old) ## if C[i] was a singleton and changed groups, collapse    
        deleteat!(model.state.lcohesions, Ci_old)
        deleteat!(model.state.Xstats, Ci_old)
        deleteat!(model.state.lsimilarities, Ci_old)
        deleteat!(model.state.lik_params, Ci_old)
        deleteat!(S, Ci_old)
        deleteat!(llik_old, Ci_old)

        model.state.C[findall(model.state.C .> Ci_old)] .-= 1 # relabel all obs with label higher than the deleted one
        Ci_out = model.state.C[i]
        K -= 1
    end

    if wasnot_single && (Ci_out != Ci_old) # if was not single and changed groups, remove from old cluster (also, there was no collapse if wasnot_single, so we can use Ci_old)
        model.state.lcohesions[Ci_old] = deepcopy(lcohesions0[Ci_old])
        model.state.Xstats[Ci_old] = deepcopy(Xstats0[Ci_old])  ## Xstats0[Ci_old] can't come from a collapsed singleton, so this is ok
        model.state.lsimilarities[Ci_old] = deepcopy(lsimilar0[Ci_old])
        ## llik_out was copied from llik0, so no change to that
        S[Ci_old] -= 1
    end

    ## otherwise, the unit was not reallocated, so change nothing

    return llik_out, K, S
end


function update_Ci_MH_Reg!(model::Model_PPMx, i::Int, K::Int, S::Vector{Int}, llik_vec::Vector{T},
    update_lik_params::Vector{Symbol}=[:mu, :sig, :beta], pswap = 0.5) where T <: Real

    ### Adaptation of Neal's (2000) Algorithm 7

    Ci_old = model.state.C[i]

    if rand() < pswap # single -> group or swap group -> single

        if S[Ci_old] == 1 # if i is a singleton, propose to move to one of the occupied groups

            ks_use = collect(1:K)
            deleteat!(ks_use, Ci_old)

            ## get cohesions, Xstats, and similarities each with obs i hypothetically added to each cluster
            lcohesions1 = Vector{typeof(model.state.lcohesions[1])}(undef, K)
            Xstats1 = Vector{typeof(model.state.Xstats[1])}(undef, K)
            lsimilar1 = Vector{typeof(model.state.lsimilarities[1])}(undef, K)

            lcg_ratios = Vector{Float64}(undef, K)
            for k in ks_use
                lcohesions1[k] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[k] + 1, true))
                Xstats1[k] = [ Similarity_stats(model.state.Xstats[k][j], model.X[i,j], :add) for j in 1:model.p ]
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
            llik_Ci_cand = llik_k(model.y[indx_Ci_cand], model.X[indx_Ci_cand,:], model.obsXIndx[indx_Ci_cand], model.state.lik_params[Ci_cand], Xstats1[Ci_cand], model.state.similarity)[:llik]
            llik_as_single = llik_vec[Ci_cand] + llik_vec[Ci_old]

            lp_accpt = lsum_cgratios + llik_Ci_cand - lcg_single - llik_as_single

            ## determine whether to accept candidate and update
            accpt = log(rand()) < lp_accpt
            if accpt
                model.state.C[i] = Ci_cand
                model.state.C[findall(model.state.C .> Ci_old)] .-= 1 # relabel all obs with label higher than the deleted one
                Ci_out = model.state.C[i]

                deleteat!(model.state.lcohesions, Ci_old)
                deleteat!(model.state.Xstats, Ci_old)
                deleteat!(model.state.lsimilarities, Ci_old)
                deleteat!(model.state.lik_params, Ci_old)
                deleteat!(llik_vec, Ci_old)

                model.state.lcohesions[Ci_out] = lcohesions1[Ci_cand] # Ci_out and state indexed after singleton deleted; lcohesions1 and Ci_cand indexed before singleton deleted
                model.state.Xstats[Ci_out] = Xstats1[Ci_cand]
                model.state.lsimilarities[Ci_out] = lsimilar1[Ci_cand]
                llik_vec[Ci_out] = llik_Ci_cand
                K -= 1
                S[Ci_cand] += 1
                deleteat!(S, Ci_old)
            end

        else # if i has company in its cluster, propose a singleton

            ## get cohesion, similarity, llik for the extra cluster
            lcohes_newclust = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, 1, true))
            Xstats_newclust = [ Similarity_stats(model.state.similarity, [model.X[i,j]]) for j = 1:model.p ]
            lsimilar_newclust = [ log_similarity(model.state.similarity, Xstats_newclust[j], true) for j in 1:model.p ]

            lcg_single = lcohes_newclust + sum(lsimilar_newclust)

            lik_params_extra = simpri_lik_params(model.state.baseline, model.p, model.state.lik_params[1], update_lik_params)
            llik_newclust = llik_k(model.y[[i]], model.X[[i],:], [model.obsXIndx[i]], lik_params_extra, Xstats_newclust, model.state.similarity)[:llik]

            ## get cohesions, Xstats, and similarities each with obs i hypothetically in or out of each cluster
            lcohesions0 = deepcopy(model.state.lcohesions)
            lsimilar0 = deepcopy(model.state.lsimilarities)

            lcohesions0[Ci_old] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[Ci_old] - 1, true))
            Xstats0_Ci_old = [ Similarity_stats(model.state.Xstats[Ci_old][j], model.X[i,j], :subtract) for j in 1:model.p ]
            for j in 1:model.p
                lsimilar0[Ci_old][j] = log_similarity(model.state.similarity, Xstats0_Ci_old[j], true)
            end

            ks_other = collect(1:K)
            deleteat!(ks_other, Ci_old)

            lcohesions1 = Vector{typeof(model.state.lcohesions[1])}(undef, K)
            Xstats1 = Vector{typeof(model.state.Xstats[1])}(undef, K)
            lsimilar1 = Vector{typeof(model.state.lsimilarities[1])}(undef, K)

            lcg_ratios = Vector{Float64}(undef, K)
            for k in ks_other
                lcohesions1[k] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[k] + 1, true))
                Xstats1[k] = [ Similarity_stats(model.state.Xstats[k][j], model.X[i,j], :add) for j in 1:model.p ]
                lsimilar1[k] = [ log_similarity(model.state.similarity, Xstats1[k][j], true) for j in 1:model.p ]
                lcg_ratios[k] = lcohesions1[k] + sum(lsimilar1[k]) - lcohesions0[k] - sum(lsimilar0[k])
            end
            lcg_ratios[Ci_old] = model.state.lcohesions[Ci_old] + sum(model.state.lsimilarities[Ci_old]) - lcohesions0[Ci_old] - sum(lsimilar0[Ci_old])

            lsum_cgratios = logsumexp(lcg_ratios)

            ## compute acceptance probability
            indx_Ci_wo_i = findall(model.state.C .== Ci_old)
            indx_Ci_wo_i = setdiff(indx_Ci_wo_i, i)
            llik_Ci_wo_i = llik_k(model.y[indx_Ci_wo_i], model.X[indx_Ci_wo_i,:], model.obsXIndx[indx_Ci_wo_i],
                                  model.state.lik_params[Ci_old], Xstats0_Ci_old, model.state.similarity)[:llik]
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

                model.state.lcohesions[Ci_old] = lcohesions0[Ci_old]
                model.state.Xstats[Ci_old] = Xstats0_Ci_old
                model.state.lsimilarities[Ci_old] = lsimilar0[Ci_old]
                llik_vec[Ci_old] = llik_Ci_wo_i

                K += 1
                S[Ci_old] -= 1
                push!(S, 1)
            end

        end

    else # move between occupied clusters

        if S[Ci_old] > 1 && K > 1 # only propose a move if not a singleton

            ## get cohesions, Xstats, and similarities each with obs i hypothetically in or out of each cluster
            lcohesions0 = deepcopy(model.state.lcohesions)
            lsimilar0 = deepcopy(model.state.lsimilarities)

            lcohesions0[Ci_old] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[Ci_old] - 1, true))
            Xstats0_Ci_old = [ Similarity_stats(model.state.Xstats[Ci_old][j], model.X[i,j], :subtract) for j in 1:model.p ]
            for j in 1:model.p
                lsimilar0[Ci_old][j] = log_similarity(model.state.similarity, Xstats0_Ci_old[j], true)
            end

            ks_notCi_old = collect(1:K)
            deleteat!(ks_notCi_old, Ci_old)

            lcohesions1 = Vector{typeof(model.state.lcohesions[1])}(undef, K)
            Xstats1 = Vector{typeof(model.state.Xstats[1])}(undef, K)
            lsimilar1 = Vector{typeof(model.state.lsimilarities[1])}(undef, K)

            lcg_ratios = Vector{Float64}(undef, K)
            for k in ks_notCi_old
                lcohesions1[k] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[k] + 1, true))
                Xstats1[k] = [ Similarity_stats(model.state.Xstats[k][j], model.X[i,j], :add) for j in 1:model.p ]
                lsimilar1[k] = [ log_similarity(model.state.similarity, Xstats1[k][j], true) for j in 1:model.p ]
                lcg_ratios[k] = lcohesions1[k] + sum(lsimilar1[k]) - lcohesions0[k] - sum(lsimilar0[k])
            end
            lcg_ratios[Ci_old] = model.state.lcohesions[Ci_old] + sum(model.state.lsimilarities[Ci_old]) - lcohesions0[Ci_old] - sum(lsimilar0[Ci_old])

            ## draw candidate
            lw_cand = deepcopy(lcg_ratios)
            lw_cand[Ci_old] = -Inf
            lw_cand = lw_cand .- maximum(lw_cand)

            Ci_cand = StatsBase.sample(StatsBase.Weights(exp.(lw_cand)))

            ## calculate acceptance prob
            ks_not_Ci_cand = collect(1:K)
            deleteat!(ks_not_Ci_cand, Ci_cand)

            lsum_cgratios_num = logsumexp(lcg_ratios[ks_notCi_old])
            lsum_cgratios_denom = logsumexp(lcg_ratios[ks_not_Ci_cand])


            llik_Ci_old_w = llik_vec[Ci_old]

            indx_Ci_old_wo = findall(model.state.C .== Ci_old)
            indx_Ci_old_wo = setdiff(indx_Ci_old_wo, i)
            llik_Ci_old_wo = llik_k(model.y[indx_Ci_old_wo], model.X[indx_Ci_old_wo,:], model.obsXIndx[indx_Ci_old_wo],
                                  model.state.lik_params[Ci_old], Xstats0_Ci_old, model.state.similarity)[:llik]

            llik_Ci_cand_wo = llik_vec[Ci_cand]

            indx_Ci_cand_w = findall(model.state.C .== Ci_cand)
            indx_Ci_cand_w = push!(indx_Ci_cand_w, i)
            llik_Ci_cand_w = llik_k(model.y[indx_Ci_cand_w], model.X[indx_Ci_cand_w,:], model.obsXIndx[indx_Ci_cand_w],
                                  model.state.lik_params[Ci_cand], Xstats1[Ci_cand], model.state.similarity)[:llik]

            lp_accpt = lsum_cgratios_num + llik_Ci_cand_w + llik_Ci_old_wo - lsum_cgratios_denom - llik_Ci_cand_wo - llik_Ci_old_w

            ## determine whether to accept candidate and update
            accpt = log(rand()) < lp_accpt
            if accpt
                model.state.C[i] = Ci_cand

                model.state.lcohesions[Ci_cand] = lcohesions1[Ci_cand]
                model.state.Xstats[Ci_cand] = Xstats1[Ci_cand]
                model.state.lsimilarities[Ci_cand] = lsimilar1[Ci_cand]
                llik_vec[Ci_cand] = llik_Ci_cand_w

                model.state.lcohesions[Ci_old] = lcohesions0[Ci_old]
                model.state.Xstats[Ci_old] = Xstats0_Ci_old
                model.state.lsimilarities[Ci_old] = lsimilar0[Ci_old]
                llik_vec[Ci_old] = llik_Ci_old_wo

                S[Ci_cand] += 1
                S[Ci_old] -= 1
            end

        end

    end

    return llik_vec, K, S
end

function update_Ci_MH_Mean!(model::Model_PPMx, i::Int, K::Int, S::Vector{Int}, llik_vec::Vector{T},
    update_lik_params::Vector{Symbol}=[:mu, :sig], pswap = 0.5) where T <: Real

    ### Adaptation of Neal's (2000) Algorithm 7

    Ci_old = model.state.C[i]

    if rand() < pswap # single -> group or swap group -> single

        if S[Ci_old] == 1 # if i is a singleton, propose to move to one of the occupied groups

            ks_use = collect(1:K)
            deleteat!(ks_use, Ci_old)

            ## get cohesions, Xstats, and similarities each with obs i hypothetically added to each cluster
            lcohesions1 = Vector{typeof(model.state.lcohesions[1])}(undef, K)
            Xstats1 = Vector{typeof(model.state.Xstats[1])}(undef, K)
            lsimilar1 = Vector{typeof(model.state.lsimilarities[1])}(undef, K)

            lcg_ratios = Vector{Float64}(undef, K)
            for k in ks_use
                lcohesions1[k] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[k] + 1, true))
                Xstats1[k] = [ Similarity_stats(model.state.Xstats[k][j], model.X[i,j], :add) for j in 1:model.p ]
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

            llik_Ci_cand = llik_k(model.y[[i]], model.state.lik_params[Ci_cand])[:llik]

            lp_accpt = lsum_cgratios + llik_Ci_cand - lcg_single - llik_vec[i]

            ## determine whether to accept candidate and update
            accpt = log(rand()) < lp_accpt
            if accpt
                model.state.C[i] = Ci_cand
                model.state.C[findall(model.state.C .> Ci_old)] .-= 1 # relabel all obs with label higher than the deleted one
                Ci_out = model.state.C[i]

                deleteat!(model.state.lcohesions, Ci_old)
                deleteat!(model.state.Xstats, Ci_old)
                deleteat!(model.state.lsimilarities, Ci_old)
                deleteat!(model.state.lik_params, Ci_old)

                model.state.lcohesions[Ci_out] = lcohesions1[Ci_cand] # Ci_out and state indexed after singleton deleted; lcohesions1 and Ci_cand indexed before singleton deleted
                model.state.Xstats[Ci_out] = Xstats1[Ci_cand]
                model.state.lsimilarities[Ci_out] = lsimilar1[Ci_cand]
                llik_vec[i] = llik_Ci_cand
                K -= 1
                S[Ci_cand] += 1
                deleteat!(S, Ci_old)
            end

        else # if i has company in its cluster, propose a singleton

            ## get cohesion, similarity, llik for the extra cluster
            lcohes_newclust = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, 1, true))
            Xstats_newclust = [ Similarity_stats(model.state.similarity, [model.X[i,j]]) for j = 1:model.p ]
            lsimilar_newclust = [ log_similarity(model.state.similarity, Xstats_newclust[j], true) for j in 1:model.p ]

            lcg_single = lcohes_newclust + sum(lsimilar_newclust)

            lik_params_extra = simpri_lik_params(model.state.baseline, model.p, model.state.lik_params[1], update_lik_params)
            llik_newclust = llik_k(model.y[[i]], lik_params_extra)[:llik]

            ## get cohesions, Xstats, and similarities each with obs i hypothetically in or out of each cluster
            lcohesions0 = deepcopy(model.state.lcohesions)
            lsimilar0 = deepcopy(model.state.lsimilarities)

            lcohesions0[Ci_old] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[Ci_old] - 1, true))
            Xstats0_Ci_old = [ Similarity_stats(model.state.Xstats[Ci_old][j], model.X[i,j], :subtract) for j in 1:model.p ]
            for j in 1:model.p
                lsimilar0[Ci_old][j] = log_similarity(model.state.similarity, Xstats0_Ci_old[j], true)
            end

            ks_other = collect(1:K)
            deleteat!(ks_other, Ci_old)

            lcohesions1 = Vector{typeof(model.state.lcohesions[1])}(undef, K)
            Xstats1 = Vector{typeof(model.state.Xstats[1])}(undef, K)
            lsimilar1 = Vector{typeof(model.state.lsimilarities[1])}(undef, K)

            lcg_ratios = Vector{Float64}(undef, K)
            for k in ks_other
                lcohesions1[k] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[k] + 1, true))
                Xstats1[k] = [ Similarity_stats(model.state.Xstats[k][j], model.X[i,j], :add) for j in 1:model.p ]
                lsimilar1[k] = [ log_similarity(model.state.similarity, Xstats1[k][j], true) for j in 1:model.p ]
                lcg_ratios[k] = lcohesions1[k] + sum(lsimilar1[k]) - lcohesions0[k] - sum(lsimilar0[k])
            end
            lcg_ratios[Ci_old] = model.state.lcohesions[Ci_old] + sum(model.state.lsimilarities[Ci_old]) - lcohesions0[Ci_old] - sum(lsimilar0[Ci_old])

            lsum_cgratios = logsumexp(lcg_ratios)

            ## compute acceptance probability
            llik_as_single = llik_newclust

            lp_accpt = lcg_single + llik_as_single - lsum_cgratios - llik_vec[i]

            ## determine whether to accept candidate and update
            accpt = log(rand()) < lp_accpt
            if accpt
                model.state.C[i] = K + 1

                push!(model.state.lik_params, lik_params_extra)
                push!(model.state.lcohesions, lcohes_newclust)
                push!(model.state.Xstats, Xstats_newclust)
                push!(model.state.lsimilarities, lsimilar_newclust)

                model.state.lcohesions[Ci_old] = lcohesions0[Ci_old]
                model.state.Xstats[Ci_old] = Xstats0_Ci_old
                model.state.lsimilarities[Ci_old] = lsimilar0[Ci_old]
                llik_vec[i] = llik_as_single

                K += 1
                S[Ci_old] -= 1
                push!(S, 1)
            end

        end

    else # move between occupied clusters

        if S[Ci_old] > 1 && K > 1 # only propose a move if not a singleton

            ## get cohesions, Xstats, and similarities each with obs i hypothetically in or out of each cluster
            lcohesions0 = deepcopy(model.state.lcohesions)
            lsimilar0 = deepcopy(model.state.lsimilarities)

            lcohesions0[Ci_old] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[Ci_old] - 1, true))
            Xstats0_Ci_old = [ Similarity_stats(model.state.Xstats[Ci_old][j], model.X[i,j], :subtract) for j in 1:model.p ]
            for j in 1:model.p
                lsimilar0[Ci_old][j] = log_similarity(model.state.similarity, Xstats0_Ci_old[j], true)
            end

            ks_notCi_old = collect(1:K)
            deleteat!(ks_notCi_old, Ci_old)

            lcohesions1 = Vector{typeof(model.state.lcohesions[1])}(undef, K)
            Xstats1 = Vector{typeof(model.state.Xstats[1])}(undef, K)
            lsimilar1 = Vector{typeof(model.state.lsimilarities[1])}(undef, K)

            lcg_ratios = Vector{Float64}(undef, K)
            for k in ks_notCi_old
                lcohesions1[k] = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, S[k] + 1, true))
                Xstats1[k] = [ Similarity_stats(model.state.Xstats[k][j], model.X[i,j], :add) for j in 1:model.p ]
                lsimilar1[k] = [ log_similarity(model.state.similarity, Xstats1[k][j], true) for j in 1:model.p ]
                lcg_ratios[k] = lcohesions1[k] + sum(lsimilar1[k]) - lcohesions0[k] - sum(lsimilar0[k])
            end
            lcg_ratios[Ci_old] = model.state.lcohesions[Ci_old] + sum(model.state.lsimilarities[Ci_old]) - lcohesions0[Ci_old] - sum(lsimilar0[Ci_old])

            ## draw candidate
            lw_cand = deepcopy(lcg_ratios)
            lw_cand[Ci_old] = -Inf
            lw_cand = lw_cand .- maximum(lw_cand)

            Ci_cand = StatsBase.sample(StatsBase.Weights(exp.(lw_cand)))

            ## calculate acceptance prob
            ks_not_Ci_cand = collect(1:K)
            deleteat!(ks_not_Ci_cand, Ci_cand)

            lsum_cgratios_num = logsumexp(lcg_ratios[ks_notCi_old])
            lsum_cgratios_denom = logsumexp(lcg_ratios[ks_not_Ci_cand])

            llik_Ci_cand = llik_k(model.y[[i]], model.state.lik_params[Ci_cand])[:llik]

            lp_accpt = lsum_cgratios_num + llik_Ci_cand - lsum_cgratios_denom - llik_vec[i]

            ## determine whether to accept candidate and update
            accpt = log(rand()) < lp_accpt
            if accpt
                model.state.C[i] = Ci_cand

                model.state.lcohesions[Ci_cand] = lcohesions1[Ci_cand]
                model.state.Xstats[Ci_cand] = Xstats1[Ci_cand]
                model.state.lsimilarities[Ci_cand] = lsimilar1[Ci_cand]

                model.state.lcohesions[Ci_old] = lcohesions0[Ci_old]
                model.state.Xstats[Ci_old] = Xstats0_Ci_old
                model.state.lsimilarities[Ci_old] = lsimilar0[Ci_old]

                llik_vec[i] = llik_Ci_cand

                S[Ci_cand] += 1
                S[Ci_old] -= 1
            end

        end

    end

    return llik_vec, K, S
end



function update_C!(model::Model_PPMx, 
    update_lik_params::Vector{Symbol}=[:mu, :sig, :beta], 
    method::Symbol=:MH, M_newclust::Int=10)
    # method one of :MH, :FC (FC mot currently in use--must be corrected)

    K = length(model.state.lik_params)
    S = StatsBase.counts(model.state.C, K)

    llik_now = Vector{Float64}(undef, K)

    obs_ord = StatsBase.sample(1:model.n, model.n, replace=false)

    if typeof(model.state.lik_params[1]) <: LikParams_PPMxReg

        for k in 1:K
            indx_k = findall(model.state.C.==k)
            llik_now[k] = llik_k(model.y[indx_k], model.X[indx_k,:], model.obsXIndx[indx_k], model.state.lik_params[k], model.state.Xstats[k], model.state.similarity)[:llik]
        end

        if method == :MH
            for i in obs_ord
                llik_now, K, S = update_Ci_MH_Reg!(model, i, K, S, llik_now, update_lik_params)  # MH, Algo 7
            end
        elseif method == :FC
            for i in obs_ord
                llik_now, K, S = update_Ci!(model, i, K, S, llik_now, update_lik_params, M_newclust)  # full Gibbs, Algo 8
            end            
        end


    elseif typeof(model.state.lik_params[1]) <: LikParams_PPMxMean

        llik_now = llik_all(model.y, model.state.C, model.state.lik_params)[:llik_vec] # vector of length n

        if method == :MH
            for i in obs_ord
                llik_now, K, S = update_Ci_MH_Mean!(model, i, K, S, llik_now, update_lik_params)  # MH, Algo 7
            end
        elseif method == :FC
            for i in obs_ord
                llik_now, K, S = update_Ci!(model, i, K, S, llik_now, update_lik_params, M_newclust)  # full Gibbs, Algo 8
            end
        end
    end

    return nothing
end

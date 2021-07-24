# using Core: Matrix, Vector
# simulate_prior.jl

export sim_partition_PPMx, sim_lik;

function sim_partition_PPMx(logα::Real, X::Union{Matrix{T}, Matrix{Union{T, Missing}}} where T <: Real, 
    similarity::Similarity_PPMx)

    n, p = size(X)
    C = zeros(Int64, n)

    C[1] = 1 # cluster assinment of first obs
    K_now = 1 # current number of clusters
    S = [1] # cluster sizes
    lcohes1 = log_cohesion_CRP(logα, 1)

    lcohesions = [ lcohes1 ] # vector of log cohesions
    stats = [ [ Similarity_NiG_indep_stats([X[1,j]]) for j = 1:p ] ] # vector (for each clust) of vectors (for each x, 1:p) of sufficient statistics for similarity, obs 1 only so far
    lsimilarities = [ [ log_similarity(similarity, stats[1][j], true) for j in 1:p ] ] # vector (for each clust) of vectors (for each x, 1:p) of similarity scores

    if n > 1
        for i in 2:n

            lcohes_cand = deepcopy(lcohesions)
            stats_cand = deepcopy(stats)
            lsimilar_cand = deepcopy(lsimilarities)

            lw = Vector{Float64}(undef, K_now + 1)

            for k in 1:K_now # for each cluster

                # cohesion with obs i added
                lcohes_cand[k] = log_cohesion_CRP(logα, S[k] + 1)

                # stats for similarity with obs i added (each X[i,:], 1:p); similarity with obs i added
                for j in 1:p
                    stats_cand[k][j] = Similarity_NiG_indep_stats(stats[k][j], X[i,j], :add)
                    lsimilar_cand[k][j] = log_similarity(similarity, stats_cand[k][j], true)
                end

                # unnormalized Pr(obs i joins cluster k) (uses current cohesions and similarities)
                lw[k] = lcohes_cand[k] + sum(lsimilar_cand[k]) - lcohesions[k] - sum(lsimilarities[k])
            end

            # weight for new singleton cluster
            stats_newclust = [ Similarity_NiG_indep_stats([X[i,j]]) for j = 1:p ]
            lsimilar_newclust = [ log_similarity(similarity, stats_newclust[j], true) for j in 1:p ]
            lw[K_now + 1] = lcohes1 + sum(lsimilar_newclust)

            # sample membership for obs i
            lw = lw .- maximum(lw)
            C[i] = StatsBase.sample(StatsBase.Weights(exp.(lw)))

            # update K_now, S, lcohesions, stats, lsimilarities
            if C[i] > K_now
                K_now += 1
                push!(S, 1)
                push!(lcohesions, lcohes1)
                push!(stats, stats_newclust)
                push!(lsimilarities, lsimilar_newclust)
            else
                S[C[i]] += 1
                lcohesions[C[i]] = deepcopy(lcohes_cand[C[i]])
                stats[C[i]] = deepcopy(stats_cand[C[i]])
                lsimilarities[C[i]] = deepcopy(lsimilar_cand[C[i]])
            end

        end
    end

    return C, K_now, S, lcohesions, stats, lsimilarities
end

function sim_lik(C::Vector{Int}, X::Union{Matrix{T}, Matrix{Union{T, Missing}}} where T <: Real, 
    similarity::Similarity_NiG_indep, Xstats::Vector{Vector{Similarity_NiG_indep_stats}}, basemeasure::Baseline_measure)

    n, p = size(X)
    K = maximum(C)
    S = StatsBase.counts(C, K)

    β = Matrix{Float64}(undef, K, p)
    for k in 1:K
        ϕ = rand(Dirichlet(p, 1.0/p))
        τ = rand(Gamma(1.0, 2.0 * basemeasure.τ0))
        ψ = rand(Exponential(2.0), p)
        β[k,:] = randn(p) .* τ .* ϕ .* ψ
    end

    μ = randn(K) .* basemeasure.σ0 .+ basemeasure.μ0
    σ = rand(K) .* basemeasure.upper_σ

    Xfill = deepcopy(X)
    y = Vector{Float64}(undef, n)

    for k in 1:K

        C_indx_now = findall(C.==k)

        for j in 1:p

            fill_indx = findall(ismissing.(X[C_indx_now,j]))
            n_fill = length(fill_indx)
            
            if n_fill < S[k]
                xbar_now = Xstats[k][j].sumx / Xstats[k][j].n
                mean_now = xbar_now # could do something else
                sd_now = (Xstats[k][j].sumx2 - Xstats[k][j].n * xbar_now^2) / (Xstats[k][j].n - 1.0) # could do something else
            else
                mean_now = similarity.m0
                sd_now = similarity.b0 / (similarity.a0 + 1.0) / similarity.sc_div0
            end

            # center Xfill
            Xfill[C_indx_now,j] .-= mean_now

            if n_fill > 0
                Xfill[C_indx_now[fill_indx],j] = randn(n_fill) .* sd_now
            end

        end

        y[C_indx_now] = randn(S[k]) .* σ[k] + Xfill[C_indx_now,:] * β[k,:] .+ μ[k]

    end

    return y, μ, β, σ 
end


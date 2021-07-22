using Core: Matrix, Vector
# simulate_prior.jl

export sim_part;

n = 30
p = 5
nmis = 10
nobs = n*p - nmis
X = Matrix{Union{Missing, Float64}}(missing, n, p)
obs_indx_sim = sample(1:(n*p), nobs; replace=false)
X[obs_indx_sim] = randn(nobs)
size(X)

X

for i in (n-10+1):n
    X[i,:] += [1.0, 3.0, 0.0, 0.0, -5.0]
end

X

# X = [0.1 10.0; -0.1 11.0; -3.0 0.0; -2.2 0.1; -2.5 -0.2]

obs_indx = [ findall(.!ismissing.(X[i,:])) for i in 1:size(X,1) ]

X[1:5,:]
obs_indx[1:5]

X[(n-10+1):n,:]
obs_indx[(n-10+1):n]

[ mean(skipmissing(X[:,j])) for j in 1:size(X,2) ]

similarity = Similarity_NiG_indep(0.0, 0.1, 4.0, 4.0)
similarity = Similarity_NiG_indep(0.0, 0.1, 1.0, 1.0)
α = 1.0
logα = log(α)

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

testC, testK, testS, testlc, teststat, testsimilar = sim_partition_PPMx(logα, X, similarity)
testC

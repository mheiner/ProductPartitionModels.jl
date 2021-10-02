# general.jl

export ObsXIndx, State_PPMx, get_lcohlsim, init_PPMx, refresh!,
Prior_cohesion, Prior_similarity, Prior_baseline, Prior_baseline_NormDLUnif, Prior_PPMx, init_PPMx_prior,
Model_PPMx;

struct ObsXIndx
    n_obs::Int
    n_mis::Int
    indx_obs::Vector{Int}
    indx_mis::Vector{Int}
end

function ObsXIndx(x::Union{Vector{Union{T, Missing}}, Vector{T}, Vector{Missing}} where T <: Real)
    p = length(x)

    indx_mis = findall(ismissing.(x))
    n_mis = length(indx_mis)

    indx_obs = setdiff(1:p, indx_mis)
    n_obs = length(indx_obs)

    n_mis + n_obs == p || throw("ObsXIndx indexing failed.")

    return ObsXIndx(n_obs, n_mis, indx_obs, indx_mis)
end

mutable struct State_PPMx{T <: LikParams_PPMx, TT <: Baseline_measure, TTT <: Cohesion_PPM, TTTT <: Similarity_PPMx,
                          TR <: Real, T5 <: Similarity_PPMxStats}
    C::Vector{Int}
    lik_params::Vector{T}

    baseline::TT
    cohesion::TTT
    similarity::TTTT

    lcohesions::Vector{TR}
    Xstats::Vector{Vector{T5}}
    lsimilarities::Vector{Vector{TR}}

    llik::TR
    iter::Int
end

function get_lcohlsim(C::Vector{Int}, X::Union{Matrix{T}, Matrix{Union{T, Missing}}},
    cohesion::TT, similarity::TTT) where {T <: Real, TT <: Cohesion_PPM, TTT <: Similarity_PPMx}
    ## calculates log cohesions, Xstats, and log similarity scores if any C, cohesion, and similarity; see similarity.jl and cohesion.jl

    p = size(X, 2)
    K = maximum(C)
    S = StatsBase.counts(C, K)

    lcohesions = [ log_cohesion(Cohesion_CRP(cohesion.logα, S[k], true)) for k in 1:K ]
    Xstats = [ [ Similarity_stats(similarity, X[findall(C .== k), j]) for j in 1:p ] for k in 1:K ]
    lsimilarities = [ [ log_similarity(similarity, Xstats[k][j]) for j in 1:p ] for k in 1:K ]

    lcohesions, Xstats, lsimilarities
end

function init_PPMx(y::Vector{T}, X::Union{Matrix{T}, Matrix{Union{T, Missing}}},
    C_init::Union{Int, Vector{Int}}=0;
    similarity_type::Symbol=:NN;
    lik_rand::Bool=true) where T <: Real

    n, p = size(X)
    n == length(y) || throw("X, y dimension mismatch.")

    if C_init == 0
        C = collect(1:n)
    elseif C_init == 1
        C = ones(Int64, n)
    elseif typeof(C_init) <: Vector{Int}
        C = C_init
    end

    K = maximum(C)
    S = StatsBase.counts(C, K)

    baseline = Baseline_NormDLUnif(0.0, 5.0, 1.0, 10.0)

    if lik_rand
        lik_params = [ LikParams_PPMxReg(randn(), # mu
                                rand(), # sig
                                randn(p), # beta
                                Hypers_DirLap(rand(Dirichlet(p, 1.0)), rand(Exponential(0.5), p), rand(Exponential(0.5*baseline.tau0))) # beta hypers
                                )
                        for k in 1:K ]
    else
        lik_params = [ LikParams_PPMxReg(0.0, # mu
                                1.0, # sig
                                zeros(p), # beta
                                Hypers_DirLap(fill(1.0/p, p), fill(0.5, p), 0.5*baseline.tau0) # beta hypers
                                )
                        for k in 1:K ]
    end

    cohesion = Cohesion_CRP(1.0, 0)

    if similarity_type == :NN
        similarity = Similarity_NN(1.0, 0.0, 1.0)
    elseif similarity_type == :NiG_indep
        similarity = Similarity_NiG_indep(0.0, 0.1, 1.0, 1.0)
    end

    lcohesions, Xstats, lsimilarities = get_lcohlsim(C, X, cohesion, similarity)

    llik = 0.0
    iter = 0

    return State_PPMx(C, lik_params, baseline, cohesion, similarity, lcohesions, Xstats, lsimilarities, llik, iter)
end

function refresh!(state::State_PPMx, y::Vector{T}, X::Union{Matrix{T}, Matrix{Union{T, Missing}}}, obsXIndx::Vector{ObsXIndx}, refresh_llik::Bool=true) where T <: Real
    ## updates the log likelihood and calculated cohesions, Xstats, and similarity scores if any of C, baseline, lik_params, cohesion, or similarity change

    lcohes_upd, Xstats_upd, lsimil_upd = get_lcohlsim(state.C, X, state.cohesion, state.similarity)

    if refresh_llik
        llik = llik_all(y, X, state.C, obsXIndx, state.lik_params, Xstats_upd, state.similarity)
        state.llik = llik # depends on C, lik_params
    end

    state.lcohesions = lcohes_upd # depends on C, cohesion
    state.Xstats = Xstats_upd # depends on C
    state.lsimilarities = lsimil_upd # depends on C, similarity

    return nothing
end

abstract type Prior_cohesion end
abstract type Prior_similarity end
abstract type Prior_baseline end

mutable struct Prior_cohesion_CRP{TR <: Real} <: Prior_cohesion
    alpha_sh::TR
    alpha_sc::TR
end

mutable struct Prior_similarity_NiG_indep{TR <: Real} <: Prior_similarity
end

mutable struct Prior_baseline_NormDLUnif{TR <: Real} <: Prior_baseline
    mu0_mean::TR
    mu0_sd::TR
    sig0_upper::TR
end

mutable struct Prior_PPMx
    cohesion::Union{Nothing, Prior_cohesion}
    similarity::Union{Nothing, Prior_similarity}
    baseline::Union{Nothing, Prior_baseline}
end

function init_PPMx_prior()
    return Prior_PPMx(nothing, nothing, Prior_baseline_NormDLUnif(0.0, 100.0, 10.0))
end

## testing unexpected behavior
# abstract type Zoo end
# mutable struct Foo{T <: Real} <: Zoo
#     x::T
#     y::T
# end
# mutable struct Bar{T <: Real}
#     z::Foo{T}
# end
# Bar(Foo(0.0, 1.0))

mutable struct Model_PPMx{T <: Real}
    y::Vector{T}
    X::Union{Matrix{T}, Matrix{Union{T, Missing}}}

    obsXIndx::Vector{ObsXIndx}
    n::Int
    p::Int

    prior::Prior_PPMx
    state::State_PPMx
end

function Model_PPMx(y::Vector{T}, X::Union{Matrix{T}, Matrix{Union{T, Missing}}},
    C_init::Union{Int, Vector{Int}}=0;
    init_lik_rand::Bool=true) where T <: Real
    n, p = size(X)
    n == length(y) || throw("X, y dimension mismatch.")
    obsXIndx = [ ObsXIndx(X[i,:]) for i in 1:n ]

    prior = init_PPMx_prior()
    state = init_PPMx(y, X, deepcopy(C_init), lik_rand=init_lik_rand)

    return Model_PPMx(deepcopy(y), deepcopy(X), obsXIndx, n, p, prior, state)
end

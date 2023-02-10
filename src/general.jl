# general.jl

export ObsXIndx, State_PPMx, get_lcohlsim, init_PPMx, refresh!,
Prior_cohesion, Prior_similarity, 
Prior_baseline, Prior_baseline_NormDLUnif, Prior_baseline_NormUnif, 
Prior_PPMx, init_PPMx_prior,
Model_PPMx;

struct ObsXIndx
    n_obs::Int
    n_mis::Int
    indx_obs::Vector{Int}
    indx_mis::Vector{Int}
end

"""
    ObsXIndx(x)

Index missing and observed elements of vector `x` and return vector: (number of observed entries, number of missing entries, indexes of observed entries, indexes of missing entries).
"""
function ObsXIndx(x::Union{Vector{Union{T, Missing}}, Vector{T}, Vector{Missing}} where T <: Real)
    p = length(x)

    indx_mis = findall(ismissing.(x))
    n_mis = length(indx_mis)

    indx_obs = setdiff(1:p, indx_mis)
    n_obs = length(indx_obs)

    n_mis + n_obs == p || throw("ObsXIndx indexing failed.")

    return ObsXIndx(n_obs, n_mis, indx_obs, indx_mis)
end

"""
    State_PPMx

Store a complete state for a PPMx, including log-likelihood value and MCMC iteration.
"""
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

"""
    get_lcohlsim(C, X, cohesion, similarity)

Calculate log cohesions, Xstats, and log similarity scores with covariate matrix `X` under any allocation vector `C`, `cohesion`, and `similarity`.
"""
function get_lcohlsim(C::Vector{Int}, X::Union{Matrix{T}, Matrix{Union{T, Missing}}},
    cohesion::TT, similarity::TTT) where {T <: Real, TT <: Cohesion_PPM, TTT <: Similarity_PPMx}
    ## calculates log cohesions, Xstats, and log similarity scores for any C, cohesion, and similarity; see similarity.jl and cohesion.jl

    p = size(X, 2)
    K = maximum(C)
    S = StatsBase.counts(C, K)

    lcohesions = [ log_cohesion(Cohesion_CRP(cohesion.logα, S[k], true)) for k in 1:K ]
    Xstats = [ [ Similarity_stats(similarity, X[findall(C .== k), j]) for j in 1:p ] for k in 1:K ]
    lsimilarities = [ [ log_similarity(similarity, Xstats[k][j]) for j in 1:p ] for k in 1:K ]

    lcohesions, Xstats, lsimilarities
end

"""
    init_PPMx(y, X, C_init=0[, similarity_type=:NN, sampling_model=:Reg, lik_rand=true])

Initialize a complete PPMx state.

If `C_init` is 0, initialize every unit to a singleton cluster; if 1, all units share one cluster; 
if a vector of integers, it gives the allocation of units.

Similarity types include `:NN`, `:NNiG_indep`, and `:NNiChisq_indep`.

Sampling models include `:Reg` (regression in the sampling model), `:Mean` (no regression).

If `lik_rand` is true, generate cluster-specific parameters from the baseline.
"""
function init_PPMx(y::Vector{T}, X::Union{Matrix{T}, Matrix{Union{T, Missing}}},
    C_init::Union{Int, Vector{Int}}=0;
    similarity_type::Symbol=:NN,
    sampling_model::Symbol=:Reg,
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

    if sampling_model == :Reg
        baseline = Baseline_NormDLUnif(0.0, 5.0, 0.1, 10.0)

        if lik_rand
            lik_params = [ LikParams_PPMxReg(randn(), # mu
                                    rand(Uniform(baseline.sig_lower, baseline.sig_upper)), # sig
                                    randn(p), # beta
                                    Hypers_DirLap(rand(Dirichlet(p, 1.0)), rand(Exponential(0.5), p), rand(Exponential(0.5*baseline.tau0))) # beta hypers
                                    )
                            for k in 1:K ]
        else
            lik_params = [ LikParams_PPMxReg(0.0, # mu
                                    0.1 * baseline.sig_upper, # sig
                                    zeros(p), # beta
                                    Hypers_DirLap(fill(1.0/p, p), fill(0.5, p), 0.5*baseline.tau0) # beta hypers
                                    )
                            for k in 1:K ]
        end
    
    elseif sampling_model == :Mean
        baseline = Baseline_NormUnif(0.0, 5.0, 10.0)

        if lik_rand
            lik_params = [ LikParams_PPMxMean(randn(), # mu
                                    rand(Uniform(baseline.sig_lower, baseline.sig_upper)) # sig
                                    )
                            for k in 1:K ]
        else
            lik_params = [ LikParams_PPMxMean(0.0, # mu
                                    0.1 * baseline.sig_upper # sig
                                    )
                            for k in 1:K ]
        end

    end

    cohesion = Cohesion_CRP(1.0, 0)

    if similarity_type == :NN
        similarity = Similarity_NN(1.0, 0.0, 1.0)
    elseif similarity_type == :NNiG_indep
        similarity = Similarity_NNiG_indep(0.0, 0.1, 2.0, 2.0)
    elseif similarity_type == :NNiChisq_indep
        similarity = Similarity_NNiChisq_indep(0.0, 0.1, 4.0, 1.0)
    end

    lcohesions, Xstats, lsimilarities = get_lcohlsim(C, X, cohesion, similarity)

    llik = 0.0
    iter = 0

    return State_PPMx(C, lik_params, baseline, cohesion, similarity, lcohesions, Xstats, lsimilarities, llik, iter)
end

"""
    refresh!(state, y, X, obsXIndx, refresh_llik=true)

Update the log likelihood and calculated cohesions, Xstats, and similarity scores in the PPMx state if any of allocation, baseline, lik_params, cohesion, or similarity change.
"""
function refresh!(state::State_PPMx, y::Vector{T}, X::Union{Matrix{T}, Matrix{Union{T, Missing}}}, obsXIndx::Vector{ObsXIndx}, refresh_llik::Bool=true) where T <: Real
    ## updates the log likelihood and calculated cohesions, Xstats, and similarity scores if any of C, baseline, lik_params, cohesion, or similarity change

    lcohes_upd, Xstats_upd, lsimil_upd = get_lcohlsim(state.C, X, state.cohesion, state.similarity)

    if refresh_llik
        
        if typeof(state.lik_params[1]) <: LikParams_PPMxReg
            llik = llik_all(y, X, state.C, obsXIndx, state.lik_params, Xstats_upd, state.similarity)[:llik]
        elseif typeof(state.lik_params[1]) <: LikParams_PPMxMean
            llik = llik_all(y, state.C, state.lik_params)[:llik]
        end
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

mutable struct Prior_similarity_NNiG_indep{TR <: Real} <: Prior_similarity
end

"""
    Prior_baseline_NormDLUnif

Prior hyperparamters for the Baseline_NormDLUnif type.
"""
mutable struct Prior_baseline_NormDLUnif{TR <: Real} <: Prior_baseline
    mu0_mean::TR
    mu0_sd::TR
    sig0_upper::TR
end

"""
    Prior_baseline_NormDLUnif

Prior hyperparamters for the Baseline_NormUnif type.
"""
mutable struct Prior_baseline_NormUnif{TR <: Real} <: Prior_baseline
    mu0_mean::TR
    mu0_sd::TR
    sig0_upper::TR
end

"""
    Prior_PPMx

Collect prior information for cohesion, similarity, and baseline in a PPMx.
"""
mutable struct Prior_PPMx
    cohesion::Union{Nothing, Prior_cohesion}
    similarity::Union{Nothing, Prior_similarity}
    baseline::Union{Nothing, Prior_baseline}
end

"""
    Prior_PPMx

Collect prior information for cohesion, similarity, and baseline in a PPMx.
"""
function init_PPMx_prior(sampling_model::Symbol=:Reg)

    if sampling_model == :Reg
        bs = Prior_baseline_NormDLUnif(0.0, 100.0, 10.0)
    elseif sampling_model == :Mean
        bs = Prior_baseline_NormUnif(0.0, 100.0, 10.0)
    end

    return Prior_PPMx(nothing, nothing, bs)
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

"""
    Model_PPMx(y, X, C_init=0, similarity_type=:NN, sampling_model=:Reg, init_lik_rand=true)

Create and initialize a complete PPMx model object with data, prior, and current (parameter) state. 

If `C_init` is 0, initialize every unit to a singleton cluster; if 1, all units share one cluster; 
if a vector of integers, it gives the allocation of units.

Similarity types include `:NN`, `:NNiG_indep`, and `:NNiChisq_indep`.

Sampling models include `:Reg` (regression in the sampling model), `:Mean` (no regression).

If `lik_rand` is true, generate cluster-specific parameters from the baseline.
"""
function Model_PPMx(y::Vector{T}, X::Union{Matrix{T}, Matrix{Union{T, Missing}}},
    C_init::Union{Int, Vector{Int}}=0;
    similarity_type::Symbol=:NN,
    sampling_model::Symbol=:Reg, # one of :Reg or :Mean
    init_lik_rand::Bool=true) where T <: Real

    n, p = size(X)
    n == length(y) || error("PPMx model initialization: X, y dimension mismatch.")
    obsXIndx = [ ObsXIndx(X[i,:]) for i in 1:n ]

    prior = init_PPMx_prior(sampling_model)
    state = init_PPMx(y, X, deepcopy(C_init), 
        similarity_type=similarity_type, 
        sampling_model=sampling_model,
        lik_rand=init_lik_rand
        )

    state.baseline.sig0 < prior.baseline.sig0_upper || error("sig0 in the baseline must be initialized below its prior upper bound.")
    all([ state.lik_params[j].sig for j in 1:maximum(state.C) ] .<= state.baseline.sig_upper) || error("error sd (sig) must be at or below sig_upper for all clusters.")
    all([ state.lik_params[j].sig for j in 1:maximum(state.C) ] .>= state.baseline.sig_lower) || error("error sd (sig) must be at or above sig_lower for all clusters.")

    return Model_PPMx(deepcopy(y), deepcopy(X), obsXIndx, n, p, prior, state)
end

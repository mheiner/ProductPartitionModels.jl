# general.jl

export ObsXIndx, State_PPMx, Model_PPMx;

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
    lik_params::T

    baseline::TT
    cohesion::TTT
    similarity::TTTT

    lcohesions::Vector{TR}
    Xstats::Vector{Vector{T5}}
    lsimilarities::Vector{Vector{TR}}

    iter::Int
end

mutable struct Model_PPMx{T <: Real}
    y::Vector{T}
    X::Union{Matrix{T}, Matrix{Union{T, Missing}}}
    
    obsXIndx::Vector{ObsXIndx}
    p::Int    

    state::State_PPMx
end



# using Base: Real
# cohesion.jl

export log_coheseion_CRP;


abstract type Cohesion_PPM end

mutable struct Cohesion_CRP <: Cohesion_PPM

    α::Real
    size::Int

    logα::Real

    Cohesion_CRP(α, size) = new(α, size, log(α))
end

function log_cohesion_CRP(logα::Real, size::Int)
    return logα + SpecialFunctions.loggamma(size)
end




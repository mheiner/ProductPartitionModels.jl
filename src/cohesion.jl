# using Base: Real
# cohesion.jl

export Cohesion_PPM, Cohesion_CRP, log_coheseion;

mutable struct Cohesion_CRP <: Cohesion_PPM

    α::Real
    size::Int

    logα::Real

    Cohesion_CRP(α, size, logα) = (α <= 0.0 || size < 0) ? error("Invalid cohesion parameters.") : new(α, size, logα)
end

function Cohesion_CRP(α::Real, size::Int)
    (α > 0.0 && size >= 0) || throw("Invalid cohesion parameters.")
    Cohesion_CRP(α, size, log(α))
end
function Cohesion_CRP(α::Real, size::Int, logα::Bool)
    if logα
        out = Cohesion_CRP(NaN, size, α)
    else
        out = Cohesion_CRP(α, size)
    end
    return out
end

function log_cohesion(arg::Cohesion_CRP)
    return arg.logα + SpecialFunctions.loggamma(arg.size)
end

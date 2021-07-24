# likelihood.jl

export Baseline_measure, Baseline_NormDLUnif;

abstract type Baseline_measure end

mutable struct Baseline_NormDLUnif <: Baseline_measure
    μ0::Real 
    σ0::Real

    τ0::Real # global shrinkage 

    upper_σ::Real

    Baseline_NormDLUnif(μ0, σ0, τ0, upper_σ) = new(μ0, σ0, τ0, upper_σ)
end





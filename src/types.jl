# types.jl

export Cohesion_PPM, 
Similarity_PPMx, Similarity_NiG_indep, Similarity_PPMxStats, Similarity_NiG_indep_stats,
Baseline_measure, Baseline_NormDLUnif, 
Hypers_shrinkReg, Hypers_DirLap, 
LikParams_PPMx, LikParams_PPMxReg;

abstract type Cohesion_PPM end

abstract type Similarity_PPMx end

mutable struct Similarity_NiG_indep <: Similarity_PPMx
    m0::Real 
    sc_div0::Real
    a0::Real
    b0::Real

    lsc_div0::Real
    lga0::Real
    lb0::Real

    Similarity_NiG_indep(m0, sc_div0, a0, b0) = new(m0, sc_div0, a0, b0, 
        log(sc_div0), SpecialFunctions.loggamma(a0), log(b0))
end

abstract type Similarity_PPMxStats end

mutable struct Similarity_NiG_indep_stats <: Similarity_PPMxStats
    n::Int
    sumx::Real
    sumx2::Real
end


abstract type Baseline_measure end

mutable struct Baseline_NormDLUnif <: Baseline_measure
    mu0::Real 
    sig0::Real

    tau0::Real # global shrinkage scale

    upper_sig::Real
end

abstract type Hypers_shrinkReg end

mutable struct Hypers_DirLap{T <: Real} <: Hypers_shrinkReg
    phi::Vector{T}
    psi::Vector{T}
    tau::Real
end

abstract type LikParams_PPMx end

mutable struct LikParams_PPMxReg{T <: Real} <: LikParams_PPMx
    mu::Real
    sig::Real

    beta::Vector{T}
    beta_hypers::Hypers_shrinkReg

    # LikParams_PPMxReg(mu, sig, beta, beta_hypers) = sig <= 0.0 ? error("St. deviation must be positive.") : new(mu, sig, beta, beta_hypers)
end

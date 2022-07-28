# types.jl

export Cohesion_PPM,
Similarity_PPMx, Similarity_NNiG_indep, Similarity_NNiChisq_indep, Similarity_NN,
Similarity_PPMxStats, Similarity_NNiG_indep_stats, Similarity_NN_stats,
Baseline_measure, Baseline_NormDLUnif, Baseline_NormUnif,
Hypers_shrinkReg, Hypers_DirLap,
LikParams_PPMx, LikParams_PPMxReg, LikParams_PPMxMean;

abstract type Cohesion_PPM end

abstract type Similarity_PPMx end

mutable struct Similarity_NNiG_indep <: Similarity_PPMx

    # IG(sig2; shape=a, scale=b) N(mu; mean=mu0, variance=sig2/sc_prec0)

    m0::Real
    sc_prec0::Real
    a0::Real
    b0::Real

    lsc_prec0::Real
    lga0::Real
    lb0::Real

    Similarity_NNiG_indep(m0, sc_prec0, a0, b0) = new(m0, sc_prec0, a0, b0,
        log(sc_prec0), SpecialFunctions.loggamma(a0), log(b0))
end

mutable struct Similarity_NNiChisq_indep <: Similarity_PPMx

    # similarity of x vector is marginal density of x when
    # x_i | mu, sig2 ~iid N(mean = mu, variance = sig2)
    # p(mu, sig2) = N(mu; mean = mu0, variance = sig2/sc_prec0) IG(sig2; shape = a, scale = b)

    m0::Real
    sc_prec0::Real

    nu0::Real
    s20::Real

    a0::Real # inverse-gamma shape parameter
    b0::Real # inverse-gamma scale parameter

    lsc_prec0::Real
    lga0::Real
    lb0::Real

    Similarity_NNiChisq_indep(m0, sc_prec0, nu0, s20) = new(m0, sc_prec0, nu0, s20,
        0.5*nu0, 0.5*nu0*s20,
        log(sc_prec0), SpecialFunctions.loggamma(0.5*nu0), log(0.5*nu0*s20))
end

mutable struct Similarity_NN <: Similarity_PPMx

    # N(zeta, sd=sd)
    # zeta ~ N(m0, sd=sd0)

    sd::Real
    m0::Real
    sd0::Real
end


abstract type Similarity_PPMxStats end

mutable struct Similarity_NNiG_indep_stats <: Similarity_PPMxStats
    n::Int
    sumx::Real
    sumx2::Real
end

mutable struct Similarity_NN_stats <: Similarity_PPMxStats
    n::Int
    sumx::Real
    sumx2::Real
end


abstract type Baseline_measure end

mutable struct Baseline_NormDLUnif <: Baseline_measure
    mu0::Real
    sig0::Real

    tau0::Real # global shrinkage scale

    sig_upper::Real
    sig_lower::Real

    Baseline_NormDLUnif(mu0, sig0, tau0, sig_upper) = new(mu0, sig0, tau0, sig_upper, 1.0e-6) # lower bound on error sd
end

mutable struct Baseline_NormUnif <: Baseline_measure
    mu0::Real
    sig0::Real

    sig_upper::Real
    sig_lower::Real

    Baseline_NormUnif(mu0, sig0, sig_upper) = new(mu0, sig0, sig_upper, 1.0e-6) # lower bound on error sd
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

mutable struct LikParams_PPMxMean <: LikParams_PPMx
    mu::Real
    sig::Real
    # LikParams_PPMxMean(mu, sig) = sig <= 0.0 ? error("St. deviation must be positive.") : new(mu, sig)
end

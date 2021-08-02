# similarity.jl

export Similarity_PPMx, Similarity_NiG_indep, Similarity_NiG_indep_stats,
log_similarity_nig;


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

mutable struct Similarity_NiG_indep_stats <: Similarity_PPMx
    n::Int
    sumx::Real
    sumx2::Real
end

function Similarity_NiG_indep_stats(x::Vector{T} where T <: Real)
    n = length(x)
    sumx = sum(x)
    sumx2 = sum( x.^2 )

    return Similarity_NiG_indep_stats(n, sumx, sumx2)
end
function Similarity_NiG_indep_stats(x::Vector{Union{Missing, T}} where T <: Real)

    obs_indx = findall(.!ismissing.(x))
    n = length(obs_indx)

    if n > 0
        sumx = sum(x)
        sumx2 = sum( x.^2 )
    else
        sumx = 0
        sumx2 = 0
    end

    return Similarity_NiG_indep_stats(n, sumx, sumx2)
end
function Similarity_NiG_indep_stats(existing::Similarity_NiG_indep_stats, x::Real, action::Symbol=:add)

    if action == :add

        n = existing.n + 1
        sumx = existing.sumx + x
        sumx2 = existing.sumx2 + x^2

    elseif action == :subtract

        n = existing.n - 1
        sumx = existing.sumx - x
        sumx2 = existing.sumx2 - x^2
    
    end

    return Similarity_NiG_indep_stats(n, sumx, sumx2)
end
function Similarity_NiG_indep_stats(existing::Similarity_NiG_indep_stats, x::Missing, action::Symbol=:add)
    return Similarity_NiG_indep_stats(existing.n, existing.sumx, existing.sumx2)
end

# test = Similarity_NiG_indep_stats(randn(100))
# test.sumx2

function log_similarity(similar::Similarity_NiG_indep, stats::Similarity_NiG_indep_stats, fulldensity::Bool=true)

    if stats.n > 0

        xbar = stats.sumx / stats.n
        ss = stats.sumx2 - stats.n * xbar^2
    
        sc_div1 = similar.sc_div0 + stats.n
        # m1 = (similar.sc_div0*similar.m0 + stats.n*xbar) / sc_div1  # never used
        a1 = similar.a0 + 0.5*stats.n
        b1 = similar.b0 + 0.5*ss + 0.5*(stats.n * similar.sc_div0)*(xbar - similar.m0)^2 / sc_div1

        out = SpecialFunctions.loggamma(a1) - 0.5*log(sc_div1) - 0.5*stats.n*log(2π) - a1*log(b1)

        if fulldensity
           out += 0.5*similar.lsc_div0 + similar.a0*similar.lb0 - similar.lga0 
        end

    else 
        out = 0.0
    end

    return out
end




# similarity.jl

export Similarity_stats, log_similarity;


function Similarity_stats(similar::Similarity_NiG_indep,
    x::Vector{T} where T <: Real)

    n = length(x)
    sumx = sum(x)
    sumx2 = sum( x.^2 )

    return Similarity_NiG_indep_stats(n, sumx, sumx2)
end
function Similarity_stats(similar::Similarity_NiG_indep,
    x::Vector{Union{Missing, T}} where T <: Real)

    obs_indx = findall(.!ismissing.(x))
    n = length(obs_indx)

    if n > 0
        sumx = sum(x[obs_indx])
        sumx2 = sum( x[obs_indx].^2 )
    else
        sumx = 0
        sumx2 = 0
    end

    return Similarity_NiG_indep_stats(n, sumx, sumx2)
end
function Similarity_stats(existing::Similarity_NiG_indep_stats, x::Real, action::Symbol=:add)

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
function Similarity_stats(existing::Similarity_NiG_indep_stats, x::Missing, action::Symbol=:add)
    return Similarity_NiG_indep_stats(existing.n, existing.sumx, existing.sumx2)
end



function Similarity_stats(similar::Similarity_NN,
    x::Vector{T} where T <: Real)

    n = length(x)
    sumx = sum(x)
    sumx2 = sum( x.^2 )

    return Similarity_NN_stats(n, sumx, sumx2)
end
function Similarity_stats(similar::Similarity_NN,
    x::Vector{Union{Missing, T}} where T <: Real)

    obs_indx = findall(.!ismissing.(x))
    n = length(obs_indx)

    if n > 0
        sumx = sum(x[obs_indx])
        sumx2 = sum( x[obs_indx].^2 )
    else
        sumx = 0
        sumx2 = 0
    end

    return Similarity_NN_stats(n, sumx, sumx2)
end
function Similarity_stats(existing::Similarity_NN_stats, x::Real, action::Symbol=:add)

    if action == :add

        n = existing.n + 1
        sumx = existing.sumx + x
        sumx2 = existing.sumx2 + x^2

    elseif action == :subtract

        n = existing.n - 1
        sumx = existing.sumx - x
        sumx2 = existing.sumx2 - x^2

    end

    return Similarity_NN_stats(n, sumx, sumx2)
end
function Similarity_stats(existing::Similarity_NN_stats, x::Missing, action::Symbol=:add)
    return Similarity_NN_stats(existing.n, existing.sumx, existing.sumx2)
end



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
function log_similarity(similar::Similarity_NN, stats::Similarity_NN_stats, fulldensity::Bool=true)

    if stats.n > 0

        sig2 = similar.sd^2
        v0 = similar.sd0^2

        m1 = stats.sumx2 - 2.0*similar.m0*stats.sumx + stats.n*similar.m0^2
        m2 = v0 * (stats.sumx - stats.n*similar.m0)^2 / (stats.n*v0 + sig2)
        m = (m1 - m2) / sig2

        logdet = stats.n*log(sig2) + log(1.0 + stats.n*v0/sig2)

        nl2p = stats.n*log(2π)

        out = -0.5 * (nl2p + logdet + m)

    else
        out = 0.0
    end

    return out
end

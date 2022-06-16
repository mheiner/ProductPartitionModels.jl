# similarity.jl

export Similarity_stats, log_similarity;


function Similarity_stats(similar::Union{Similarity_NNiG_indep, Similarity_NNiChisq_indep},
    x::Vector{T} where T <: Real)

    n = length(x)
    sumx = sum(x)
    sumx2 = sum( x.^2 )

    return Similarity_NNiG_indep_stats(n, sumx, sumx2) # stats same between Similarity_NNiG_indep, Similarity_NNiChisq_indep
end
function Similarity_stats(similar::Union{Similarity_NNiG_indep, Similarity_NNiChisq_indep},
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

    return Similarity_NNiG_indep_stats(n, sumx, sumx2) # stats same between Similarity_NNiG_indep, Similarity_NNiChisq_indep
end
function Similarity_stats(existing::Similarity_NNiG_indep_stats, x::Real, action::Symbol=:add)

    if action == :add

        n = existing.n + 1
        sumx = existing.sumx + x
        sumx2 = existing.sumx2 + x^2

    elseif action == :subtract

        n = existing.n - 1
        sumx = existing.sumx - x
        sumx2 = existing.sumx2 - x^2

    end

    return Similarity_NNiG_indep_stats(n, sumx, sumx2)
end
function Similarity_stats(existing::Similarity_NNiG_indep_stats, x::Missing, action::Symbol=:add)
    return Similarity_NNiG_indep_stats(existing.n, existing.sumx, existing.sumx2)
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

function logdens_nn_marg(sig2::T, m0::T, v0::T, sumx::T, sumx2::T, n::Int) where T <: Real

    ## Calculates the log of the marginal density of x (vector of length n) under model
    ## x | mu ~iid normal(mu, sig2), mu ~ normal(m0, v0)

    sig2 > 0.0 || DomainError(sig2, "Variance must be positive.")
    v0 > 0.0 || DomainError(v0, "Variance must be positive.")
    sumx2 > 0.0 || DomainError(sumx2, "Sum of squares must be positive.")
    n > 0 || DomainError(n, "Sample size must be positive.")

    n = float(n)

    m1 = sumx2 - 2.0*m0*sumx + n*m0^2
    m2 = v0 * (sumx - n*m0)^2 / (n*v0 + sig2)
    m = (m1 - m2) / sig2

    logdet = n*log(sig2) + log(1.0 + n*v0/sig2)

    nl2p = n*log(2π)

    return -0.5 * (nl2p + logdet + m)
end

function log_similarity(similar::Union{Similarity_NNiG_indep, Similarity_NNiChisq_indep},
    stats::Similarity_NNiG_indep_stats, fulldensity::Bool=true)

    if stats.n > 0

        xbar = stats.sumx / stats.n
        ss = stats.sumx2 - stats.n * xbar^2

        sc_prec1 = similar.sc_prec0 + stats.n
        # m1 = (similar.sc_prec0*similar.m0 + stats.n*xbar) / sc_prec1  # never used
        a1 = similar.a0 + 0.5*stats.n
        b1 = similar.b0 + 0.5*ss + 0.5*(stats.n * similar.sc_prec0)*(xbar - similar.m0)^2 / sc_prec1

        out = SpecialFunctions.loggamma(a1) - 0.5*log(sc_prec1) - 0.5*stats.n*log(2π) - a1*log(b1)

        if fulldensity
           out += 0.5*similar.lsc_prec0 + similar.a0*similar.lb0 - similar.lga0
        end

    else
        out = 0.0
    end

    return out
end
function log_similarity(similar::Similarity_NN, stats::Similarity_NN_stats, fulldensity::Bool=true)

    if stats.n > 0

        out = logdens_nn_marg(similar.sd^2, similar.m0, similar.sd0^2, stats.sumx, stats.sumx2, stats.n)

    else
        out = 0.0
    end

    return out
end

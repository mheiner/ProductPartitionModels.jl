# slice_sample.jl

export TargetArgs,
TargetArgs_NormRegBeta, TargetArgs_NormRegSig,
logtarget,
ellipSlice, shrinkSlice;

abstract type TargetArgs end

mutable struct TargetArgs_NormRegBeta{T <: Real} <: TargetArgs
    y::Vector{T}
    X::Matrix{T}
    errorsd::T
end

function logtarget(beta::Vector{T} where T <: Real, args::TargetArgs_NormRegBeta)
    Xbeta = args.X * beta
    ee = args.y - Xbeta
    ss = sum(ee.^2)
    out = -0.5 * ss / args.errorsd^2
    return out, ss
end

mutable struct TargetArgs_NormRegSig{T <: Real} <: TargetArgs
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
end

function logtarget(errorsd::Real, args::TargetArgs_NormRegSig)
    Xbeta = args.X * args.beta
    ee = args.y - Xbeta
    ss = sum(ee.^2)
    out = -0.5 * ss / errorsd^2 - length(args.y)*log(errorsd)
    return out, ss
end



# Algorithm 1 from Nishihara, Murray, Adams (2014)
# logtarget is assumed to return a tuple with the target value as first output
function ellipSlice(x::Vector{T}, mu::Vector{T}, sig2::Vector{T},
    logtarget::Function, logtarget_args::TargetArgs,
    maxiter::Int=5000) where T <: Real

    L = length(x)
    length(mu) == length(sig2) == L || throw("All arguments must have same length.")
    all(sig2 .> 0.0) || throw("Prior variances must be positive.")

    nu = randn(L) .* sqrt.(sig2) + mu
    u = rand()

    lt = logtarget(x, logtarget_args)
    lz = lt[:llik] + log(u)
    theta = 2π * rand()

    theta_min = theta - 2π
    theta_max = theta

    x_cand = (x - mu)*cos(theta) + (nu - mu)*sin(theta) + mu
    lt_cand = logtarget(x_cand, logtarget_args)
    iter = 1

    while lt_cand[:llik] <= lz && iter <= maxiter
        if theta < 0
            theta_min = theta
        else
            theta_max = theta
        end
        theta = rand()*(theta_max - theta_min) + theta_min
        x_cand = (x - mu)*cos(theta) + (nu - mu)*sin(theta) + mu
        lt_cand = logtarget(x_cand, logtarget_args)
        iter += 1
    end

    iter <= maxiter || throw("Elliptical slice sampler exceeded max iterations.")

    return x_cand, lt_cand, iter
end

# Figure 5 from Neal (2003)
# logtarget is assumed to return a tuple with the target value as first output
function shrinkSlice(x::Real, L::Real, R::Real,
    logtarget::Function, logtarget_args::TargetArgs, maxiter::Int=5000)

    ((x > L) && (x < R)) || throw("Slice sampler input value outside of support.")

    lt = logtarget(x, logtarget_args)
    lz = lt[:llik] + log(rand())

    Lnew = L
    Rnew = R

    x_cand = rand()*(Rnew - Lnew) + Lnew
    lt_cand = logtarget(x_cand, logtarget_args)
    iter = 1

    while lt_cand[:llik] <= lz && iter <= maxiter
        if x_cand < x
            Lnew = x_cand
        else
            Rnew = x_cand
        end
        x_cand = rand()*(Rnew - Lnew) + Lnew
        lt_cand = logtarget(x_cand, logtarget_args)
        iter += 1
    end

    iter <= maxiter || throw("Slice sampler exceeded max iterations.")

    return x_cand, lt_cand, iter
end

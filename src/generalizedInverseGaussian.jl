# started by  Patrik Waldmann
# sampling implemented by Neal Grantham https://github.com/JuliaStats/Distributions.jl/blob/c0bea481d4345bb426d9095d0a989be5aa9c7b81/src/univariate/continuous/generalizedinversegaussian.jl
# updated by Dylan Festa https://github.com/dylanfesta/Distributions.jl/blob/master/src/univariate/continuous/generalizedinversegaussian.jl
# thread at https://github.com/JuliaStats/Distributions.jl/pull/587

import Base.rand, Base.convert, Distributions.params, Distributions.partype,
        Distributions.mode, Distributions.mean, Distributions.var,
        Distributions.pdf, Distributions.logpdf;

"""
    GeneralizedInverseGaussian(a,b,p)
The *generalized inverse Gaussian distribution* with parameters `a` and  `b` and `p` has probability density function
```math
f(x; a, b, p) = \\frac{(a/b)^{p/2}}{2K_p(\\sqrt{ab})}x^{(p-1)}
e^{-(ax+b/x)/2}, \\quad x > 0
```
where ``K_p`` is a modified Bessel function of the second kind ; `a`> 0, `b`>0, and `p` real number.
```julia
GeneralizedInverseGaussian(a, b, p)    # Generalized Inverse Gaussian distribution with parameters parameters a > 0, b > 0 and p real
params(d)           # Get the parameters, i.e. (a, b, p)
```
External links
* [Generalized Inverse Gaussian distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution)
"""
struct GeneralizedInverseGaussian{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    p::T

    function GeneralizedInverseGaussian{T}(a::T, b::T, p::T) where T
        # @check_args(GeneralizedInverseGaussian, a > zero(a) && b > zero(b))
        new{T}(a, b, p)
    end
end

GeneralizedInverseGaussian(a::T, b::T, p::T) where {T<: Real} = GeneralizedInverseGaussian{T}(a, b, p)
GeneralizedInverseGaussian(a::Real, b::Real, p::Real) = GeneralizedInverseGaussian(promote(a, b, p)...)
GeneralizedInverseGaussian(a::Integer, b::Integer, p::Integer) = GeneralizedInverseGaussian(Float64(a), Float64(b), Float64(p))

# @distr_support GeneralizedInverseGaussian 0.0 Inf


#### Conversions

function convert(::Type{GeneralizedInverseGaussian{T}}, a::S, b::S, p::S) where {T <: Real, S <: Real}
    GeneralizedInverseGaussian(T(a), T(b), T(p))
end
function convert(::Type{GeneralizedInverseGaussian{T}}, d::GeneralizedInverseGaussian{S}) where {T <: Real, S <: Real}
    GeneralizedInverseGaussian(T(d.a), T(d.b), T(d.p))
end

#### Parameters

params(d::GeneralizedInverseGaussian) = (d.a, d.b, d.p)
@inline partype(d::GeneralizedInverseGaussian{T}) where T <: Real = T


#### Statistics

function mean(d::GeneralizedInverseGaussian)
    a, b, p = params(d)
    q = sqrt(a * b)
    (sqrt(b) * besselk(p + 1, q)) / (sqrt(a) * besselk(p, q))
end

function var(d::GeneralizedInverseGaussian)
    a, b, p = params(d)
    q = sqrt(a * b)
    r = besselk(p, q)
    (b / a) * ((besselk(p + 2, q) / r) - (besselk(p + 1, q) / r)^2)
end

mode(d::GeneralizedInverseGaussian) = ((d.p - 1) + sqrt((d.p - 1)^2 + d.a * d.b)) / d.a


#### Evaluation

function pdf(d::GeneralizedInverseGaussian{T}, x::Real) where T <: Real
    if x > 0
        a, b, p = params(d)
        (((a / b)^(p / 2)) / (2 * besselk(p, sqrt(a * b)))) * (x^(p - 1)) * exp(- (a * x + b / x) / 2)
    else
        zero(T)
    end
end

function logpdf(d::GeneralizedInverseGaussian{T}, x::Real) where T <: Real
    if x > 0
        a, b, p = params(d)
        (p / 2) * (log(a) - log(b)) - log(2 * besselk(p, sqrt(a * b))) + (p - 1) * log(x) - (a * x + b / x) / 2
    else
        -T(Inf)
    end
end


# See eq. (5) in Lemonte & Cordeiro (2011)
# Statistics & Probability Letters 81:506–517
# F(x) = 1 - (ρ + σ), where ρ and σ are infinite sums
# calculated up to truncation below
# function cdf(d::GeneralizedInverseGaussian{T}, x::Real) where T<:Real
#     if x > 0
#         a, b, p = params(d)
#         c = (((a / b)^(p / 2)) / (2 * besselk(p, sqrt(a * b))))
#         η = a / 2
#         ω = b / 2
#         λ = p
#         lη = log(η)
#         lω = log(ω)
#         lx = log(x)
#         gammasign(z) = iseven(floor(Int,min(0.0,z))) ? +1.0 : -1.0
#         # calculate first term ρ
#         ρ = 0.0
#         converged = false
#         j = 0
#         while !converged && j < 100
#             ρ_old = ρ
#             # ρ += c * (-1)^j * gamma(λ - j) * exp( (-p + j) * lη + j * lω - lfactorial(j))
#             ρ += gammasign(λ-j)*exp(lgamma(λ - j)-j*(lη + lω)-lfactorial(j))
#             converged = abs(ρ - ρ_old) < eps()
#             j += 1
#         end
#         ρ *= c*η^(-λ)
#         # calculate second term σ
#         σ = 0.0
#         converged = false
#         i = 0
#         while !converged && i < 100
#             σ_old = σ
#             k = i
#             for j in (0:k)
#                 l = k * lη + j * lω + (k - j + λ) * lx - lfactorial(k) - lfactorial(j)
#                 σ += (-1)^(k + j + 1)* c * exp(l)/(k - j + λ)
#             end
#             converged = abs(σ - σ_old) < eps()
#             i += 1
#         end
#         1 - (ρ + σ)
#     else
#         zero(T)
#     end
# end
#
# function mgf(d::GeneralizedInverseGaussian{T}, t::Real) where T <: Real
#     if t == zero(t)
#         one(T)
#     else
#         a, b, p = params(d)
#         (a / (a - 2t))^(p / 2) * besselk(p, sqrt(b * (a - 2t))) / besselk(p, sqrt(a * b))
#     end
# end
#
# function cf(d::GeneralizedInverseGaussian{T}, t::Real) where T <: Real
#     if t == zero(t)
#         one(T) + zero(T) * im
#     else
#         a, b, p = params(d)
#         (a / (a - 2t * im))^(p / 2) * besselk(p, sqrt(b * (a - 2t * im))) / besselk(p, sqrt(a * b))
#     end
# end



#### Sampling

# rand method from:
# Hörmann, W. & J. Leydold. (2014). Generating generalized inverse Gaussian random variates.
# J. Stat. Comput. 24: 547–557. doi:10.1007/s11222-013-9387-3

function rand(d::GeneralizedInverseGaussian)
    a, b, p = params(d)
    α = sqrt(a / b)
    β = sqrt(a * b)
    λ = abs(p)
    if (λ > 1) || (β > 1)
        x = _rou_shift(λ, β)
    elseif β >= min(0.5, (2 / 3) * sqrt(1 - λ))
        x = _rou(λ, β)
    else
        x = _hormann(λ, β)
    end
    p >= 0 ? x / α : 1 / (α * x)
end

function _gigqdf(x::Real, λ::Real, β::Real)
    (x^(λ - 1)) * exp(-β * (x + 1 / x) / 2)
end

function _lgigqdf(x::Real, λ::Real, β::Real)
    (λ - 1.0) * log(x) - β * (x + 1.0 / x) / 2.0
end

function _hormann(λ::Real, β::Real, maxiter=2000)
    # compute bounding rectangles
    m = β / (1 - λ + sqrt((1 - λ)^2 + β^2))  # mode
    x0 = β / (1 - λ)
    xstar = max(x0, 2 / β)
    # in subdomain (0, x0)
    k1 = _gigqdf(m, λ, β)
    a1 = k1 * x0
    # in subdomain (x0, 2 / β), may be empty
    if x0 < 2 / β
        k2 = exp(-β)
        a2 = λ == 0 ? k2 * log(2 / (β^2)) : k2 * ((2 / β)^λ - x0^λ) / λ
    else
        k2 = 0
        a2 = 0
    end
    # in subdomain (xstar, Inf)
    k3 = xstar^(λ - 1)
    a3 = 2k3 * exp(-xstar * β / 2) / β
    a = a1 + a2 + a3

    # perform rejection sampling
    iter = 1
    # while true
    while iter <= maxiter
        u = rand()
        v = a * rand()
        if v <= a1
            # in subdomain (0, x0)
            x = x0 * v / a1
            h = k1
        elseif v <= a1 + a2
            # in subdomain (x0, 2 / β)
            v -= a1
            x = λ == 0 ? β * exp(v * exp(β)) : (x0^λ + v * λ / k2)^(1 / λ)
            h = k2 * x^(λ - 1)
        else
            # in subdomain (xstar, Inf)
            v -= a1 + a2
            x = -2log(exp(-xstar * β / 2) - v * β / (2k3)) / β
            h = k3 * exp(-x * β / 2)
        end
        if u * h <= _gigqdf(x, λ, β)
            return x
        end
        iter += 1
    end
    error("Reached maxiter of Hormann method with λ=$(λ) and β=$(β)")
end

function _rou(λ::Real, β::Real, maxiter=2000)
    # compute bounding rectangle
    m = β / (1 - λ + sqrt((1 - λ)^2 + β^2))  # mode
    xpos = (1 + λ + sqrt((1 + λ)^2 + β^2)) / β
    vpos = sqrt(_gigqdf(m, λ, β))
    upos = xpos * sqrt(_gigqdf(xpos, λ, β))

    # perform rejection sampling
    iter = 1
    # while true
    while iter < maxiter
        u = upos * rand()
        v = vpos * rand()
        x = u / v
        if v^2 <= _gigqdf(x, λ, β)
            return x
        end
        iter += 1
    end
    error("Reached maxiter of ratio of uniforms method with λ=$(λ) and β=$(β)")
end

# function _rou_shift(λ::Real, β::Real, maxiter=5000)
#     # compute bounding rectangle
#     m = (λ - 1.0 + sqrt((λ - 1.0)^2 + β^2)) / β  # mode
#     a = -2.0(λ + 1.0) / β - m
#     b = 2.0*(λ - 1.0) * m / β - 1.0
#     p = b - (a^2) / 3.0
#     q = 2.0*(a^3) / 27.0 - (a * b) / 3.0 + m
#     ϕ = acos(-(q / 2.0) * sqrt(-27.0 / (p^3)))  # Cardano's formula
#     r = sqrt(-4.0*p / 3.0)
#     xneg = r * cos(ϕ / 3.0 + 4π / 3.0) - a / 3.0
#     xpos = r * cos(ϕ / 3.0) - a / 3.0
#
#     vpos = sqrt(_gigqdf(m, λ, β))
#     lvpos = 0.5 * _lgigqdf(m, λ, β)
#
#     uneg = (xneg - m) * sqrt(_gigqdf(xneg, λ, β))
#
#     upos = (xpos - m) * sqrt(_gigqdf(xpos, λ, β))
#     lupos = log(xpos - m) + 0.5 * _lgigqdf(xpos, λ, β)
#
#     # perform rejection sampling
#     iter = 1
#     # while true
#     while iter < maxiter
#         u = (upos - uneg) * rand() + uneg
#         v = vpos * rand()
#         x = max(u / v + m, 0)
#         if v^2 <= _gigqdf(x, λ, β)
#             return x
#         end
#         iter += 1
#     end
#     error("Reached maxiter of shifted ratio of uniforms method with λ=$(λ) and β=$(β)")
# end

function _rou_shift(λ::Real, β::Real, maxiter=5000)
    # compute bounding rectangle
    m = (λ - 1.0 + sqrt((λ - 1.0)^2 + β^2)) / β  # mode
    a = -2.0(λ + 1.0) / β - m
    b = 2.0*(λ - 1.0) * m / β - 1.0
    p = b - (a^2) / 3.0
    q = 2.0*(a^3) / 27.0 - (a * b) / 3.0 + m
    ϕ = acos(-(q / 2.0) * sqrt(-27.0 / (p^3)))  # Cardano's formula
    r = sqrt(-4.0*p / 3.0)
    xneg = r * cos(ϕ / 3.0 + 4π / 3.0) - a / 3.0
    xpos = r * cos(ϕ / 3.0) - a / 3.0

    lg_at_mode = _lgigqdf(m, λ, β)
    lg_at_xneg = _lgigqdf(xneg, λ, β)
    lg_at_xpos = _lgigqdf(xneg, λ, β)

    lg_ref = max(lg_at_mode, lg_at_xneg, lg_at_xpos) # to prevent underflow

    vpos = exp( 0.5 * (lg_at_mode - lg_ref) )
    uneg = (xneg - m) * exp( 0.5 * (lg_at_xneg - lg_ref) )
    upos = (xpos - m) * exp( 0.5 * (lg_at_xpos - lg_ref) )

    # perform rejection sampling
    iter = 1
    # while true
    while iter < maxiter
        u = (upos - uneg) * rand() + uneg
        v = vpos * rand()
        x = max(u / v + m, 0.0)
        if 2.0*log(v) <= ( _lgigqdf(x, λ, β) - lg_ref )
            return x
        end
        iter += 1
    end
    error("Reached maxiter of shifted ratio of uniforms method with λ=$(λ) and β=$(β)")
end

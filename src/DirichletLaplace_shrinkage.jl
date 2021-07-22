# DirichletLaplace_shrinkage.jl

# export ;

# using Distributions
# include("generalizedinversegaussian.jl")

mutable struct DirLap{T <: Real}
    K::Int # in the paper this is n
    α::T # in the paper this is a
    β::Vector{T} # in the paper this is theta
    ψ::Vector{T}
    ϕ::Vector{T}
    τ::T
end

function update_ψ(ϕ::Vector{T}, β::Vector{T}, τ::T) where T <: Real

    μ = τ .* ϕ ./ abs.(β)
    ψinv = [ rand( InverseGaussian(μ[k]) ) for k in 1:length(μ) ]

    ψ = 1.0 ./ ψinv

    inftest = ψ .== Inf
    if any(inftest)
        indx = findall(inftest)
        ψ[indx] .= exp(500.0) # cheater's method
    end

    return ψ
end

function update_τ(ϕ::Vector{T}, β::Vector{T}, α::T) where T <: Real

    K = length(β)
    p = float(K)*(α - 1.0)
    b = 2.0 * sum( abs.(β) ./ ϕ )

    return rand( GeneralizedInverseGaussian(1.0, b, p) )
end

function update_ϕ(β::Vector{T}, α::T) where T<: Real

    p = α - 1.0
    b = 2.0 .* abs.(β)
    Tvec = [ rand( GeneralizedInverseGaussian(1.0, b[k], p) ) for k in 1:length(β) ]

    return Tvec ./ sum(Tvec)
end

function rpost_normlmDiagBeta_beta1(y::Vector{T}, X::Matrix{T},
  σ2::T, Vdiag::Vector{T}, β0::T=0.0) where T <: Real

  σ2 > 0.0 || throw(ArgumentError("σ2 must be positive."))

  n,p = size(X) # assumes X was a matrix
  length(y) == n || throw(ArgumentError("y and X dimension mismatch"))
  ystar = y .- β0

  A = Diagonal(1.0 ./ Vdiag) + X'X ./ σ2
  U = (cholesky(A)).U
  Ut = transpose(U)

  # μ_a = At_ldiv_B(U, (X'ystar/σ2))
  μ_a = Ut \ (X'ystar ./ σ2)
  μ = U \ μ_a

  z = randn(p)
  β = U \ z + μ

  zerotest = β .== 0.0
  if any(zerotest)
      indx = findall(zerotest)
      β[indx] .= exp(-500.0) # cheater's method
  end

  return β
end

# function rpost_normlmDiagBeta_beta1b(y::Vector{T}, X::Matrix{T},
#   σ2::T, Vdiag::Vector{T}, β0::T=0.0) where T <: Real
#
#   σ2 > 0.0 || throw(ArgumentError("σ2 must be positive."))
#
#   n,p = size(X) # assumes X was a matrix
#   length(y) == n || throw(ArgumentError("y and X dimension mismatch"))
#   ystar = y .- β0
#
#   Prec = Diagonal(1.0 ./ Vdiag) + X'X ./ σ2
#   pot = X'ystar ./ σ2
#
#   β = rand( MvNormalCanon(pot, Prec) )
#
#   zerotest = β .== 0.0
#   if any(zerotest)
#       indx = findall(zerotest)
#       β[indx] .= exp(-500.0) # cheater's method
#   end
#
#   return β
# end


# function Gibbs_DL!(y, X, σ2, β0, DL::DirLap)
#
#     DL.β = rpost_normlmDiagBeta_beta1(y, X, σ2, (DL.τ^2 .* DL.ψ .* DL.ϕ.^2), β0)
#
#     ## these cannot accept exact zeros for β
#     DL.ψ = update_ψ(DL.ϕ, DL.β, DL.τ)
#     DL.τ = update_τ(DL.ϕ, DL.β, DL.α)
#     DL.ϕ = update_ϕ(DL.β, DL.α)
#
#     return nothing
# end
#
#
# K = 5
# βtrue = zeros(K)
# βtrue[[2, 3]] = [2.0, -3.0]
# # βtrue[[2, 8, 16]] = [2.0, -3.0, 6.0]
# β0true = 0.0
# σ2true = 0.1
# n = 20
# X = randn(n, K)
# y = β0true .+ X * βtrue .+ sqrt(σ2true)*randn(n)
#
# nsim = 1000
# betasim = zeros(nsim, K)
#
# DL = DirLap(K, 0.5, zeros(K), exp.(randn(K)), fill(1.0/float(K), K), 1.0)
# using LinearAlgebra
# Gibbs_DL!(y, X, σ2true, β0true, DL)
#
# for ii in 1:nsim
#
#     Gibbs_DL!(y, X, σ2true, β0true, DL)
#     betasim[ii,:] = deepcopy(DL.β)
#     if ii % 100 == 0
#         println("iter $(ii) of $(nsim)")
#     end
#
# end
#
# using RCall
# R"library(bayesplot)"
# @rput betasim
# R"colnames(betasim) = paste('beta', 1:ncol(betasim))"
# R"mcmc_intervals(betasim)"
#
# ## testing different sampling functions for beta
# betasim_a = permutedims(hcat([rpost_normlmDiagBeta_beta1(y, X, σ2true, fill(100.0, K), β0true) for ii = 1:nsim]...))
# # betasim_b = permutedims(hcat([rpost_normlmDiagBeta_beta1b(y, X, σ2true, fill(100.0, K), β0true) for ii = 1:nsim]...))
#
# @rput betasim_a
# R"colnames(betasim_a) = paste('beta', 1:ncol(betasim_a))"
# R"mcmc_intervals(betasim_a)"
#
#
# println(mean(betasim_a, dims=1))
# println(var(betasim_a, dims=1))
#
# println(mean(betasim_b, dims=1))
# println(var(betasim_b, dims=1))
#
# using BenchmarkTools
# @benchmark rpost_normlmDiagBeta_beta1(y, X, σ2true, fill(9.0, K), β0true) # faster
# @benchmark rpost_normlmDiagBeta_beta1b(y, X, σ2true, fill(9.0, K), β0true)

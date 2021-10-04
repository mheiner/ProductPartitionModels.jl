# test_nnlogdens.jl

sig2 = 1.5
m0 = 5.0
v0 = 2.0

mu = sqrt(v0) * randn() + m0
n = 10
y = sqrt(sig2) .* randn(n) .+ mu

sumy = sum(y)
sumy2 = sum(y.^2)

d0 = logdens_nn_marg(sig2, m0, v0, sumy, sumy2, n)

using Distributions
using LinearAlgebra
Sigma = fill(v0, n, n) + Diagonal(fill(sig2, n))
d1 = Distributions.logpdf(Distributions.MvNormal(fill(m0, n), Sigma), y)

d0 ≈ d1

## test parts
# logdet(Sigma)
# n*log(sig2) + log(1.0 + n*v0/sig2)
#
# Siginv = 1.0/sig2 * I - v0 / (sig2 * (n*v0 + sig2)) * ones(n, n)
# inv(Sigma)
#
# maximum(abs.(vec(Siginv - inv(Sigma))))
# d = y .- m0
#
# m1 = sumy2 - 2.0*m0*sumy + n*m0^2
# m2 = v0 * (sumy - n*m0)^2 / (n*v0 + sig2)
# (m1 - m2) / sig2
#
# d'*(Sigma \ d)
#
# -0.5 * ( n*log(2π) + logdet(Sigma) +  d'*(Sigma \ d) )

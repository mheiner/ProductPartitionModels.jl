# testSlice.jl

# run pkg> activate to be outside the package
using Plots
# using Plotly # run pkg> activate to be outside the package
Plots.PlotlyBackend()

using ProductPartitionModels
using StatsBase

n = 100
beta = [1.0, 5.0, -2.0, 0.0, 0.0]
p = length(beta)
sig = 0.5

X = randn(n,p)
y = X*beta + randn(n)*sig

n_iter = 2000
sim_b = Matrix{Float64}(undef, n_iter, p)
sim_sig = Vector{Float64}(undef, n_iter)

prior_beta_mean = zeros(p)
prior_beta_var = 1000*ones(p)
prior_sig_upper = 2.0

b_now = randn(p)
sig_now = rand()*2.0

for ii in 1:n_iter

    b_now, trash = ellipSlice(b_now, 
        prior_beta_mean, prior_beta_var,
        logtarget, TargetArgs_NormRegBeta(y, X, sig_now))
    sig_now, trash = shrinkSlice(sig_now, 0.0, prior_sig_upper,
        logtarget, TargetArgs_NormRegSig(y, X, b_now))

    sim_b[ii,:] = b_now
    sim_sig[ii] = sig_now

    if ii % 100 == 0 
        println("iter $ii of $n_iter")
    end
end

plot(sim_b)
beta
plot(sim_sig)
sig

n_burn = 1000
plot(sim_b[(n_burn+1):n_iter,:])
plot(sim_sig[(n_burn+1):n_iter])


# testMCMC.jl

using ProductPartitionModels
using StatsBase


n = 100
p = 2
prop_mis = 0.3
nmis = Int(floor(prop_mis*n*p))
nobs = n*p - nmis
X = Matrix{Union{Missing, Float64}}(missing, n, p)
obs_indx_sim = sample(1:(n*p), nobs; replace=false)
X[obs_indx_sim] = randn(nobs)
size(X)
X

Ctrue = vcat(fill(1, Int(floor(.5*n))), fill(2, Int(floor(.3*n))), fill(3, Int(floor(.2*n))))
length(Ctrue) == n
for i in findall(Ctrue .== 2)
    X[i,1:2] += [9.0, -2.0]
end
for i in findall(Ctrue .== 3)
  X[i,1:2] += [-8.0, 6.0]
end
X

α = 0.5
logα = log(α)
cohesion = Cohesion_CRP(logα, 0, true)
# similarity = Similarity_NiG_indep(0.0, 0.1, 1.0, 1.0)
similarity = Similarity_NiG_indep(0.0, 0.1, 4.0, 4.0)
C = deepcopy(Ctrue)
# C, K, S, lcohes, Xstat, lsimilar = sim_partition_PPMx(logα, X, similarity)
# C
# K
# S

K = maximum(C)
lcohes, Xstat, lsimilar = get_lcohlsim(C, X, cohesion, similarity)

## G0; controls only y|x
μ0 = 3.0
σ0 = 5.0
τ0 = 1.0 # scale of DL shrinkage
upper_σ = 3.0
G0 = Baseline_NormDLUnif(μ0, σ0, τ0, upper_σ)

y, μ, β, σ = sim_lik(C, X, similarity, Xstat, G0)

# y
μ
β
σ

# mod = Model_PPMx(y, X, C)
mod = Model_PPMx(y, X, 0) # C_init = 0 --> n clusters ; 1 --> 1 cluster
fieldnames(typeof(mod))
fieldnames(typeof(mod.state))
mod.state.C

mod.state.baseline = deepcopy(G0)
mod.state.cohesion = deepcopy(cohesion)
mod.state.similarity = deepcopy(similarity)

mod.prior
mod.prior.baseline.tau02_sh = 49.0
mod.prior.baseline.tau02_sc = 50.0

refresh!(mod.state, mod.y, mod.X, mod.obsXIndx, true)
mod.state.llik
mod.state.baseline.tau0 = 1.0

using Dates
timestart = Dates.now()

mcmc!(mod, 500,
    save=false,
    thin=1,
    n_procs=1,
    report_filename="",
    report_freq=100,
    update=[:C, :lik_params, :mu0, :sig0] #, :tau0]
)

etr(timestart; n_iter_timed=500, n_keep=1000, thin=1, outfilename="")

sims = mcmc!(mod, 1000,
    save=true,
    thin=1,
    n_procs=1,
    report_filename="",
    report_freq=100,
    update=[:C, :lik_params, :mu0, :sig0], #, :tau0],
    monitor=[:C, :mu, :sig, :beta, :mu0, :sig0] #, :tau0]
)

sims[1]
sims[1000]
sims[2][:lik_params][2]
mod.state.lik_params[1].mu

sims_llik = [ sims[ii][:llik] for ii in 1:length(sims) ]
sims_K = [ maximum(sims[ii][:C]) for ii in 1:length(sims) ]
Kmax = maximum(sims_K)
sims_S = permutedims( hcat( [ counts(sims[ii][:C], Kmax) for ii in 1:length(sims) ]...) )
sims_Sord = permutedims( hcat( [ sort(counts(sims[ii][:C], maximum(sims_K)), rev=true) for ii in 1:length(sims) ]...) )

using Plots
Plots.PlotlyBackend()


plot(sims_llik)
plot(sims_K)
plot(sims_S)

[ sims[ii][:C][18] for ii in 1:length(sims) ]
counts([ sims[ii][:C][57] for ii in 1:length(sims) ])

## monitoring lik_params is only useful if C is not changing
Kuse = 3

sims_mu = [ sims[ii][:lik_params][kk][:mu] for ii in 1:length(sims), kk in 1:Kuse ]
plot(sims_mu)
plot(sims_mu[:,1])
μ

sims_sig = [ sims[ii][:lik_params][kk][:sig] for ii in 1:length(sims), kk in 1:Kuse ]
plot(sims_sig)
plot(sims_sig[:,2])
σ

sims_beta = [ sims[ii][:lik_params][kk][:beta][j] for ii in 1:length(sims), kk in 1:Kuse, j in 1:p ]
plot(reshape(sims_beta[:,1,:], (length(sims), p)))
plot(reshape(sims_beta[:,2,:], (length(sims), p)))
plot(reshape(sims_beta[:,3,:], (length(sims), p)))
plot(reshape(sims_beta[:,4,:], (length(sims), p)))
β

plot(reshape(sims_beta[:,3,2], (length(sims))))

sims_mu0 = [ sims[ii][:baseline][:mu0] for ii in 1:length(sims) ]
mod.state.baseline.mu0
plot(sims_mu0)
mod.prior.baseline.mu0_mean
mod.prior.baseline.mu0_sd
μ0

sims_sig0 = [ sims[ii][:baseline][:sig0] for ii in 1:length(sims) ]
mod.state.baseline.sig0
plot(sims_sig0)
mod.prior.baseline.sig0_upper
σ0

sims_tau0 = [ sims[ii][:baseline][:tau0] for ii in 1:length(sims) ]
mod.state.baseline.tau0
plot(sims_tau0[findall(sims_tau0 .< 5.0)])


using Plotly # run pkg> activate to be outside the package

C_use = deepcopy(C)
C_use = deepcopy(mod.state.C)

indx_cc = findall( [ all(.!ismissing.(X[i,1:2])) for i in 1:n ] )
indx_x1m = findall( [ ismissing(X[i,1]) & !ismissing(X[i,2]) for i in 1:n ] )
indx_x2m = findall( [ !ismissing(X[i,1]) & ismissing(X[i,2]) for i in 1:n ] )
indx_allmiss = findall( [ all(ismissing.(X[i,:])) for i in 1:n ] )

colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

trace1 = Plotly.scatter3d(Dict(
  :x => convert(Vector{Float64}, X[indx_cc,1]),
  :y => convert(Vector{Float64}, X[indx_cc,2]),
  :z => y[indx_cc],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_use[indx_cc]], :size => 5.0)
))
Plotly.plot([trace1])

trace2 = Plotly.scatter3d(Dict(
  :x => convert(Vector{Float64}, X[indx_x2m,1]),
  :y => zeros(length(indx_x2m)),
  :z => y[indx_x2m],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_use[indx_x2m]], :size => 5.0)
))
Plotly.plot([trace2])

trace3 = Plotly.scatter3d(Dict(
  :x => zeros(length(indx_x2m)),
  :y => convert(Vector{Float64}, X[indx_x1m,2]),
  :z => y[indx_x1m],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_use[indx_x1m]], :size => 5.0)
))
Plotly.plot([trace3])

trace4 = Plotly.scatter3d(Dict(
  :x => zeros(length(indx_allmiss)),
  :y => zeros(length(indx_allmiss)),
  :z => y[indx_allmiss],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_use[indx_allmiss]], :size => 5.0)
))
Plotly.plot([trace4])


### test prediction

## in sample

Ypred_is = postPred(mod, sims)
using RCall
@rput Ypred_is y
R"hist(Ypred_is[,90], breaks=20); abline(v=y[90], col='blue')"
R"intv = apply(Ypred_is, 2, quantile, c(0.05, 0.95))"
R"cover = y < intv[2,] & y > intv[1,]"
R"mean(cover)"

## import from R?

using RCall
R"
Xr = matrix(rnorm(20), ncol=2);
Xr[sample(prod(dim(Xr)), 5)] = NA;
Xr
"

@rget Xr # works!
typeof(Xr)
typeof(Xr) <: Union{Matrix{T}, Matrix{Union{T, Missing}}} where T <: Real
Xrr = Matrix(Xr)
typeof(Xrr) <: Union{Matrix{T}, Matrix{Union{T, Missing}}} where T <: Real

Xrr[1,:] = [-5.0, missing]

Ypred, Cpred = postPred(Xrr, mod, sims)

Xrr
using RCall
@rput Ypred
StatsBase.counts(Cpred[:,1], 0:10)
R"hist(Ypred[,1], breaks=20)"



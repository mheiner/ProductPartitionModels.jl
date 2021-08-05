# testMCMC.jl

using Plots
Plots.PlotlyBackend()
using Plotly # run pkg> activate to be outside the package

using ProductPartitionModels
using StatsBase

# n = 30
# p = 5
# nmis = 10
# nobs = n*p - nmis
# X = Matrix{Union{Missing, Float64}}(missing, n, p)
# obs_indx_sim = sample(1:(n*p), nobs; replace=false)
# X[obs_indx_sim] = randn(nobs)
# size(X)
# X
# for i in (n-10+1):n
#     X[i,:] += [1.0, 3.0, 0.0, 0.0, -5.0]
# end
# X

# X = [0.1 10.0; -0.1 11.0; -3.0 0.0; -2.2 0.1; -2.5 -0.2]

n = 1000
p = 2
nmis = 500
nobs = n*p - nmis
X = Matrix{Union{Missing, Float64}}(missing, n, p)
obs_indx_sim = sample(1:(n*p), nobs; replace=false)
X[obs_indx_sim] = randn(nobs)
size(X)
X
for i in (1):(200)
    X[i,:] += [1.0, -5.0]
end
for i in (201):(500)
    X[i,:] += [4.0, 1.0]
end
X

# C, K, S, lcohes, Xstat, lsimilar = sim_partition_PPMx(logα, X, similarity)
# C
# K
# S

## alternatively, fix C
α = 1.0
C = vcat(fill(1, 200), fill(2, 300), fill(3, 500))
K = maximum(C)
cohesion = Cohesion_CRP(log(α), 0, true)
similarity = Similarity_NiG_indep(0.0, 0.1, 1.0, 1.0)
lcohes, Xstat, lsimilar = get_lcohlsim(C, X, cohesion, similarity)

## G0; controls only y|x
μ0 = 0.0
σ0 = 20.0
τ0 = 5.0 # scale of DL shrinkage
upper_σ = 3.0
G0 = Baseline_NormDLUnif(μ0, σ0, τ0, upper_σ)

y, μ, β, σ = sim_lik(C, X, similarity, Xstat, G0)

y
μ
β
σ


mod = Model_PPMx(y, X, C)
fieldnames(typeof(mod))
mod.X
mod.obsXIndx[2]
mod.obsXIndx[2].n_obs
fieldnames(typeof(mod.state))
mod.state.C
mod.state.lik_params[1]
mod.state.baseline
mod.state.iter

mod.state.baseline = deepcopy(G0)
mod.state.cohesion = deepcopy(cohesion)
mod.state.similarity = deepcopy(similarity)

refresh!(mod.state, mod.y, mod.X, mod.obsXIndx)
mod.state.llik

mcmc!(mod, 1000,
    save=false,
    thin=1,
    n_procs=1,
    report_filename="out_progress.txt",
    report_freq=10000,
    update=[:lik_params],
    monitor=[:mu, :sig, :beta]
)

sims = mcmc!(mod, 1000,
    save=true,
    thin=1,
    n_procs=1,
    report_filename="out_progress.txt",
    report_freq=10000,
    update=[:lik_params],
    monitor=[:mu, :sig, :beta]
)

sims[1]
sims[1000]
sims[2][:lik_params][2]
mod.state.lik_params[1].mu


sims_mu = [ sims[ii][:lik_params][kk][:mu] for ii in 1:length(sims), kk in 1:K ]
plot(sims_mu)
plot(sims_mu[:,1])
μ

sims_sig = [ sims[ii][:lik_params][kk][:sig] for ii in 1:length(sims), kk in 1:K ]
plot(sims_sig)
plot(sims_sig[:,1])
σ

sims_beta = [ sims[ii][:lik_params][kk][:beta][j] for ii in 1:length(sims), kk in 1:K, j in 1:p ]
plot(reshape(sims_beta[:,1,:], (length(sims), p)))
plot(reshape(sims_beta[:,2,:], (length(sims), p)))
plot(reshape(sims_beta[:,3,:], (length(sims), p)))
β

plot(reshape(sims_beta[:,3,2], (length(sims))))


indx_cc = findall( [ all(.!ismissing.(X[i,:])) for i in 1:n ] )
indx_x1m = findall( [ ismissing(X[i,1]) & !ismissing(X[i,2]) for i in 1:n ] )
indx_x2m = findall( [ !ismissing(X[i,1]) & ismissing(X[i,2]) for i in 1:n ] )
indx_allmiss = findall( [ all(ismissing.(X[i,:])) for i in 1:n ] )

colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

trace1 = scatter3d(Dict(
  :x => convert(Vector{Float64}, X[indx_cc,1]),
  :y => convert(Vector{Float64}, X[indx_cc,2]),
  :z => y[indx_cc],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C[indx_cc]], :size => 5.0)
))
plot([trace1])

trace2 = scatter3d(Dict(
  :x => convert(Vector{Float64}, X[indx_x2m,1]),
  :y => zeros(length(indx_x2m)),
  :z => y[indx_x2m],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C[indx_x2m]], :size => 5.0)
))
plot([trace2])

trace3 = scatter3d(Dict(
  :x => zeros(length(indx_x2m)),
  :y => convert(Vector{Float64}, X[indx_x1m,2]),
  :z => y[indx_x1m],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C[indx_x1m]], :size => 5.0)
))
plot([trace3])


trace4 = scatter3d(Dict(
  :x => zeros(length(indx_allmiss)),
  :y => zeros(length(indx_allmiss)),
  :z => y[indx_allmiss],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C[indx_allmiss]], :size => 5.0)
))
plot([trace4])





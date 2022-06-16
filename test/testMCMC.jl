# testMCMC.jl

## go into package mode and enter: activate
## to run this code without modifying the ProductPartitionModels package

using ProductPartitionModels
using StatsBase
using Random

using RCall
R"load('~/Documents/research/PPMx_missing/data/MSPE_ncov_10_missType_MAR_perMiss_0_dataType_1_data_1.RData')"
R"load('~/research/PPMx_missing/data/MSPE_ncov_10_missType_MAR_perMiss_0.1_dataType_1_data_1.RData')"
R"load('~/Documents/research/PPMx_missing/ozone/postsim/ozone_nvar2_rep36.RData'); Xmat=Xcontn; Xpred=Xcontt"
@rget ytn Xmat Xpred
y = float(deepcopy(ytn))
X = Matrix(float(deepcopy(Xmat)))
Xpred = Matrix(float(deepcopy(Xpred)))
n, p = size(X)

@rget y_mean y_sd

@rget ytt
y_mean = mean(vcat(ytt, ytn))

y_sd = StatsBase.std(vcat(ytt, ytn))

y_use = (y .- y_mean) ./ y_sd

# X = Matrix{Union{Missing, Float64}}(missing, n, p)
# for i in 1:n
#     for p in 1:p
#         if !ismissing(Xmat[i,p])
#             X[i,p] = 0.0
#             X[i,p] += float(Xmat[i,p])
#         end
#     end
# end

Random.seed!(220607)

n = 100
p = 2
prop_mis = 0.2
nmis = Int(floor(prop_mis*n*p))
nobs = n*p - nmis
X = Matrix{Union{Missing, Float64}}(missing, n, p)
obs_indx_sim = sample(1:(n*p), nobs; replace=false)

X[obs_indx_sim] = randn(nobs)
X[obs_indx_sim] = 0.1*randn(nobs)

size(X)
X

Ctrue = vcat(fill(1, Int(floor(.5*n))), fill(2, Int(floor(.3*n))), fill(3, Int(floor(.2*n))))
length(Ctrue) == n
for i in findall(Ctrue .== 2)
  X[i,1:2] += [0.5, -0.25]
  # X[i,1:2] += [3.0, -1.5]
    # X[i,1:2] += [9.0, -2.0]
end
for i in findall(Ctrue .== 3)
  X[i,1:2] += [0.0, 0.4]
  # X[i,1:2] += [0.0, 3.0]
  # X[i,1:2] += [-8.0, 6.0]
end
X

# α = 0.5
α = 1.0
logα = log(α)
cohesion = Cohesion_CRP(logα, 0, true)

# similarity = Similarity_NiG_indep(0.0, 0.5, 4.0, 2.0)
# similarity = Similarity_NiG_indep(0.0, 0.1, 1.0, 1.0)
# similarity = Similarity_NiG_indep(0.0, 0.1, 4.0, 4.0)
similarity = Similarity_NiG_indep(0.0, 2.0, 4.0, 0.2)

# similarity = Similarity_NN(sqrt(0.5), 0.0, 1.0)
# similarity = Similarity_NN(1.0, 0.0, 1.0)

C = deepcopy(Ctrue)
# C, K, S, lcohes, Xstat, lsimilar = sim_partition_PPMx(logα, X, similarity)
# C
# K
# S

K = maximum(C)
lcohes, Xstat, lsimilar = get_lcohlsim(C, X, cohesion, similarity)

## G0; controls only y|x
μ0 = 0.5
σ0 = 1.0
τ0 = 1.0 # scale of DL shrinkage
σ_upper = 10.0
G0 = Baseline_NormDLUnif(μ0, σ0, τ0, σ_upper)

y, μ, β, σ = sim_lik(C, X, similarity, Xstat, G0)

y_use = deepcopy(y)
# y
μ
β
σ

# model = Model_PPMx(y, X, C)
# model = Model_PPMx(y_use, X, 0, similarity_type=:NN, init_lik_rand=false) # C_init = 0 --> n clusters ; 1 --> 1 cluster
model = Model_PPMx(y_use, X, 0, similarity_type=:NiG_indep, init_lik_rand=false) # C_init = 0 --> n clusters ; 1 --> 1 cluster
fieldnames(typeof(model))
fieldnames(typeof(model.state))
model.state.C

model.state.baseline = deepcopy(G0)
model.state.cohesion = deepcopy(cohesion)
model.state.similarity = deepcopy(similarity)

# model.state.similarity = Similarity_NN(sqrt(0.5), 0.0, 1.0)
# model.state.baseline = Baseline_NormDLUnif(0.0, 0.1, 0.1, 10.0/y_sd)
# model.prior.baseline = Prior_baseline_NormDLUnif(0.0, 10.0/y_sd, 10.0/y_sd)
# model.prior.baseline = Prior_baseline_NormDLUnif(0.0, 10.0/y_sd, 30.0/y_sd)

model.prior
# model.prior.baseline.tau02_sh = 49.0 # got rid of this
# model.prior.baseline.tau02_sc = 50.0

refresh!(model.state, model.y, model.X, model.obsXIndx, true)
model.state.llik
model.state.baseline.tau0 = 1.0
model.state.baseline.tau0 = 0.1

using Dates
timestart = Dates.now()

mcmc!(model, 1000,
    save=false,
    thin=1,
    n_procs=1,
    report_filename="",
    report_freq=100,
    # update=[:C, :mu, :sig, :mu0, :sig0]
    update=[:C, :mu, :sig, :beta, :mu0, :sig0]
)

etr(timestart; n_iter_timed=1000, n_keep=1000, thin=1, outfilename="")

sims = mcmc!(model, 1000,
    save=true,
    thin=5,
    n_procs=1,
    report_filename="",
    report_freq=100,
    # update=[:C, :mu, :sig, :mu0, :sig0],
    update=[:C, :mu, :sig, :beta, :mu0, :sig0],
    monitor=[:C, :mu, :sig, :beta, :mu0, :sig0]
)

sims0 = deepcopy(sims)
sims == sims0

sims[1]
sims[1000]
sims[2][:lik_params][2]
model.state.lik_params[1].mu

sims_llik = [ sims[ii][:llik] for ii in 1:length(sims) ]
sims_K = [ maximum(sims[ii][:C]) for ii in 1:length(sims) ]
Kmax = maximum(sims_K)
C_mat = permutedims( hcat( [ sims[ii][:C] for ii in 1:length(sims) ]...) )
sims_S = permutedims( hcat( [ counts(sims[ii][:C], Kmax) for ii in 1:length(sims) ]...) )
sims_Sord = permutedims( hcat( [ sort(counts(sims[ii][:C], maximum(sims_K)), rev=true) for ii in 1:length(sims) ]...) )

using Plots
Plots.PlotlyBackend()


Plots.plot(sims_llik)
Plots.plot(sims_K)
Plots.plot(sims_S)

[ sims[ii][:C][18] for ii in 1:length(sims) ]
counts([ sims[ii][:C][57] for ii in 1:length(sims) ])

## monitoring lik_params is only useful if C is not changing
Kuse = 3

sims_mu_mat = [ sims[ii][:lik_params][sims[ii][:C][i]][:mu] for ii in 1:length(sims), i in 1:model.n ]
sims_mu = [ sims[ii][:lik_params][kk][:mu] for ii in 1:length(sims), kk in 1:Kuse ]
μ
Plots.plot(sims_mu)
plot(sims_mu[:,1])

sims_sig = [ sims[ii][:lik_params][kk][:sig] for ii in 1:length(sims), kk in 1:Kuse ]
σ
Plots.plot(sims_sig)
plot(sims_sig[:,2])

p = size(X,2)
sims_beta_array = [ sims[ii][:lik_params][sims[ii][:C][i]][:beta][kk] for ii in 1:length(sims), i in 1:model.n, kk in 1:model.p ]
sims_beta = [ sims[ii][:lik_params][kk][:beta][j] for ii in 1:length(sims), kk in 1:Kuse, j in 1:p ]
β
Plots.plot(reshape(sims_beta[:,1,:], (length(sims), p)))
Plots.plot(reshape(sims_beta[:,2,:], (length(sims), p)))
Plots.plot(reshape(sims_beta[:,3,:], (length(sims), p)))
plot(reshape(sims_beta[:,4,:], (length(sims), p)))

plot(reshape(sims_beta[:,3,2], (length(sims))))

sims_mu0 = [ sims[ii][:baseline][:mu0] for ii in 1:length(sims) ]
model.state.baseline.mu0
plot(sims_mu0)
model.prior.baseline.mu0_mean
model.prior.baseline.mu0_sd
μ0

sims_sig0 = [ sims[ii][:baseline][:sig0] for ii in 1:length(sims) ]
model.state.baseline.sig0
plot(sims_sig0)
model.prior.baseline.sig0_upper
σ0

# sims_tau0 = [ sims[ii][:baseline][:tau0] for ii in 1:length(sims) ]
# model.state.baseline.tau0
# plot(sims_tau0[findall(sims_tau0 .< 5.0)])




using Plotly # run pkg> activate to be outside the package
using RCall

## use SALSO to find best c
R"library('salso')"
@rput C_mat
R"Cpsm = psm(C_mat)"
R"Crsalso = salso(C_mat)"
@rget Crsalso


symbols3d = ["circle", "cross", "diamond", "circle-open", "cross-open", "diamond-open"]
colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

x1range = collect(extrema(skipmissing(X[:,1]))) + [-1, 1]
x2range = collect(extrema(skipmissing(X[:,2]))) + [-1, 1]
yrange = collect(extrema(y)) + [-1, 1]


# C_use = deepcopy(C)
# C_use = deepcopy(model.state.C)

StatsBase.countmap(Crsalso)
misclass = (Ctrue .!= Crsalso) .* 1

C_color = deepcopy(Ctrue)
C_shape = deepcopy(Crsalso) + 3*misclass # so that misclassified obs are open-style symbols


indx_cc = findall( [ all(.!ismissing.(X[i,1:2])) for i in 1:n ] )
indx_x1m = findall( [ ismissing(X[i,1]) & !ismissing(X[i,2]) for i in 1:n ] )
indx_x2m = findall( [ !ismissing(X[i,1]) & ismissing(X[i,2]) for i in 1:n ] )
indx_allmiss = findall( [ all(ismissing.(X[i,:])) for i in 1:n ] )


trace1 = Plotly.scatter3d(Dict(
  :x => convert(Vector{Float64}, X[indx_cc,1]),
  :y => convert(Vector{Float64}, X[indx_cc, 2]),
  :z => y[indx_cc],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_color[indx_cc]], :size => 5.0, :symbol => symbols3d[C_shape[indx_cc]])
))
Plotly.plot([trace1], Layout(height=700, width=700, scene_aspectratio=attr(x=1, y=1, z=1), 
            scene=attr(xaxis_title="x1", yaxis_title="x2", zaxis_title="y", 
            xaxis=attr(range=x1range), yaxis=attr(range=x2range), zaxis=attr(range=yrange)),
            title="Complete cases"))

trace1_flat = Plotly.scatter(Dict(
  :x => convert(Vector{Float64}, X[indx_cc, 1]),
  :y => convert(Vector{Float64}, X[indx_cc, 2]),
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_color[indx_cc]], :size => 5.0, :symbol => symbols3d[C_shape[indx_cc]])
))
Plotly.plot([trace1_flat], Layout(height=500, width=500, scene_aspectratio=attr(x=1, y=1), 
            xaxis_range=x1range, yaxis_range=x2range,
            xaxis_title="x1", yaxis_title="x2", title="Complete cases; x1 and x2"))


# trace2 = Plotly.scatter3d(Dict(
#   :x => convert(Vector{Float64}, X[indx_x2m, 1]),
#   :y => zeros(length(indx_x2m)),
#   :z => y[indx_x2m],
#   :opacity => 0.7,
#   :showscale => false,
#   :mode => "markers",
#   :marker => Dict(:color => colors[C_use[indx_x2m]], :size => 5.0, 
#   :symbol => symbols3d[C_use[indx_x2m]])
# ))
# Plotly.plot([trace2], Layout(height=700, width=700, scene_aspectratio=attr(x=1, y=1, z=1), 
#             scene=attr(xaxis_title="x1", yaxis_title="x2", zaxis_title="y", 
#             xaxis=attr(range=x1range), yaxis=attr(range=x2range), zaxis=attr(range=yrange))))

trace2_flat = Plotly.scatter(Dict(
  :x => convert(Vector{Float64}, X[indx_x2m, 1]),
  :y => y[indx_x2m],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_color[indx_x2m]], :size => 5.0, :symbol => symbols3d[C_shape[indx_x2m]])
))
Plotly.plot([trace2_flat], Layout(height=500, width=500, scene_aspectratio=attr(x=1, y=1), 
            xaxis_range=x1range, yaxis_range=yrange,
            xaxis_title="x1", yaxis_title="y", title="x2 missing"))



# trace3 = Plotly.scatter3d(Dict(
#   :x => zeros(length(indx_x2m)),
#   :y => convert(Vector{Float64}, X[indx_x1m, 2]),
#   :z => y[indx_x1m],
#   :opacity => 0.7,
#   :showscale => false,
#   :mode => "markers",
#   :marker => Dict(:color => colors[C_use[indx_x1m]], :size => 5.0)
# ))
# Plotly.plot([trace3], Layout(height=700, width=700, scene_aspectratio=attr(x=1, y=1, z=1)))

trace3_flat = Plotly.scatter(Dict(
  :x => convert(Vector{Float64}, X[indx_x1m, 2]),
  :y => y[indx_x1m],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_color[indx_x1m]], :size => 5.0, :symbol => symbols3d[C_shape[indx_x1m]])
))
Plotly.plot([trace3_flat], Layout(height=500, width=500, scene_aspectratio=attr(x=1, y=1), 
            xaxis_range=x2range, yaxis_range=yrange,
            xaxis_title="x2", yaxis_title="y", title="x1 missing"))



# trace4 = Plotly.scatter3d(Dict(
#   :x => zeros(length(indx_allmiss)),
#   :y => zeros(length(indx_allmiss)),
#   :z => y[indx_allmiss],
#   :opacity => 0.7,
#   :showscale => false,
#   :mode => "markers",
#   :marker => Dict(:color => colors[C_use[indx_allmiss]], :size => 5.0)
# ))
# Plotly.plot([trace4], Layout(height=700, width=700, scene_aspectratio=attr(x=1, y=1, z=1)))

trace4_flat = Plotly.scatter(Dict(
  :x => zeros(length(indx_allmiss)),
  :y => y[indx_allmiss],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_color[indx_allmiss]], :size => 9.0, :symbol => symbols3d[C_shape[indx_allmiss]])
))
Plotly.plot([trace4_flat], Layout(height=500, width=500, scene_aspectratio=attr(x=1, y=1), 
            xaxis_range=[-0.5, 0.5], yaxis_range=yrange,
            xaxis_title="", yaxis_title="y", title="x1 and x2 missing",
            xaxis_tickmode="array", xaxis_tickvals=[0], xaxis_ticktext=[""]))





### test prediction

## in sample

Ypred_is = postPred(model, sims)
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

# Ypred, Cpred, Mpred = postPred(Xrr, model, sims)
Ypred, Cpred, Mpred = postPred(Xpred, model, sims)
sims == sims0


Xrr
using RCall
@rput Ypred
StatsBase.counts(Cpred[:,1], 0:10)
R"hist(Ypred[,1], breaks=20)"

R"  xseq = seq(from=-2.0, to=3.0, length=50)
    Xsurf = as.matrix(expand.grid(xseq, xseq)) # fit a surface
"
@rget xseq Xsurf

Ysurf, Csurf, Msurf = postPred(Xsurf, model, sims)
sims == sims0
Ysurf_mean = mean(Ysurf, dims=1)[1,:]
Ysurf_mean = mean(Ysurf, dims=1)[1,:] * y_sd .+ y_mean

using Plotly

Ysurf_mean_mat = Matrix(reshape(Ysurf_mean, fill(length(xseq),2)...))
trace1 = surface(Dict(
  :x => xseq,
  :y => xseq,
  :z => Ysurf_mean_mat,
  :colorscale => "Viridis",
  :opacity => 0.4,
  :showscale => false,
  :type => "surface"
))
data = [trace1]
Plotly.plot(data)

indx_cc = findall( [ all(.!ismissing.(X[i,1:2])) for i in 1:n ] )
trace2 = Plotly.scatter3d(Dict(
  :x => convert(Vector{Float64}, X[indx_cc,1]),
  :y => convert(Vector{Float64}, X[indx_cc,2]),
  :z => y[indx_cc],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers"
  # :marker => Dict(:color => colors[1], :size => 5.0)
))

data = [trace1, trace2]
Plotly.plot(data)

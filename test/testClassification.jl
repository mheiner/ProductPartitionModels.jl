# testClassification.jl

## go into package mode and enter: activate
## to run this code without modifying the ProductPartitionModels package
using Pkg
Pkg.activate()

using ProductPartitionModels
using StatsBase
using Random

Random.seed!(220607)

n = 500
p = 2
prop_mis = 0.25
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
    # X[i,1:2] += [3.0, -1.5]
    X[i,1:2] += [-3.0, -1.5]
end
for i in findall(Ctrue .== 3)
#   X[i,1:2] += [0.0, 3.0]
  X[i,1:2] += [1.0, 3.0]
end
X

α = 1.0
logα = log(α)
cohesion = Cohesion_CRP(logα, 0, true)
similarity = Similarity_NN(1.0, 0.0, 1.0)

C = deepcopy(Ctrue)
K = maximum(C)
lcohes, Xstat, lsimilar = get_lcohlsim(C, X, cohesion, similarity)

## G0; controls only y|x
μ0 = 0.0
σ0 = 5.0
τ0 = 1.0 # scale of DL shrinkage
σ_upper = 10.0
G0 = Baseline_NormDLUnif(μ0, σ0, τ0, σ_upper)

lik_params_tmp = [ simpri_lik_params(G0, p) for k in 1:K ]

lik_params_tmp[1].mu = 1.5
lik_params_tmp[2].mu = 2.5
lik_params_tmp[3].mu = -5.0

lik_params_tmp[1].sig = 1.2
lik_params_tmp[2].sig = 0.5
lik_params_tmp[3].sig = 0.8

lik_params_tmp[1].beta = [-0.9, 2.0]
lik_params_tmp[2].beta = [-0.3, -1.0]
lik_params_tmp[3].beta = [0.7, 0.0]

y, μ, β, σ = sim_lik(C, X, similarity, Xstat, lik_params_tmp)

y_use = deepcopy(y)
μ
β
σ

model = Model_PPMx(y_use, X, 0, similarity_type=:NN, init_lik_rand=false) # C_init = 0 --> n clusters ; 1 --> 1 cluster
fieldnames(typeof(model))
fieldnames(typeof(model.state))
model.state.C

model.state.baseline = deepcopy(G0)
model.state.cohesion = deepcopy(cohesion)
model.state.similarity = deepcopy(similarity)

model.prior

refresh!(model.state, model.y, model.X, model.obsXIndx, true)
model.state.llik
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

C_mat = permutedims( hcat( [ sims[ii][:C] for ii in 1:length(sims) ]...) )


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

StatsBase.countmap(Crsalso)
misclass = (Ctrue .!= Crsalso) .* 1

C_color = deepcopy(Ctrue)
C_shape = deepcopy(Ctrue)
C_shape = deepcopy(Crsalso) + 3*misclass # so that misclassified obs are open-style symbols


indx_cc = findall( [ all(.!ismissing.(X[i,1:2])) for i in 1:n ] )
indx_x1m = findall( [ ismissing(X[i,1]) & !ismissing(X[i,2]) for i in 1:n ] )
indx_x2m = findall( [ !ismissing(X[i,1]) & ismissing(X[i,2]) for i in 1:n ] )
indx_allmiss = findall( [ all(ismissing.(X[i,:])) for i in 1:n ] )

length(indx_cc)
length(indx_x1m)
length(indx_x2m)
length(indx_allmiss)

sum(misclass[indx_cc])
sum(misclass[indx_x1m])
sum(misclass[indx_x2m])
sum(misclass[indx_allmiss])


trace1 = Plotly.scatter3d(Dict(
  :x => convert(Vector{Float64}, X[indx_cc,1]),
  :y => convert(Vector{Float64}, X[indx_cc, 2]),
  :z => y[indx_cc],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_color[indx_cc]], :size => 4.0, :symbol => symbols3d[C_shape[indx_cc]])
))
pp = Plotly.plot([trace1], Layout(height=460, width=460, scene_aspectratio=attr(x=1.0, y=1.0, z=1.0), 
            scene=attr(xaxis_title="x1", yaxis_title="x2", zaxis_title="y", 
            xaxis=attr(range=x1range), yaxis=attr(range=x2range), zaxis=attr(range=yrange)),
            title="Complete cases"))
            

relayout!(pp, scene_camera=attr(up=attr(x=0, y=0, z=1), center=attr(x=0.0, y=-0.1, z=-0.25), eye=attr(x=1.5, y=-1.5, z=0.7), margin=attr(l=-100, r=-100, b=-100, t=-100))) # margin doesn't seem to do anything
savefig(pp, "/Users/mjheiner/Desktop/testC_1a.pdf", height=460, width=460)
            
relayout!(pp, scene_camera=attr(up=attr(x=0, y=0, z=1), center=attr(x=0.0, y=-0.0, z=-0.4), eye=attr(x=0.7, y=-2.3, z=0.6), margin=attr(l=-100, r=-100, b=-100, t=-100))) # margin doesn't seem to do anything
savefig(pp, "/Users/mjheiner/Desktop/testC_1b.pdf", height=460, width=460)

relayout!(pp, scene_camera=attr(up=attr(x=0, y=0, z=1), center=attr(x=0.0, y=0.1, z=-0.4), eye=attr(x=2.2, y=-0.7, z=0.6), margin=attr(l=-100, r=-100, b=-100, t=-100))) # margin doesn't seem to do anything
savefig(pp, "/Users/mjheiner/Desktop/testC_1c.pdf", height=460, width=460)

trace1_flat = Plotly.scatter(Dict(
  :x => convert(Vector{Float64}, X[indx_cc, 1]),
  :y => convert(Vector{Float64}, X[indx_cc, 2]),
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_color[indx_cc]], :size => 7.0, :symbol => symbols3d[C_shape[indx_cc]])
))
pp = Plotly.plot([trace1_flat], Layout(height=350, width=350, scene_aspectratio=attr(x=1, y=1), 
            xaxis_range=x1range, yaxis_range=x2range,
            xaxis_title="x1", yaxis_title="x2", title="Complete cases; x1 and x2"))

savefig(pp, "/Users/mjheiner/Desktop/testC_1flat.pdf", height=350, width=350)


trace2_flat = Plotly.scatter(Dict(
  :x => convert(Vector{Float64}, X[indx_x2m, 1]),
  :y => y[indx_x2m],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_color[indx_x2m]], :size => 7.0, :symbol => symbols3d[C_shape[indx_x2m]])
))
pp = Plotly.plot([trace2_flat], Layout(height=350, width=350, scene_aspectratio=attr(x=1, y=1), 
            xaxis_range=x1range, yaxis_range=yrange,
            xaxis_title="x1", yaxis_title="y", title="x2 missing"))

savefig(pp, "/Users/mjheiner/Desktop/testC_2flat.pdf", height=350, width=350)



trace3_flat = Plotly.scatter(Dict(
  :x => convert(Vector{Float64}, X[indx_x1m, 2]),
  :y => y[indx_x1m],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_color[indx_x1m]], :size => 7.0, :symbol => symbols3d[C_shape[indx_x1m]])
))
pp = Plotly.plot([trace3_flat], Layout(height=350, width=350, scene_aspectratio=attr(x=1, y=1), 
            xaxis_range=x2range, yaxis_range=yrange,
            xaxis_title="x2", yaxis_title="y", title="x1 missing",
            xaxis_tickmode="array", 
            xaxis_tickvals=range(start=round(x2range[1]), step=2, stop=round(x2range[2]))))

savefig(pp, "/Users/mjheiner/Desktop/testC_3flat.pdf", height=350, width=350)


trace4_flat = Plotly.scatter(Dict(
  :x => 0.15*abs.(rand(length(indx_allmiss))) .* (2*(StatsBase.ordinalrank(y[indx_allmiss]) .% 2) .- 1),
  :y => y[indx_allmiss],
  :opacity => 0.7,
  :showscale => false,
  :mode => "markers",
  :marker => Dict(:color => colors[C_color[indx_allmiss]], :size => 8.0, :symbol => symbols3d[C_shape[indx_allmiss]])
))
pp = Plotly.plot([trace4_flat], Layout(height=350, width=200, scene_aspectratio=attr(x=1, y=1), 
            xaxis_range=[-0.5, 0.5], yaxis_range=yrange,
            xaxis_title="", yaxis_title="y", title="x1 and x2 missing",
            xaxis_tickmode="array", xaxis_tickvals=[0], xaxis_ticktext=[""]))

savefig(pp, "/Users/mjheiner/Desktop/testC_4flat.pdf", height=350, width=200)

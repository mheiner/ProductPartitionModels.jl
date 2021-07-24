# manualtests.jl

using Plots
using Plotly # run pkg> activate to be outside the package
Plots.PlotlyBackend()

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
nmis = 300
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

# obs_indx = [ findall(.!ismissing.(X[i,:])) for i in 1:size(X,1) ]

# X[1:5,:]
# obs_indx[1:5]

# X[(n-10+1):n,:]
# obs_indx[(n-10+1):n]

# [ mean(skipmissing(X[:,j])) for j in 1:size(X,2) ]

similarity = Similarity_NiG_indep(0.0, 1.0, 4.0, 4.0)
# similarity = Similarity_NiG_indep(0.0, 0.1, 1.0, 1.0)
α = 0.5
logα = log(α)

C, K, S, lcohes, Xstat, similar = sim_partition_PPMx(logα, X, similarity)
C
K
S

## G0; controls only y|x
μ0 = 0.0
σ0 = 20.0
τ0 = 1.0 # scale of DL shrinkage
upper_σ = 10.0
G0 = Baseline_NormDLUnif(μ0, σ0, τ0, upper_σ)

y, μ, β, σ = sim_lik(C, X, similarity, Xstat, G0)

y

xlim = [minimum(skipmissing(X[:,1])), maximum(skipmissing(X[:,1]))]
ylim = [minimum(skipmissing(X[:,2])), maximum(skipmissing(X[:,2]))]
zlim = [minimum(y), maximum(y)]
layout = Layout( # broken?
    Dict(
    :scene => Dict(
        :xaxis => Dict(:range => xlim), 
        :yaxis => Dict(:range => ylim), 
        :zaxis => Dict(:range => zlim)
        )
    )
)

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





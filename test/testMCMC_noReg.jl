# testMCMC_noReg.jl

# testMCMC.jl

## go into package mode and enter: activate
## to run this code without modifying the ProductPartitionModels package

using Pkg
Pkg.activate()


using ProductPartitionModels
using StatsBase
using Random

using RCall
R"library(matrixStats); library(salso); source('test/posterior_by_hand_functions.R')"

# X = reshape([-2.0, -1.5, 0.5], 3, 1)
# y = [-1.0, 1.0, -2.5]
# n = 3
# p = 1

n = 4
p = 2

X = randn(n,p)
y = randn(n)

X = fill(missing, (3,1))
y = [-0.5, 0.0, 1.0]
n, p = size(X)

# X[3:n,1] .+= 2.0
# y[3:n] .+= 2.0

# X[n, 2] -= 5.0
# y[n] += 5.0

# X
# y

## similarity
alpha = 1.0
v = 1.0^2
m0 = 0.0
s20 = 0.1^2
sc_prec0 = 0.1
nu0 = 4.0

## baseline
sig = 1.0
sig2 = sig^2
mu0 = 0.0
sig0 = 1.0
sig20 = sig0^2
sigmax = 10.0

@rput n p y X alpha v m0 s20 sc_prec0 nu0 sig2 mu0 sig20

R"(rho = enumerate.partitions(n));"
R"(B = nrow(rho));"
# R"type = 'NA';" # PPM
R"type = 'nn';"
# R"type = 'nnig';"
R"(lw_rho = rho_lpri(rho, type, alpha, X, v, m0, s20, sc_prec0, nu0)[[2]]);"
R"(lw_rho_post = rho_lpost(rho, lw_rho, y, sig2, mu0, sig20));"
R"sum(exp(lw_rho_post));"
R"round(tapply(exp(lw_rho), apply(rho, 1, max), FUN = sum), 5);"
R"round(tapply(exp(lw_rho_post), apply(rho, 1, max), FUN = sum), 5);"
@rget rho lw_rho_post

cohesion = Cohesion_CRP(log(alpha), 0, true)

similarity = Similarity_NN(sqrt(v), m0, sqrt(s20))
simtype = :NN

# similarity = Similarity_NNiChisq_indep(m0, sc_prec0, nu0, s20)
# simtype = :NNiChisq_indep

sampling_model = :Mean
sampling_model = :Reg # running this with betas all fixed at 0 should give same answer as :Mean

algo_c = :MH # Algo 7
algo_c = :FC # Aglo 8
M_newclust = 1

if sampling_model == :Mean
    G0 = Baseline_NormUnif(mu0, sig0, sigmax)
elseif sampling_model == :Reg
    G0 = Baseline_NormDLUnif(mu0, sig0, 0.1, sigmax)
end

model = Model_PPMx(y, X, 0, # C_init = 0 --> n clusters ; 1 --> 1 cluster
                         similarity_type=simtype, 
                         sampling_model=sampling_model,
                         init_lik_rand=false) # init_lik_rand = false sets all betas to 0
fieldnames(typeof(model))
fieldnames(typeof(model.state))
model.state.C
model.prior # use default

model.state.baseline = deepcopy(G0)
model.state.cohesion = deepcopy(cohesion)
model.state.similarity = deepcopy(similarity)

model.state.lik_params[1].sig

mcmc!(model, 10000,
    save = false,
    thin = 1,
    n_procs = 1,
    report_filename = "",
    report_freq = 50000,
    # update = [:C, :mu, :sig, :mu0, :sig0]
    update = [:C, :mu],
    upd_c_mtd = algo_c,
    M_newclust = M_newclust
)

sims = mcmc!(model, 100000,
    save = true,
    thin = 5,
    n_procs = 1,
    report_filename = "",
    report_freq = 50000,
    # update = [:C, :mu, :sig, :mu0, :sig0],
    # update = [:C, :mu, :sig, :beta, :mu0, :sig0],
    update = [:C, :mu],
    # monitor = [:C],
    monitor = [:C, :mu, :sig, :beta, :mu0, :sig0, :llik_mat],
    upd_c_mtd = algo_c,
    M_newclust = M_newclust
)

sims_K = [ maximum(sims[ii][:C]) for ii in eachindex(sims) ]
counts(sims_K) ./ length(sims_K) |> print

R"round(tapply(exp(lw_rho_post), apply(rho, 1, max), FUN = sum), 5);"
R"round(tapply(exp(lw_rho), apply(rho, 1, max), FUN = sum), 5);"

## individual partition probs
sims_Cord = [ relabel_c(sims[ii][:C]) for ii in eachindex(sims)]
sims_which_rho = [ findall([rho[i,:] == sims_Cord[ii] for i in 1:size(rho)[1]])[1] for ii in eachindex(sims) ]
sims_post_rho = counts(sims_which_rho) ./ length(sims)
w_post_rho = exp.(lw_rho_post)

hcat(hcat(sims_post_rho, w_post_rho))
sims_post_rho - w_post_rho
maximum(abs.(sims_post_rho - w_post_rho))

# sims_llik = [ sims[ii][:llik] for ii in eachindex(sims) ]
# sims_llik_mat = permutedims( hcat( [ sims[ii][:llik_mat] for ii in eachindex(sims) ]...) )
# Kmax = maximum(sims_K)
# sims_S = permutedims( hcat( [ counts(sims[ii][:C], Kmax) for ii in eachindex(sims) ]...) )
# sims_Sord = permutedims( hcat( [ sort(counts(sims[ii][:C], maximum(sims_K)), rev=true) for ii in eachindex(sims) ]...) )

# using Plots
# # Plots.PlotlyBackend() # don't use unless you have to--tends to crash

# Plots.plot(sims_llik_mat[:,1])
# Plots.plot(sims_llik)
# Plots.plot(sims_K)
# Plots.plot(sims_S)

R"library('ppmSuite')"
ppmSuite = R"gaussian_ppmx"
# @benchmark 
ppms = ppmSuite(y=y, X=X, meanModel=1, cohesion=1, M=alpha,
                      similarity_function=1,
                      consim=1, # 1 for NN, 2 for N-NIG
                      calibrate=0, 
                      simParms=[similarity.m0, similarity.sd0^2, similarity.sd^2, 1.0, 1.0, 1.0, 1.0],
                    #   simParms=[similarity.m0, similarity.s20, 1.0, similarity.sc_prec0, similarity.nu0, 1.0, 1.0],
                      modelPriors = [model.prior.baseline.mu0_mean, 
                      model.prior.baseline.mu0_sd^2, 
                      sigmax, 
                      model.prior.baseline.sig0_upper], 
                      draws=2500000, burn=2000000, thin=10,
                      verbose=false
)

nclust_ppms = [ ppms[:nclus][i] for i in 1:50000 ]
counts(nclust_ppms) ./ length(nclust_ppms) # from ppmSuite
counts(sims_K) ./ length(sims_K) # from ProductPartitionModels
R"round(tapply(exp(lw_rho_post), apply(rho, 1, max), FUN = sum), 5);"
R"round(tapply(exp(lw_rho), apply(rho, 1, max), FUN = sum), 5);"

# like_ppms = [ ppms[:like][i,j] for i in 1:1000, j in 1:5 ]
# llike_ppms = sum(log.(like_ppms), dims=2)[:,1]
# Plots.plot(llike_ppms)

# Plots.plot(nclust_ppmx)
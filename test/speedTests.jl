# testMCMC.jl

## go into package mode and enter: activate
## to run this code without modifying the ProductPartitionModels package

using Pkg
Pkg.activate()


using ProductPartitionModels
using StatsBase
using Random

using BenchmarkTools

using RCall


### loop through settings

prop_mis = 0.2

# n = 100
# p = 5

# datatype = :llinear
# datatype = :flat

# sampling_model = :Reg # running this with betas all fixed at 0 should give same answer as :Mean
# sampling_model = :Mean

report_file = open("benchmarks.txt", "a+")

for n in [100, 300]
  for p in [5, 10]
    for datatype in [:flat, :llinear]
      for sampling_model in [:Mean, :Reg]

        Random.seed!(230118)

        nmis = Int(floor(prop_mis*n*p))
        nobs = n*p - nmis
        
        X = Matrix{Union{Missing, Float64}}(missing, n, p)
        X[1:(n*p)] = randn(n*p)
        
        Ctrue = vcat(fill(1, Int(floor(.5n))), fill(2, Int(floor(.2*n))), fill(3, Int(floor(.2*n))), fill(4, Int(floor(.1*n))))
        length(Ctrue) == n || throw("C,n mismatch")
        
        for i in findall(Ctrue .== 2)
          X[i,1:2] += [3.0, -1.5]
        end
        for i in findall(Ctrue .== 3)
          X[i,1:2] += [0.0, 3.0]
        end
        for i in findall(Ctrue .== 4)
          X[i,1:2] += [-2.0, -2.0]
        end
        X
        
        beta = [ zeros(p) for k in 1:4 ]
        if datatype == :llinear
          beta[1][[1,2,5]] = [-1.0, 0.0, 1.0]
          beta[2][[1,5]] = [2.0, 1.0]
          beta[3][[2,5]] = [1.0, 1.0]
          beta[4][5] = 1.0
        end
        
        mu = [-1.0, 0.0, 1.0, 2.0]
        
        y = [ mu[Ctrue[i]] + X[i,:]'beta[Ctrue[i]] + 0.5*randn() for i in 1:n ]
        
        Xuse = deepcopy(X)
        mis_indx = sample(1:(n*p), nmis; replace=false)
        for ii in mis_indx
          Xuse[ii] = missing
        end
        
        model = Model_PPMx(y, Xuse, 0, 
                           similarity_type=:NNiChisq_indep, sampling_model=sampling_model, 
                           init_lik_rand=false) # C_init = 0 --> n clusters ; 1 --> 1 cluster
        
        model.state.similarity = Similarity_NNiChisq_indep(0.0, 0.1, 10.0, 0.7^2)
        model.state.baseline.sig_upper = 1.5
        
        if sampling_model == :Reg
          model.state.baseline.tau0 = 0.1
        end
        
        bb = @benchmarkable mcmc!($model, 1000,
            save=false,
            thin=1,
            n_procs=1,
            report_filename="",
            report_freq=100,
            update=[:C, :mu, :sig, :beta, :mu0, :sig0]
        )
        bbb = run(bb, samples=10, seconds=5000)
        # median(bbb) |> println
        # std(bbb) |> println
        # dump(bbb) |> println
        
        write(report_file, "n=$n, p=$p, datatype=$datatype, model=$sampling_model:\n")
        write(report_file, "median\n$(median(bbb))\n")
        write(report_file, "std\n$(std(bbb))\n\n\n")
        
        if sampling_model == :Mean
          R"library('ppmSuite')"
          ppmSuite = R"gaussian_ppmx"
          bppm = @benchmarkable ppmSuite(y=$y, X=$Xuse, meanModel=1, cohesion=1, M=1.0,
                              similarity_function=1,
                              consim=2, calibrate=0, 
                              simParms=[$model.state.similarity.m0, 
                                $model.state.similarity.s20, 
                                1.0, 
                                $model.state.similarity.sc_prec0, 
                                $model.state.similarity.nu0, 
                                1.0, 1.0],
                              modelPriors = [$model.prior.baseline.mu0_mean, 
                                $model.prior.baseline.mu0_sd^2, 
                                $model.state.baseline.sig_upper, 
                                $model.prior.baseline.sig0_upper], 
                              draws=1000, burn=0, thin=1,
                              verbose=false
          )

          bbppm = run(bppm, samples=10, seconds=5000)
          # median(bbppm) |> println
          # std(bbppm) |> println
          # dump(bbppm) |> println
        write(report_file, "n=$n, p=$p, datatype=$datatype, model=ppmSuite:\n")
        write(report_file, "median\n$(median(bbppm))\n")
        write(report_file, "std\n$(std(bbppm))\n\n\n")

        end
        
      end
    end
  end
end

close(report_file)


# using Plotly # run pkg> activate to be outside the package
# using RCall

# print(model.state.C)
# println(maximum(model.state.C))
# println(counts(model.state.C))

# symbols3d = ["circle", "cross", "diamond", "circle-open", "cross-open", "diamond-open"]
# colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

# x1range = collect(extrema(skipmissing(X[:,1]))) + [-1, 1]
# x2range = collect(extrema(skipmissing(X[:,2]))) + [-1, 1]
# yrange = collect(extrema(y)) + [-1, 1]


# C_color = deepcopy(Ctrue)
# C_shape = deepcopy(Ctrue)

# C_color = deepcopy(model.state.C)
# C_shape = deepcopy(model.state.C)


# trace1 = Plotly.scatter3d(Dict(
#   :x => convert(Vector{Float64}, X[:,1]),
#   :y => convert(Vector{Float64}, X[:, 2]),
#   :z => y[:],
#   :opacity => 0.7,
#   :showscale => false,
#   :mode => "markers",
#   :marker => Dict(:color => colors[C_color], :size => 5.0) #, :symbol => symbols3d[C_shape])
# ))
# Plotly.plot([trace1], Layout(height=700, width=700, scene_aspectratio=attr(x=1, y=1, z=1),
#             scene=attr(xaxis_title="x1", yaxis_title="x2", zaxis_title="y",
#             xaxis=attr(range=x1range), yaxis=attr(range=x2range), zaxis=attr(range=yrange)),
#             title="Complete cases"))


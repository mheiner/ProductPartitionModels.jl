module ProductPartitionModels

# using Base: Real, subtract_singletontype
# using Distributions: SpecialFunctions
using SpecialFunctions
using StatsBase
using Distributions
using Random, Random123
# using LinearAlgebra
using Distributed
using Dates

using StatsFuns: logsumexp

include("types.jl")
include("general.jl")

include("cohesion.jl")
include("similarity.jl")

include("likelihood.jl")

include("slice_sample.jl")
include("generalizedInverseGaussian.jl")
include("DirichletLaplace_shrinkage.jl")

include("update_lik_params.jl")
include("update_config.jl")
include("update_baseline.jl")

include("simulate_prior.jl")
include("mcmc.jl")
include("postPred.jl")

end # module

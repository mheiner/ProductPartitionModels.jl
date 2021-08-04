module ProductPartitionModels

# using Base: Real, subtract_singletontype
# using Distributions: SpecialFunctions
using SpecialFunctions
using StatsBase
using Distributions

include("types.jl")
include("general.jl")

include("cohesion.jl")
include("similarity.jl")

include("likelihood.jl")

include("slice_sample.jl")
include("generalizedInverseGaussian.jl")
include("DirichletLaplace_shrinkage.jl")

include("update_lik_params.jl")

include("simulate_prior.jl")

end # module

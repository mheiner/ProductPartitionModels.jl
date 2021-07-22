module ProductPartitionModels

using Base: Real
using Distributions: SpecialFunctions
using Distributions
using StatsBase
using SpecialFunctions

include("general.jl")

include("cohesion.jl")
include("similarity.jl")

include("generalizedInverseGaussian.jl")
include("DirichletLaplace_shrinkage.jl")

include("simulate_prior.jl")

end # module

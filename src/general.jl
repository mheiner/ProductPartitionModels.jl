# general.jl

export ObsXIndx;

struct ObsXIndx
    n_obs::Int
    n_mis::Int
    indx_obs::Vector{Int}
    indx_mis::Vector{Int}
end

function ObsXIndx(x::Union{Vector{Union{T, Missing}}, Vector{T}, Vector{Missing}} where T <: Real)
    p = length(x)

    indx_mis = findall(ismissing.(x))
    n_mis = length(indx_mis)

    indx_obs = setdiff(1:p, indx_mis)
    n_obs = length(indx_obs)

    n_mis + n_obs == p || throw("ObsXIndx indexing failed.")

    return ObsXIndx(n_obs, n_mis, indx_obs, indx_mis)
end



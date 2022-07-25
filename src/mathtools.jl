## mathtools.jl

# logsumexp is already imported from StatsFuns package

export logsumexp;

"""
logsumexp(x[, usemax])

Computes `log(sum(exp(x)))` in a stable manner.

### Example
```julia
x = rand(5)
logsumexp(x)
log(sum(exp.(x)))
```
"""
function logsumexp(x::Array{Float64}, usemax::Bool=true)
if usemax
m = maximum(x)
else
m = minimum(x)
end

return m + log(sum(exp.(x .- m)))
end

"""
logsumexp(x, region[, usemax])

Computes `log(sum(exp(x)))` in a stable manner along dimensions specified.

### Example
```julia
x = reshape(collect(1:24)*1.0, (2,3,4))
logsumexp(x, 2)
```
"""
function logsumexp(x::Array{Float64}, region, usemax::Bool=true)
if usemax
ms = maximum(x, dims=region)
else
ms = minimum(x, dims=region)
end
bc_xminusms = broadcast(-, x, ms)

expxx = exp.(bc_xminusms)
sumexpxx = sum(expxx, dims=region)

return log.(sumexpxx) .+ ms
end
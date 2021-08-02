# likelihood.jl

export Baseline_measure, Baseline_NormDLUnif, 
Hypers_shrinkReg, Hypers_DirLap, LikParams_PPMxReg, 
llik_k, llik_all;

abstract type Baseline_measure end

mutable struct Baseline_NormDLUnif <: Baseline_measure
    μ0::Real 
    σ0::Real

    τ0::Real # global shrinkage 

    upper_σ::Real

    Baseline_NormDLUnif(μ0, σ0, τ0, upper_σ) = new(μ0, σ0, τ0, upper_σ)
end

abstract type Hypers_shrinkReg end

mutable struct Hypers_DirLap{T <: Real} <: Hypers_shrinkReg
    phi::Vector{T}
    psi::Vector{T}
    tau::Real
end

mutable struct LikParams_PPMxReg{T <: Real}
    mu::Real
    sig::Real

    beta::Vector{T}
    beta_hypers::Hypers_shrinkReg

    # LikParams_PPMxReg(mu, sig, beta, beta_hypers) = sig <= 0.0 ? error("St. deviation must be positive.") : new(mu, sig, beta, beta_hypers)
end

function aux_moments_k(Xstats_k::Vector{Similarity_NiG_indep_stats}, similarity::Similarity_NiG_indep)

    p = length(Xstats_k)

    mean_out = Vector{Float64}(undef, p)
    sd_out = Vector{Float64}(undef, p)

    for j in 1:p
        if Xstats_k[j].n > 0
            xbar_now = Xstats_k[j].sumx / Xstats_k[j].n
            mean_out[j] = xbar_now # could do something else
            if Xstats_k[j].n > 1
                s2_now = (Xstats_k[j].sumx2 - Xstats_k[j].n * xbar_now^2) / (Xstats_k[j].n - 1.0)
                sd_out[j] = sqrt(s2_now) # could do something else
            else
                sd_out[j] = sqrt(similarity.b0 / (similarity.a0 + 1.0) / similarity.sc_div0) # could do something else
            end 
        else
            mean_out[j] = similarity.m0 # could do something else
            sd_out[j] = sqrt(similarity.b0 / (similarity.a0 + 1.0) / similarity.sc_div0) # could do something else
        end
    end

    return (mean_out, sd_out)

end

function llik_k(y_k::Vector{T} where T <: Real, X_k::Union{Matrix{T}, Matrix{Union{T, Missing}}} where T <: Real, 
    ObsXIndx_k::Vector{ObsXIndx}, lik_params_k::TT where TT <: LikParams_PPMxReg, Xstats_k::Vector{Similarity_NiG_indep_stats},
    similarity::Similarity_NiG_indep)
    
    aux_mean, aux_sd = aux_moments_k(Xstats_k, similarity) # each length-p vectors

    n_k = length(y_k)
    n_k > 0 || throw("Likelihood calculation must include at least one observation.")

    llik_out = 0.0

    for iii in 1:n_k
        
        mean_now = lik_params_k.mu
        
        xi = deepcopy(X_k[iii,:])
        if ObsXIndx_k[iii].n_obs > 0
            indx_xiobs = ObsXIndx_k[iii].indx_obs
            xiOc = xi[indx_xiobs] - aux_mean[indx_xiobs]
            mean_now += xiOc'lik_params_k.beta[indx_xiobs]
        end
        
        var_now = lik_params_k.sig^2

        if ObsXIndx_k[iii].n_mis > 0
            indx_ximis = ObsXIndx_k[iii].indx_mis
            var_now += sum( (aux_sd[indx_ximis] .* lik_params_k.beta[indx_ximis]).^2 )
        end

        llik_out += -0.5*log(2π) - 0.5*log(var_now) - 0.5*(y_k[iii] - mean_now)^2/var_now

    end

    return llik_out
end

function llik_all(y::Vector{T} where T <: Real, X::Union{Matrix{T}, Matrix{Union{T, Missing}}} where T <: Real,
    C::Vector{Int}, ObsXIndx::Vector{ObsXIndx}, 
    llik_params::Vector{TT} where TT <: LikParams_PPMxReg, Xstats::Vector{Vector{Similarity_NiG_indep_stats}},
    similarity::Similarity_NiG_indep)

    # n, p = size(X)
    K = maximum(C)
    # S = StatsBase.counts(C, K)

    llik_out = 0.0

    for k in 1:K
        indx_k = findall(C.==k)
        llik_out += llik_k(y[indx_k], X[indx_k,:], ObsXIndx[indx_k], llik_params[k], Xstats[k], similarity)
    end

    return llik_out
end


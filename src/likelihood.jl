# likelihood.jl

export llik_k, llik_all;

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

function llik_k(y_k::Vector{T}, X_k::Union{Matrix{T}, Matrix{Union{T, Missing}}}, 
    ObsXIndx_k::Vector{ObsXIndx}, lik_params_k::TT where TT <: LikParams_PPMxReg, Xstats_k::Vector{Similarity_NiG_indep_stats},
    similarity::Similarity_NiG_indep) where T <: Real
    
    aux_mean, aux_sd = aux_moments_k(Xstats_k, similarity) # each length-p vectors

    n_k = length(y_k)
    n_k > 0 || throw("Likelihood calculation must include at least one observation.")

    llik_out = 0.0

    means = Vector{T}(undef, n_k)
    vars = Vector{T}(undef, n_k)

    for iii in 1:n_k
        
        means[iii] = deepcopy(lik_params_k.mu)
        
        xi = deepcopy(X_k[iii,:])
        if ObsXIndx_k[iii].n_obs > 0
            indx_xiobs = ObsXIndx_k[iii].indx_obs
            xiOc = xi[indx_xiobs] - aux_mean[indx_xiobs]
            means[iii] += xiOc'lik_params_k.beta[indx_xiobs]
        end
        
        vars[iii] = lik_params_k.sig^2

        if ObsXIndx_k[iii].n_mis > 0
            indx_ximis = ObsXIndx_k[iii].indx_mis
            vars[iii] += sum( (aux_sd[indx_ximis] .* lik_params_k.beta[indx_ximis]).^2 )
        end

        llik_out += -0.5*log(2π) - 0.5*log(vars[iii]) - 0.5*(y_k[iii] - means[iii])^2/vars[iii]

    end

    return llik_out, means, vars
end
function llik_k(y_k::Vector{T}, means::Vector{T}, vars::Vector{T}, sig_old::T, sig_new::T) where T <: Real
    n_k = length(y_k)
    vars_out = deepcopy(vars)
    llik_out = 0.0
    for iii in 1:n_k
        vars_out[iii] -= sig_old^2
        vars_out[iii] += sig_new^2
        llik_out += -0.5*log(2π) - 0.5*log(vars_out[iii]) - 0.5*(y_k[iii] - means[iii])^2/vars_out[iii]
    end
    llik_out, means, vars_out
end

function llik_all(y::Vector{T}, X::Union{Matrix{T}, Matrix{Union{T, Missing}}},
    C::Vector{Int}, ObsXIndx::Vector{ObsXIndx}, 
    lik_params::Vector{TT} where TT <: LikParams_PPMxReg, Xstats::Vector{Vector{Similarity_NiG_indep_stats}},
    similarity::Similarity_NiG_indep) where T <: Real

    # n, p = size(X)
    K = maximum(C)
    # S = StatsBase.counts(C, K)

    llik_out = 0.0

    for k in 1:K
        indx_k = findall(C.==k)
        llik_out += llik_k(y[indx_k], X[indx_k,:], ObsXIndx[indx_k], lik_params[k], Xstats[k], similarity)[1]
    end

    return llik_out
end

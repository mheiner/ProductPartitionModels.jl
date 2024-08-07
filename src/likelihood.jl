# likelihood.jl

export llik_k, llik_all;

# function aux_moments_k(Xstats_k::Vector{Similarity_NNiG_indep_stats}, similarity::Similarity_NNiG_indep)

#   # raw moments, no prior unless empty

#     p = length(Xstats_k)

#     mean_out = Vector{Float64}(undef, p)
#     sd_out = Vector{Float64}(undef, p)

#     for j in 1:p
#         if Xstats_k[j].n > 0
#             xbar_now = Xstats_k[j].sumx / Xstats_k[j].n
#             mean_out[j] = xbar_now # could do something else
#             if Xstats_k[j].n > 1
#                 s2_now = (Xstats_k[j].sumx2 - Xstats_k[j].n * xbar_now^2) / (Xstats_k[j].n - 1.0)
#                 sd_out[j] = sqrt(s2_now) # could do something else
#             else
#                 sd_out[j] = sqrt(similarity.b0 / (similarity.a0 + 1.0) / similarity.sc_prec0) # could do something else
#             end
#         else
#             mean_out[j] = similarity.m0 # could do something else
#             sd_out[j] = sqrt(similarity.b0 / (similarity.a0 + 1.0) / similarity.sc_prec0) # could do something else
#         end
#     end

#     return (mean_out, sd_out)
# end

function aux_moments_k(Xstats_k::Vector{Similarity_NNiG_indep_stats}, similarity::Union{Similarity_NNiG_indep, Similarity_NNiChisq_indep})

    ## Unit-information N-IG priors

    p = length(Xstats_k)

    mean_out = Vector{Float64}(undef, p)
    sd_out = Vector{Float64}(undef, p)

    for j in 1:p
        n_now = float(Xstats_k[j].n)
        np1 = n_now + 1.0
        @inbounds mean_out[j] = (similarity.m0 + Xstats_k[j].sumx) / np1 # could do something else

        s0 = similarity.b0 / similarity.a0 # prior harmonic mean

        if n_now > 0.0
            xbar_now = Xstats_k[j].sumx / n_now
            ss_now = (Xstats_k[j].sumx2 - n_now * xbar_now^2)
            @inbounds sd_out[j] = sqrt( ( s0 + ss_now + n_now * (xbar_now - similarity.m0)^2 / np1 ) / np1 ) # could do something else
        else
            @inbounds sd_out[j] = sqrt(s0)
        end
    end

    return (mean_out, sd_out)
end

# function aux_moments_k(Xstats_k::Vector{Similarity_NNiG_indep_stats}, similarity::Similarity_NNiG_indep, phi::T=1.00, n0::T=1.00) where T <: Real

#     ## general N-IG priors

#     p = length(Xstats_k)

#     mean_out = Vector{Float64}(undef, p)
#     sd_out = Vector{Float64}(undef, p)

#     for j in 1:p
#         n_now = float(Xstats_k[j].n)

#         npphi = n_now + phi
#         npn0 = n_now + n0
#         s0 = similarity.b0 / similarity.a0 # prior harmonic mean

#         mean_out[j] = (phi * similarity.m0 + Xstats_k[j].sumx) / npphi # could do something else

#         # sd_out[j] = sqrt( (n0 * s0 + ss_now + n_now * phi * (xbar_now - similarity.m0)^2 / npphi ) / npn0 ) # could do something else

#         if n_now > 0.0 # this distinction is for some reason ?? crucial to success; without it, clusters glob together, ESPECIALLY obs with any missing covariates. phi and n0 don't seem to matter much if we have this clause ==> because I was dividing by zero...
#             xbar_now = Xstats_k[j].sumx / n_now
#             ss_now = (Xstats_k[j].sumx2 - n_now * xbar_now^2)
#             sd_out[j] = sqrt( (n0 * s0 + ss_now + n_now * phi * (xbar_now - similarity.m0)^2 / npphi ) / npn0 )
#             # sd_out[j] = sqrt( (n0 * s0 + ss_now ) / npn0 ) # could do something else
#         else
#             sd_out[j] = sqrt(s0)
#         end

#     end

#     return (mean_out, sd_out)
# end

function aux_moments_k(Xstats_k::Vector{Similarity_NN_stats}, similarity::Similarity_NN)

    ## Unit-information N-IG priors

    p = length(Xstats_k)

    mean_out = Vector{Float64}(undef, p)
    sd_out = Vector{Float64}(undef, p)

    for j in 1:p
        n_now = float(Xstats_k[j].n)
        np1 = n_now + 1.0
        mean_out[j] = (similarity.m0 + Xstats_k[j].sumx) / np1 # could do something else

        s0 = similarity.sd^2

        if n_now > 0.0
            xbar_now = Xstats_k[j].sumx / n_now
            ss_now = (Xstats_k[j].sumx2 - n_now * xbar_now^2)
            sd_out[j] = sqrt( ( s0 + ss_now + n_now * (xbar_now - similarity.m0)^2 / np1 ) / np1 ) # could do something else
        else
            sd_out[j] = sqrt(s0)
        end
    end

    return (mean_out, sd_out)
end



function aux_moments_empty(similarity::Union{Similarity_NNiG_indep, Similarity_NNiChisq_indep})
    return ( deepcopy(similarity.m0), sqrt(similarity.b0 / similarity.a0) )
end
function aux_moments_empty(similarity::Similarity_NN)
    return ( deepcopy(similarity.m0), deepcopy(similarity.sd) )
end


function llik_k(y_k::Vector{T}, X_k::Union{Matrix{T}, Matrix{Union{T, Missing}}, Matrix{Missing}},
    ObsXIndx_k::Vector{ObsXIndx},
    lik_params_k::TT where TT <: LikParams_PPMxReg,
    Xstats_k::Vector{TTT} where TTT <: Similarity_PPMxStats,
    similarity::TTTT where TTTT <: Similarity_PPMx) where T <: Real

    aux_mean, aux_sd = aux_moments_k(Xstats_k, similarity) # each length-p vectors

    n_k = length(y_k)
    n_k > 0 || throw("Likelihood calculation must include at least one observation.")

    llik_out = Vector{T}(undef, n_k)
    means = Vector{T}(undef, n_k)
    vars = Vector{T}(undef, n_k)

    for iii in 1:n_k

        means[iii] = deepcopy(lik_params_k.mu)

        if ObsXIndx_k[iii].n_obs > 0
            indx_xiobs = ObsXIndx_k[iii].indx_obs
            @inbounds ziO = (X_k[iii, indx_xiobs] - aux_mean[indx_xiobs]) ./ aux_sd[indx_xiobs]
            means[iii] += ziO'lik_params_k.beta[indx_xiobs]
        end

        @inbounds vars[iii] = lik_params_k.sig^2

        if ObsXIndx_k[iii].n_mis > 0
            indx_ximis = ObsXIndx_k[iii].indx_mis
            @inbounds vars[iii] += sum( lik_params_k.beta[indx_ximis].^2 )
        end

        @inbounds llik_out[iii] = -0.5*log(2π) - 0.5*log(vars[iii]) - 0.5*(y_k[iii] - means[iii])^2/vars[iii]

    end

    return Dict(:llik => sum(llik_out), :means => means, :vars => vars, :llik_vec => llik_out)
end
function llik_k(y_k::Vector{T}, means::Vector{T}, vars::Vector{T}, sig_old::T, sig_new::T) where T <: Real
    n_k = length(y_k)
    vars_out = deepcopy(vars)
    llik_out = Vector{T}(undef, n_k)
    for iii in 1:n_k
        vars_out[iii] -= sig_old^2
        vars_out[iii] += sig_new^2
        @inbounds llik_out[iii] = -0.5*log(2π) - 0.5*log(vars_out[iii]) - 0.5*(y_k[iii] - means[iii])^2/vars_out[iii]
    end
    Dict(:llik => sum(llik_out), :means => means, :vars => vars_out, :llik_vec => llik_out)
end
function llik_k(y_k::Vector{T}, lik_params_k::TT where TT <: LikParams_PPMxMean) where T <: Real

    n_k = length(y_k)
    n_k > 0 || throw("Likelihood calculation must include at least one observation.")

    v = lik_params_k.sig^2
    halflogv = 0.5*log(v)
    halflog2pi = 0.5*log(2π)

    llik_out = fill( -halflog2pi - halflogv, n_k )

    for iii in 1:n_k
        @inbounds llik_out[iii] -= 0.5*(y_k[iii] - lik_params_k.mu)^2 / v
    end

    return Dict(:llik => sum(llik_out), :llik_vec => llik_out)
end

function llik_all(y::Vector{T}, X::Union{Matrix{T}, Matrix{Union{T, Missing}}, Matrix{Missing}},
    C::Vector{Int}, ObsXIndx::Vector{ObsXIndx},
    lik_params::Vector{TT} where TT <: LikParams_PPMxReg,
    Xstats::Vector{Vector{TTT}} where TTT <: Similarity_PPMxStats,
    similarity::TTTT where TTTT <: Similarity_PPMx) where T <: Real

    K = maximum(C)
    n = length(y)

    llik_out = Vector{T}(undef, n)

    for k in 1:K
        indx_k = findall(C.==k)
        @inbounds llik_out[indx_k] = llik_k(y[indx_k], X[indx_k,:], ObsXIndx[indx_k], lik_params[k], Xstats[k], similarity)[:llik_vec]
    end

    return Dict(:llik => sum(llik_out), :llik_vec => llik_out)
end
function llik_all(y::Vector{T}, C::Vector{Int}, 
    lik_params::Vector{TT} where TT <: LikParams_PPMxMean) where T <: Real

    K = maximum(C)
    n = length(y)

    llik_out = Vector{T}(undef, n)

    for k in 1:K
        indx_k = findall(C.==k)
        @inbounds llik_out[indx_k] = llik_k(y[indx_k], lik_params[k])[:llik_vec]
    end

    return Dict(:llik => sum(llik_out), :llik_vec => llik_out)
end
function llik_all(model::Model_PPMx)

    if typeof( model.state.lik_params[1] ) <: LikParams_PPMxReg
        DD = llik_all(model.y, model.X, model.state.C, model.obsXIndx,
                      model.state.lik_params, model.state.Xstats, model.state.similarity)
    elseif typeof( model.state.lik_params[1] ) <: LikParams_PPMxMean
        DD = llik_all(model.y, model.state.C, model.state.lik_params)
    end

    return DD
end

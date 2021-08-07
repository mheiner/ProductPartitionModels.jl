# mcmc.jl

export timemod!, etr, mcmc!;

## from Arthur Lui
function deepcopyFields(state::T, fields::Vector{Symbol}) where T
  substate = Dict{Symbol, Any}()

  for field in fields
    substate[field] = deepcopy(getfield(state, field))
  end

  return substate
end

function postSimsInit(n_keep::Int, init_state::Union{State_PPMx},
    monitor::Vector{Symbol}=[:C, :mu, :sig, :beta])

    monitor_outer = intersect(monitor, fieldnames(typeof(init_state)))
    monitor_lik = intersect(monitor, fieldnames(typeof(init_state.lik_params[1])))

    state = deepcopyFields(init_state, monitor_outer)

    state[:lik_params] = [ deepcopyFields(init_state.lik_params[1], monitor_lik) ]

    state[:llik] = 0.0

    sims = [ deepcopy(state) for i = 1:n_keep ]

    return sims
end

## MCMC timing for benchmarks
function timemod!(n::Int64, model::Union{Model_PPMx}, niter::Int, outfilename::String; n_procs=1, save=false)
    outfile = open(outfilename, "a+")
    write(outfile, "\ntiming for $(niter) iterations each with $(n_procs) parallel processes:\n")
    for i in 1:n
        tinfo = @timed mcmc!(model, niter, n_procs=n_procs, save=save)
        write(outfile, "trial $(i), elapsed: $(tinfo[2]) seconds, allocation: $(tinfo[3]/1.0e6) Megabytes\n")
    end
    close(outfile)
end

## estimate time remaining
function etr(timestart::DateTime, n_keep::Int, thin::Int, outfilename::String)
    timeendburn = now()
    durperiter = (timeendburn - timestart).value / 1.0e5 # in milliseconds
    milsecremaining = durperiter * (n_keep * thin)
    estimatedfinish = now() + Dates.Millisecond(Int64(round(milsecremaining)))
    report_file = open(outfilename, "a+")
    write(report_file, "Completed burn-in at $(durperiter/1.0e3*1000.0) seconds per 1000 iterations \n
      $(durperiter/1.0e3/60.0*1000.0) minutes per 1000 iterations \n
      $(durperiter/1.0e3/60.0/60.0*1000.0) hours per 1000 iterations \n
      estimated completion time $(estimatedfinish)")
    close(report_file)
end
function etr(timestart::DateTime; n_iter_timed::Int, n_keep::Int, thin::Int, outfilename::String)
    timeendburn = now()
    durperiter = (timeendburn - timestart).value / float(n_iter_timed) # in milliseconds
    milsecremaining = durperiter * (n_keep * thin)
    estimatedfinish = now() + Dates.Millisecond(Int64(round(milsecremaining)))
    report_file = open(outfilename, "a+")
    write(report_file, "Completed burn-in at $(durperiter/1.0e3*1000.0) seconds per 1000 iterations \n
      $(durperiter/1.0e3/60.0*1000.0) minutes per 1000 iterations \n
      $(durperiter/1.0e3/60.0/60.0*1000.0) hours per 1000 iterations \n
      estimated completion time $(estimatedfinish)")
    close(report_file)
end


function mcmc!(model::Model_PPMx, n_keep::Int;
    save::Bool=true,
    thin::Int=1,
    n_procs::Int=1,
    report_filename::String="out_progress.txt",
    report_freq::Int=10000,
    update::Vector{Symbol}=[:C, :lik_params],
    monitor::Vector{Symbol}=[:C, :mu, :sig, :beta]
    )

    ## output files
    report_file = open(report_filename, "a+")
    write(report_file, "Commencing MCMC at $(Dates.now()) for $(n_keep * thin) iterations.\n")

    ## split update parameters
    update_outer = intersect(update, fieldnames(typeof(model.state)))
    # update_lik = intersect(update, setdiff( fieldnames(typeof(model.state.lik_params[1])), [:beta_hypers]))
    update_lik = intersect(update, [:lik_params])
    up_lik = length(update_lik) > 0
    update_baseline = intersect(update, fieldnames(typeof(model.state.baseline)))
    up_baseline = length(update_baseline) > 0
    update_cohesion = intersect(update, fieldnames(typeof(model.state.cohesion)))
    up_cohesion = length(update_cohesion) > 0
    update_similarity = intersect(update, fieldnames(typeof(model.state.similarity)))
    up_similarity = length(update_similarity) > 0

    ## collect posterior samples
    if save
        sims = postSimsInit(n_keep, model.state, monitor)
        monitor_outer = intersect(monitor, fieldnames(typeof(model.state)))
        monitor_lik = intersect(monitor, fieldnames(typeof(model.state.lik_params[1])))
    end

    ## sampling
    for i in 1:n_keep
        for j in 1:thin

            if (:C in update_outer)
                update_C!(model) # refreshes model state except llik
            end

            if up_lik
                update_lik_params!(model)
                # update_lik_params!(model.state, model.prior, model.y, update_mixcomps, n_procs=n_procs) # if we want to go parallel at some point
            end

            model.state.iter += 1
            if model.state.iter % report_freq == 0
                write(report_file, "Iter $(model.state.iter) at $(Dates.now())\n")
                model.state.llik = llik_all(model.y, model.X, model.state.C, model.obsXIndx, 
                    model.state.lik_params, model.state.Xstats, model.state.similarity)
                write(report_file, "Log-likelihood $(model.state.llik)\n")
            end

        end

        if save
            for field in monitor_outer
                sims[i][field] = deepcopy(getfield(model.state, field))
            end
            if length(monitor_lik) > 0
                sims[i][:lik_params] = [ deepcopyFields(model.state.lik_params[k], monitor_lik) for k in 1:length(model.state.lik_params) ]
            end
            sims[i][:llik] = llik_all(model.y, model.X, model.state.C, model.obsXIndx, 
                model.state.lik_params, model.state.Xstats, model.state.similarity)
        end

    end

    model.state.llik = llik_all(model.y, model.X, model.state.C, model.obsXIndx, 
                    model.state.lik_params, model.state.Xstats, model.state.similarity)

    close(report_file)

    if save
        return sims
    else
        return model.state.iter
    end

end




fit_PPMx = function(y, X, Xpred, nburn, nkeep, nthin, pred_insamp=FALSE, 
                    progressfile="", report_freq=5000,
                    sampling_model="Reg",
                    cohesion, similarity, baseline, baseline_prior, 
                    upd_beta=TRUE, # irrelevant if sampling_model = "Mean"
                    y_grid=NULL, crossxy=TRUE) {
  
  julia_assign("y", y)
  julia_command("y = float(y)", show_value=FALSE)
  
  julia_assign("X", X)
  julia_command("X = Matrix(float(X))", show_value=FALSE)
  
  julia_assign("nburn", nburn)
  julia_command("nburn = Int(nburn)", show_value=FALSE)
  julia_assign("nkeep", nkeep)
  julia_command("nkeep = Int(nkeep)", show_value=FALSE)
  julia_assign("nthin", nthin)
  julia_command("nthin = Int(nthin)", show_value=FALSE)
  julia_assign("progressfile", progressfile)
  julia_assign("report_freq", report_freq)
  
  julia_assign("logalpha", log(cohesion$alpha))
  julia_command("cohesion = Cohesion_CRP(logalpha, 0, true)", show_value=FALSE)  
  
  if (similarity$type == "NNiG_indep") {
    julia_command("simil_type = :NNiG_indep", show_value=FALSE)
    julia_assign("simil_m0", similarity$m0)
    julia_assign("simil_sc_prec0", similarity$sc_prec0)
    julia_assign("simil_a0", similarity$a0)
    julia_assign("simil_b0", similarity$b0)
    julia_command("similarity = Similarity_NNiG_indep(simil_m0, simil_sc_prec0, simil_a0, simil_b0)", show_value=FALSE)
  } else if (similarity$type == "NNiChisq_indep") {
    julia_command("simil_type = :NNiChisq_indep", show_value=FALSE)
    julia_assign("simil_m0", similarity$m0)
    julia_assign("simil_sc_prec0", similarity$sc_prec0)
    julia_assign("simil_nu0", similarity$nu0)
    julia_assign("simil_s20", similarity$s20)
    julia_command("similarity = Similarity_NNiChisq_indep(simil_m0, simil_sc_prec0, simil_nu0, simil_s20)", show_value=FALSE)
  } else if (similarity$type == "NN") {
    julia_command("simil_type = :NN", show_value=FALSE)
    julia_assign("simil_sd", similarity$sd)
    julia_assign("simil_m0", similarity$m0)
    julia_assign("simil_sd0", similarity$sd0)
    julia_command("similarity = Similarity_NN(simil_sd, simil_m0, simil_sd0)", show_value=FALSE)
  }
  
  if (sampling_model == "Reg") {
    julia_command("sampling_model = :Reg", show_value=FALSE)
    
    julia_assign("base_mu0", baseline$mu0)
    julia_assign("base_sig0", baseline$sig0)
    julia_assign("base_tau0", baseline$tau0)
    julia_assign("base_sig_upper", baseline$sig_upper)
    julia_command("G0 = Baseline_NormDLUnif(base_mu0, base_sig0, base_tau0, base_sig_upper)", show_value=FALSE)
    
    julia_assign("basePri_mu0_mean", baseline_prior$mu0_mean)
    julia_assign("basePri_mu0_sd", baseline_prior$mu0_sd)
    julia_assign("basePri_sig0_upper", baseline_prior$sig0_upper)
    julia_command("basePri = Prior_baseline_NormDLUnif(basePri_mu0_mean, basePri_mu0_sd, basePri_sig0_upper)", show_value=FALSE)
  } else if (sampling_model == "Mean") {
    julia_command("sampling_model = :Mean", show_value=FALSE)
    
    julia_assign("base_mu0", baseline$mu0)
    julia_assign("base_sig0", baseline$sig0)
    julia_assign("base_sig_upper", baseline$sig_upper)
    julia_command("G0 = Baseline_NormUnif(base_mu0, base_sig0, base_sig_upper)", show_value=FALSE)
    
    julia_assign("basePri_mu0_mean", baseline_prior$mu0_mean)
    julia_assign("basePri_mu0_sd", baseline_prior$mu0_sd)
    julia_assign("basePri_sig0_upper", baseline_prior$sig0_upper)
    julia_command("basePri = Prior_baseline_NormUnif(basePri_mu0_mean, basePri_mu0_sd, basePri_sig0_upper)", show_value=FALSE)
  }
  
  if (upd_beta) { # :beta in the update vector will be ignored if sampling_model == :Mean
    julia_command("mod = Model_PPMx(y, X, 0, similarity_type=simil_type, sampling_model=sampling_model, init_lik_rand=true)", show_value=FALSE) # C_init = 0 --> n clusters ; 1 --> 1 cluster
    julia_command("upd_params = [:C, :mu, :sig, :beta, :mu0, :sig0]", show_value=FALSE)
  } else {
    julia_command("mod = Model_PPMx(y, X, 0, similarity_type=simil_type, sampling_model=sampling_model, init_lik_rand=false)", show_value=FALSE) # C_init = 0 --> n clusters ; 1 --> 1 cluster
    julia_command("upd_params = [:C, :mu, :sig, :mu0, :sig0]", show_value=FALSE)
  }
  
  julia_command("for i in 1:length(mod.state.lik_params) mod.state.lik_params[i].sig = 0.1 end", show_value=FALSE) # temporary hack
  julia_command("mod.state.cohesion = deepcopy(cohesion)", show_value=FALSE)
  julia_command("mod.state.similarity = deepcopy(similarity)", show_value=FALSE)
  julia_command("mod.state.baseline = deepcopy(G0)", show_value=FALSE)
  julia_command("mod.prior.baseline = deepcopy(basePri)", show_value=FALSE)
  
  julia_command('timestart = Dates.now()')
  
  julia_command('mcmc!(mod, nburn,
        save=false,
        thin=1,
        n_procs=1,
        report_filename=progressfile,
        report_freq=Int(report_freq),
        update=upd_params
  )', show_value=FALSE)
  
  julia_command('etr(timestart; n_iter_timed=nburn, n_keep=nkeep, thin=nthin, outfilename=progressfile)', show_value=FALSE)
  
  julia_command('sims = mcmc!(mod, nkeep,
               save=true,
               thin=nthin,
               n_procs=1,
               report_filename=progressfile,
               report_freq=Int(report_freq),
               update=upd_params,
               monitor=[:C, :mu, :sig, :beta, :mu0, :sig0, :llik_mat]
  )', show_value=FALSE)
  
  sim_llik = julia_eval("[ sims[ii][:llik] for ii in 1:nkeep ]")
  sim_llik_mat = julia_eval("permutedims( hcat( [ sims[ii][:llik_mat] for ii in 1:nkeep ]...) )")
  sim_nclus = julia_eval("[ maximum(sims[ii][:C]) for ii in 1:nkeep ]")
  sim_Si = julia_eval("permutedims( hcat( [ sims[ii][:C] for ii in 1:nkeep ]...) )")
  
  sim_mu = julia_eval("[ sims[ii][:lik_params][sims[ii][:C][i]][:mu] for ii in 1:nkeep, i in 1:mod.n ]")
  sim_sig = julia_eval("[ sims[ii][:lik_params][sims[ii][:C][i]][:sig] for ii in 1:nkeep, i in 1:mod.n ]")

  sim_mu0 = julia_eval("[ sims[ii][:baseline][:mu0] for ii in 1:nkeep ]")
  sim_sig0 = julia_eval("[ sims[ii][:baseline][:sig0] for ii in 1:nkeep ]")
  
  if (sampling_model == "Reg") {
    sim_beta = julia_eval("[ sims[ii][:lik_params][sims[ii][:C][i]][:beta][kk] for ii in 1:nkeep, i in 1:mod.n, kk in 1:mod.p ]")
    out = list(llik=sim_llik, llik_mat=sim_llik_mat, nclus=sim_nclus, Si=sim_Si, mu=sim_mu, sig=sim_sig, beta=sim_beta, mu0=sim_mu0, sig0=sim_sig0)
  } else if (sampling_model == "Mean") {
    out = list(llik=sim_llik, llik_mat=sim_llik_mat, nclus=sim_nclus, Si=sim_Si, mu=sim_mu, sig=sim_sig, mu0=sim_mu0, sig0=sim_sig0)    
  }
  
  
  if (!is.null(Xpred)) {
    out$Pred = list()
    if (is.list(Xpred)) {
      nlist = length(Xpred)
      for ( l in 1:nlist ) {
        julia_assign("Xpred", Xpred[[l]])
        julia_command("Xpred = Matrix(float(Xpred))", show_value=FALSE)
        julia_command("Ypred, Cpred, Mean_pred = postPred(Xpred, mod, sims, upd_params)", show_value=FALSE)
        
        out$Pred[[l]] = list()
        out$Pred[[l]]$Xpred = Xpred[[l]]
        out$Pred[[l]]$crossxy = crossxy[[l]]
        out$Pred[[l]]$Ypred = julia_eval("Ypred")
        out$Pred[[l]]$Cpred = julia_eval("Cpred")
        out$Pred[[l]]$Mpred = julia_eval("Mean_pred")
        
        if (!is.null(y_grid)) {
          stopifnot(is.list(y_grid))
          stopifnot(is.list(crossxy))
          stopifnot(length(y_grid) == nlist)
          stopifnot(length(crossxy) == nlist)
          
          julia_assign("y_grid", y_grid[[l]])
          julia_assign("crossxy", crossxy[[l]])
          julia_command("y_grid = float(y_grid)", show_value=FALSE)
          julia_command("ppld = postPredLogdens(Xpred, y_grid, mod, sims, update_params=upd_params, crossxy=crossxy)", show_value=FALSE)
          out$Pred[[l]]$PPlogDens = julia_eval("ppld")
          out$Pred[[l]]$y_grid = y_grid[[l]]
        }
        
      }
    } else { # i.e., if Xpred is not a list...
      julia_assign("Xpred", Xpred)
      julia_command("Xpred = Matrix(float(Xpred))", show_value=FALSE)
      julia_command("Ypred, Cpred, Mean_pred = postPred(Xpred, mod, sims, upd_params)", show_value=FALSE)

      out$Pred[[1]] = list()
      out$Pred[[1]]$Xpred = Xpred
      out$Pred[[1]]$crossxy = crossxy
      out$Pred[[1]]$Ypred = julia_eval("Ypred")
      out$Pred[[1]]$Cpred = julia_eval("Cpred")
      out$Pred[[1]]$Mpred = julia_eval("Mean_pred")
      
      if (!is.null(y_grid)) {
        julia_assign("y_grid", y_grid)
        julia_assign("crossxy", crossxy)
        julia_command("y_grid = float(y_grid)", show_value=FALSE)
        julia_command("ppld = postPredLogdens(Xpred, y_grid, mod, sims, update_params=upd_params, crossxy=crossxy)", show_value=FALSE)
        out$Pred[[1]]$PPlogDens = julia_eval("ppld")
        out$Pred[[1]]$y_grid = y_grid
      }
      
    }
  }

  if (pred_insamp) {
    julia_command("Ypred_insamp, Mpred_insamp = postPred(mod, sims)", show_value=FALSE)
    out$Ypred_insamp = julia_eval("Ypred_insamp")
    out$Mpred_insamp = julia_eval("Mpred_insamp")
    
    julia_command("ppld_is = postPredLogdens(X, y, mod, sims, update_params=upd_params, crossxy=false)", show_value=FALSE)
    out$logDens_insamp = julia_eval("ppld_is")
  }
  
  out$y = y
  out$X = X
  
  return(out)
}

pred_PPMx = function(Xpred, upd_params=c("mu", "sig", "beta", "mu0", "sig0")) {
  julia_assign("Xpred1", Xpred)
  julia_command("Xpred1 = Matrix(float(Xpred1))", show_value=FALSE)

  julia_command("upd_params = Vector{Symbol}(undef, 0)", show_value=FALSE)
  if ("mu" %in% upd_params) {
    julia_command("push!(upd_params, :mu)", show_value=FALSE)
  }
  if ("sig" %in% upd_params) {
    julia_command("push!(upd_params, :sig)", show_value=FALSE)
  }
  if ("beta" %in% upd_params) {
    julia_command("push!(upd_params, :beta)", show_value=FALSE)
  }
  if ("mu0" %in% upd_params) {
    julia_command("push!(upd_params, :mu0)", show_value=FALSE)
  }
  if ("sig0" %in% upd_params) {
    julia_command("push!(upd_params, :sig0)", show_value=FALSE)
  }
  
  julia_command("Ypred1, Cpred1, Mpred1 = postPred(Xpred1, mod, sims, upd_params)", show_value=FALSE)
  
  out = list()
  out$Ypred = julia_eval("Ypred1")
  out$Cpred = julia_eval("Cpred1")
  out$Mpred = julia_eval("Mpred1")
  
  out
}

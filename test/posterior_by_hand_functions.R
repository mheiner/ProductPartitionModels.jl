lg_nn = function(x, v, m0, s20, theta = 0.0) {
  # theta dummy, doesn't matter

  out = 0
  
  x_use = na.omit(x)
  n = length(x_use)

  if (n > 0) {
    s21 = 1.0 / (1.0 / s20 + n / v)
    m1 = s21 * (m0 / s20 + sum(x_use) / v)
    
    llik = sum( dnorm(x_use, theta, sqrt(v), log=TRUE) )
    lpri = dnorm(theta, m0, sqrt(s20), log=TRUE)
    lpost = dnorm(theta, m1, sqrt(s21), log=TRUE)
    
    out = llik + lpri - lpost
  }
  out
}


lg_nnig = function(x, m0, sc_prec0, nu0, s20, theta = c(0.0, 1.0)) {
  # theta dummy, doesn't matter
  
  out = 0
  
  x_use = na.omit(x)
  n = length(x_use)

  if (n > 0) {
    xbar = mean(x_use)
    ss = sum((x_use - xbar)^2)
    
    a0 = 0.5*nu0
    b0 = 0.5*nu0*s20
    
    sc_prec1 = sc_prec0 + n
    m1 = (sc_prec0*m0 + n*xbar) / sc_prec1
    a1 = 0.5 * (nu0 + n)
    b1 = 0.5 * (nu0*s20 + ss + n*sc_prec0*(xbar - m0)^2 / sc_prec1)
    
    llik = sum( dnorm(x_use, theta[1], sqrt(theta[2]), log=TRUE) )
    lpri = dnorm(theta[1], m0, sqrt(theta[2]/sc_prec0), log=TRUE) + dgamma(1.0/theta[2], shape=a0, rate=b0, log=TRUE) - 2*log(theta[2])
    lpost = dnorm(theta[1], m1, sqrt(theta[2]/sc_prec1), log=TRUE) + dgamma(1.0/theta[2], shape=a1, rate=b1, log=TRUE) - 2*log(theta[2])
    
    out = llik + lpri - lpost    
  }  
  out
}


rho_lpri = function(rho, type="nn", alpha, X, v, m0, s20, sc_prec0, nu0) {
  
  p = ncol(X)
  
  B = nrow(rho)
  lw = numeric(B)
  lalph = log(alpha)
  
  for (bb in 1:B) {
    
    J = max(rho[bb,])
    
    lA = J * lalph
    lB = sum( lgamma( table(rho[bb,]) ) )
    
    lC = 0
    for (j in 1:J) {
      for (pp in 1:p) {
        if (type == "nn") {
          lC = lC + lg_nn(X[which(rho[bb,] == j), pp], v, m0, s20)
        } else if (type == "nnig") {
          lC = lC + lg_nnig(X[which(rho[bb,] == j), pp], m0, sc_prec0, nu0, s20)
        }
      }
    }
    
    lw[bb] = lA + lB + lC
    
  }
  
  list(lw, lw - logSumExp(lw))
}


rho_lpost = function(rho, lw_pri, y, sig2, mu0, sig20, theta=0.0) {
  B = length(lw_pri)
  lw_out = lw_pri
  
  for (bb in 1:B) {
    J = max(rho[bb,])
    for (j in 1:J) {
      lw_out[bb] = lw_out[bb] + lg_nn(y[which(rho[bb,] == j)], sig2, mu0, sig20, theta) # this lg_nn refers to likelihood
    }
  }
  
  lw_out - logSumExp(lw_out)
}

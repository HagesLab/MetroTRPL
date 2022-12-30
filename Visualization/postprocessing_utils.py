# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:53:02 2022

@author: cfai2
"""
import numpy as np
import os
from secondary_parameters import mu_eff, LI_tau_eff, HI_tau_srh, LI_tau_srh
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
importr('coda')
as_mcmc_list = robjects.r["mcmc.list"]
as_mcmc = robjects.r["as.mcmc"]
calc_ESS = robjects.r['effectiveSize']
"""
function (x) 
{
    if (is.mcmc.list(x)) {
        ess <- do.call("rbind", lapply(x, effectiveSize))
        ans <- apply(ess, 2, sum)
    }
    else {
        x <- as.mcmc(x)
        x <- as.matrix(x)
        spec <- spectrum0.ar(x)$spec
        ans <- ifelse(spec == 0, 0, nrow(x) * apply(x, 2, var)/spec)
    }
    return(ans)
}
"""
gl = robjects.r['gelman.diag']
"""
function (x, confidence = 0.95, transform = FALSE, autoburnin = TRUE, 
    multivariate = TRUE) 
{
    x <- as.mcmc.list(x)
    if (nchain(x) < 2) 
        stop("You need at least two chains")
    if (autoburnin && start(x) < end(x)/2) 
        x <- window(x, start = end(x)/2 + 1)
    Niter <- niter(x)
    Nchain <- nchain(x)
    Nvar <- nvar(x)
    xnames <- varnames(x)
    if (transform) 
        x <- gelman.transform(x)
    x <- lapply(x, as.matrix)
    S2 <- array(sapply(x, var, simplify = TRUE), dim = c(Nvar, 
        Nvar, Nchain))
    W <- apply(S2, c(1, 2), mean)
    xbar <- matrix(sapply(x, apply, 2, mean, simplify = TRUE), 
        nrow = Nvar, ncol = Nchain)
    B <- Niter * var(t(xbar))
    if (Nvar > 1 && multivariate) {
        if (is.R()) {
            CW <- chol(W)
            emax <- eigen(backsolve(CW, t(backsolve(CW, B, transpose = TRUE)), 
                transpose = TRUE), symmetric = TRUE, only.values = TRUE)$values[1]
        }
        else {
            emax <- eigen(qr.solve(W, B), symmetric = FALSE, 
                only.values = TRUE)$values
        }
        mpsrf <- sqrt((1 - 1/Niter) + (1 + 1/Nvar) * emax/Niter)
    }
    else mpsrf <- NULL
    w <- diag(W)
    b <- diag(B)
    s2 <- matrix(apply(S2, 3, diag), nrow = Nvar, ncol = Nchain)
    muhat <- apply(xbar, 1, mean)
    var.w <- apply(s2, 1, var)/Nchain
    var.b <- (2 * b^2)/(Nchain - 1)
    cov.wb <- (Niter/Nchain) * diag(var(t(s2), t(xbar^2)) - 2 * 
        muhat * var(t(s2), t(xbar)))
    V <- (Niter - 1) * w/Niter + (1 + 1/Nchain) * b/Niter
    var.V <- ((Niter - 1)^2 * var.w + (1 + 1/Nchain)^2 * var.b + 
        2 * (Niter - 1) * (1 + 1/Nchain) * cov.wb)/Niter^2
    df.V <- (2 * V^2)/var.V
    df.adj <- (df.V + 3)/(df.V + 1)
    B.df <- Nchain - 1
    W.df <- (2 * w^2)/var.w
    R2.fixed <- (Niter - 1)/Niter
    R2.random <- (1 + 1/Nchain) * (1/Niter) * (b/w)
    R2.estimate <- R2.fixed + R2.random
    R2.upper <- R2.fixed + qf((1 + confidence)/2, B.df, W.df) * 
        R2.random
    psrf <- cbind(sqrt(df.adj * R2.estimate), sqrt(df.adj * R2.upper))
    dimnames(psrf) <- list(xnames, c("Point est.", "Upper C.I."))
    out <- list(psrf = psrf, mpsrf = mpsrf)
    class(out) <- "gelman.diag"
    out
}
"""
gw = robjects.r['geweke.diag']
"""
geweke.diag = function (x, frac1 = 0.1, frac2 = 0.5) 
{
    if (frac1 < 0 || frac1 > 1) {
        stop("frac1 invalid")
    }
    if (frac2 < 0 || frac2 > 1) {
        stop("frac2 invalid")
    }
    if (frac1 + frac2 > 1) {
        stop("start and end sequences are overlapping")
    }
    if (is.mcmc.list(x)) {
        return(lapply(x, geweke.diag, frac1, frac2))
    }
    x <- as.mcmc(x)
    xstart <- c(start(x), floor(end(x) - frac2 * (end(x) - start(x))))
    xend <- c(ceiling(start(x) + frac1 * (end(x) - start(x))), 
        end(x))
    y.variance <- y.mean <- vector("list", 2)
    for (i in 1:2) {
        y <- window(x, start = xstart[i], end = xend[i])
        y.mean[[i]] <- apply(as.matrix(y), 2, mean)
        y.variance[[i]] <- spectrum0.ar(y)$spec/niter(y)
    }
    z <- (y.mean[[1]] - y.mean[[2]])/sqrt(y.variance[[1]] + y.variance[[2]])
    out <- list(z = z, frac = c(frac1, frac2))
    class(out) <- "geweke.diag"
    return(out)
}
"""
def recommend_logscale(which_param, do_log):
    if which_param == "Sf+Sb":
        recommend = (do_log["Sf"] or do_log["Sb"])
        
    elif which_param == "tauN+tauP":
        recommend = (do_log["tauN"] or do_log["tauP"])
        
    elif which_param == "Cn+Cp":
        recommend = (do_log["Cn"] or do_log["Cp"])
        
    elif which_param == "tau_eff" or which_param == "tau_srh":
        recommend = (do_log["tauN"] or (do_log["Sf"] or do_log["Sb"]))
        
    elif which_param == "HI_tau_srh":
        recommend = (do_log["tauN"] or do_log["tauP"] or (do_log["Sf"] or do_log["Sb"]))
        
    elif which_param == "mu_eff":
        recommend = (do_log["mu_n"] or do_log["mu_p"])
        
    elif which_param in do_log:
        recommend = do_log[which_param]
        
    else:
        recommend = False

    return recommend

def package_all_accepted(MS, names, is_active, do_log):
    means = []
    for i, param in enumerate(names):
        if not is_active.get(param, 0):
            continue
        
        a = getattr(MS.H, f"mean_{param}")
        
        if do_log[param]:
            a = np.log10(a)
            
        means.append(a)
    return np.array(means)

# Deprecated
def load_all_accepted(path, names, is_active, do_log):
    means = []
    for i, param in enumerate(names):
        if not is_active.get(param, 0):
            continue
        
        a = np.load(os.path.join(path, f"mean_{param}.npy"))
        
        if do_log[param]:
            a = np.log10(a)
            
        means.append(a)
    return np.array(means)

def binned_stderr(vals, bins):
    """
    Calculate a standard error for MC chains with states denoted by vals[] by
    computing variance of subsample means.
    
    vals[] must be evenly subdivisible when bins are applied.

    Parameters
    ----------
    vals : 1D or 2D array
        List of states or parameter values visited by MC chains. One array per chain.
        1D arrays (that come from a single chain) are promoted to 2D
    bins : 1D array
        List of indices to divide vals[] using numpy.split.

    Returns
    -------
    sub_means : 2D array
        List of means from each subdivided portion of vals[]. One list per chain.
    stderr : 1D array
        Standard error computed by sqrt(var(sub_means)). This should be divided
        by sqrt(ESS) to get a sample stderr. One value per chain.

    """
    if vals.ndim == 1:
        vals = np.expand_dims(vals, axis=0)
    
    accepted_subs = np.split(vals, bins, axis=1)
    
    lengths = np.array(list(map(lambda x: x.shape[1], accepted_subs)))
    if not np.all(lengths == lengths[0]):
        raise ValueError("Uneven binning")
        
    num_bins = len(accepted_subs)
    num_chains = len(vals)
    sub_means = np.zeros((num_chains,num_bins))
    for s, sub in enumerate(accepted_subs):
        sub_means[:, s] = np.mean(sub, axis=1)
        
    stderr = np.std(sub_means, ddof=1, axis=1)# / np.sqrt(num_bins)
        
    
    return sub_means, stderr

def ASJD(means):   
    diffs = np.diff(means, axis=2)
    diffs = diffs ** 2
    diffs = np.sum(diffs, axis=0) # [chain, iter]
    
    diffs = np.mean(diffs, axis=1) # [chain]
    return diffs

def ESS(means, do_log, verbose=True):
    if do_log:
        means = np.log10(means)

    if means.ndim == 1:
        means = np.expand_dims(means, 0)
    
    # Since R matrices are column-major ordering
    nc, nr = means.shape
    xvec = robjects.FloatVector(means.flatten())
    xr = robjects.r.matrix(xvec, nrow=nr,ncol=nc)

    each_chain_ess = calc_ESS(xr)
    print(each_chain_ess)
    return np.sum(each_chain_ess)

def geweke(means, do_log, split_l=0.1, split_r=0.5):
    if do_log:
        means = np.log10(means)

    geweke_z = gw(robjects.FloatVector(means), split_l, split_r)[0][0]
    return geweke_z

def gelman(means, do_log):
    if do_log:
        means = np.log10(means)

    if means.ndim == 1:
        means = np.expand_dims(means, 0)
    
    # Since R matrices are column-major ordering
    xr = as_mcmc_list(*[as_mcmc(robjects.FloatVector(chain)) for chain in means])
    g = gl(xr, confidence=0.95,transform=False,
           autoburnin=True,multivariate=True)
    return g

def fetch_param(MS, which_param, **kwargs):
    """
    Calculate a new param from directly sampled params

    Parameters
    ----------
    MS : MetroState()
        A MetroState object generated by metropolis.py
    which_param : str
        Name tag of new param.
    **kwargs : dict
        thickness : float
            thickness in nm of material

    Returns
    -------
    proposed : ndarray
        List of calculated new param values proposed at each iteration
    accepted : ndarray
        List of calculated new param values actually accepted by the random walk

    """
    if which_param == "Sf+Sb":
        proposed = MS.H.Sf + MS.H.Sb
        accepted = MS.H.mean_Sf + MS.H.mean_Sb
        
    elif which_param == "tauN+tauP":
        proposed = MS.H.tauN + MS.H.tauP
        accepted = MS.H.mean_tauN + MS.H.mean_tauP
        
    elif which_param == "Cn+Cp":
        proposed = MS.H.Cn + MS.H.Cp
        accepted = MS.H.mean_Cn + MS.H.mean_Cp
        
    elif which_param == "tau_eff":
        mu_a = mu_eff(MS.H.mu_n, MS.H.mu_p)
        mean_mu_a = mu_eff(MS.H.mean_mu_n, MS.H.mean_mu_p)
        
        thickness = kwargs.get("thickness", 2000)
        
        proposed = LI_tau_eff(MS.H.ks, MS.H.p0, MS.H.tauN, 
                              MS.H.Sf, MS.H.Sb, MS.H.Cp, 
                              thickness, mu_a)
        accepted = LI_tau_eff(MS.H.mean_ks, MS.H.mean_p0, MS.H.mean_tauN, 
                              MS.H.mean_Sf, MS.H.mean_Sb, MS.H.mean_Cp,
                              thickness, mean_mu_a)
        
    elif which_param == "tau_srh":
        mu_a = mu_eff(MS.H.mu_n, MS.H.mu_p)
        mean_mu_a = mu_eff(MS.H.mean_mu_n, MS.H.mean_mu_p)
        
        thickness = kwargs.get("thickness", 2000)
        
        proposed = LI_tau_srh(MS.H.tauN, MS.H.Sf, MS.H.Sb, 
                              thickness, mu_a)
        accepted = LI_tau_srh(MS.H.mean_tauN, MS.H.mean_Sf, MS.H.mean_Sb,
                              thickness, mean_mu_a)
        
    elif which_param == "mu_eff":
        proposed = mu_eff(MS.H.mu_n, MS.H.mu_p)
        accepted = mu_eff(MS.H.mean_mu_n, MS.H.mean_mu_p)
        
    elif which_param == "HI_tau_srh":
        mu_a = mu_eff(MS.H.mu_n, MS.H.mu_p)
        mean_mu_a = mu_eff(MS.H.mean_mu_n, MS.H.mean_mu_p)
        
        thickness = kwargs.get("thickness", 2000)
        proposed = HI_tau_srh(MS.H.tauN, MS.H.tauP,
                              MS.H.Sf, MS.H.Sb, thickness, mu_a)
        accepted = HI_tau_srh(MS.H.mean_tauN, MS.H.mean_tauP,
                              MS.H.mean_Sf, MS.H.mean_Sb, thickness, mean_mu_a)
        
        
    return proposed, accepted

def calc_contours(x_accepted, y_accepted, clevels, which_params, size=1000, do_logx=False, do_logy=False,xrange=None,yrange=None):
    if xrange is not None:
        minx = min(xrange)
        maxx = max(xrange)
    else:
        minx = min(x_accepted.flatten())
        maxx = max(x_accepted.flatten())
    if yrange is not None:
        miny = min(yrange)
        maxy = max(yrange)
    else:
        miny = min(y_accepted.flatten())
        maxy = max(y_accepted.flatten())
    if do_logx:
        cx = np.geomspace(minx, maxx, size)
    else:
        cx = np.linspace(minx, maxx, size)
        
    if do_logy:
        cy = np.geomspace(miny, maxy, size)
    else:
        cy = np.linspace(miny, maxy, size)

    cx, cy = np.meshgrid(cx, cy)

    if "Sf" in which_params and "Sb" in which_params:
        # Sf+Sb
        cZ = cx+cy
        
    elif "mu_n" in which_params and "mu_p" in which_params:
        cZ = mu_eff(cx, cy)
        #cZ = cx+cy
        
    elif "Sf+Sb" in which_params and "tauN" in which_params:
        kb = 0.0257 #[ev]
        q = 1
        
        D = 20 * kb / q * 10**14 / 10**9
        tau_surf = (2000 / ((cx)*0.01)) + (2000**2 / (np.pi ** 2 * D))

        cZ = (tau_surf**-1 + cy**-1)**-1
        
    elif "Sf+Sb" in which_params and "tauN+tauP" in which_params:
        kb = 0.0257 #[ev]
        q = 1
        
        D = 20 * kb / q * 10**14 / 10**9
        tau_surf = 2*(311 / ((cx)*0.01)) + (311**2 / (np.pi ** 2 * D))

        cZ = (tau_surf**-1 + cy**-1)**-1
        
    elif "Cn" in which_params and "Cp" in which_params:
        cZ = cx+cy
    else:
        print(which_params)
        raise NotImplementedError
        
    return (cx, cy, cZ, clevels)
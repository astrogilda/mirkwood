
import numpy as np
from scipy.stats import norm

EPS = 1e-6


#### error metrics #####
#deterministic metrics ######
def nrmse(yt, yp):
    yt = np.asarray(yt).flatten()
    yp = np.asarray(yp).flatten()
    iqr = (np.quantile(yt, 0.95) - np.quantile(yt, 0.05)) + EPS   
    return np.sqrt(np.mean((yt-yp)**2))/iqr#/np.mean(yt**2))

def nmae(yt, yp):
    yt = np.asarray(yt).flatten()
    yp = np.asarray(yp).flatten()
    iqr = (np.quantile(yt, 0.95) - np.quantile(yt, 0.05)) + EPS   
    return np.mean(abs(yt-yp))/iqr
    #return np.mean(abs(yt-yp))#/np.mean(abs(np.mean(yt)))

def medianae(yt, yp):
    yt = np.asarray(yt).flatten()
    yp = np.asarray(yp).flatten()
    return np.median(abs(yt-yp))#/np.mean(abs(np.mean(yt)))


def mape(yt, yp):
    yt = np.asarray(yt).flatten()
    return np.mean(abs((yt-yp)/(yt + EPS)))

def bias(yt, yp):
    yt = np.asarray(yt).flatten()
    yp = np.asarray(yp).flatten()
    return np.mean((yp>=yt)*1. + (yp<yt)*-1.)


def nbe(yt, yp):
    yt = np.asarray(yt).flatten()
    yp = np.asarray(yp).flatten()
    iqr = (np.quantile(yt, 0.95) - np.quantile(yt, 0.05)) + EPS   
    return np.mean(yp-yt)/iqr
    #return np.mean(yp-yt)#/np.mean(yt + EPS)

# probabilistic metrics ###
def ace(yt, yp, confint=0.6827):
    yt = np.asarray(yt).flatten()
    yp_mean, yp_lower, yp_upper = yp
    yp_mean = np.asarray(yp_mean).reshape(-1,)
    yp_lower = np.asarray(yp_lower).reshape(-1,)
    yp_upper = np.asarray(yp_upper).reshape(-1,)
    alpha = 1-confint
    c = np.equal(np.greater_equal(yt, yp_lower), np.less_equal(yt, yp_upper))
    ace_alpha = np.nanmean(c) - (1-alpha)
    return ace_alpha

def pinaw(yt, yp, confint=0.6827):
    yt = np.asarray(yt).flatten()
    yp_mean, yp_lower, yp_upper = yp
    yp_mean = np.asarray(yp_mean).reshape(-1,)
    yp_lower = np.asarray(yp_lower).reshape(-1,)
    yp_upper = np.asarray(yp_upper).reshape(-1,)
    alpha = 1-confint
    iqr = (np.quantile(yt, 0.95) - np.quantile(yt, 0.05)) + EPS  
    pinaw = np.mean(yp_upper - yp_lower)/(np.max(yt) - np.min(yt))
    pinaw = np.mean(yp_upper - yp_lower)/iqr
    return pinaw

def cdf_normdist(y, loc=0, scale=1):
    y = np.asarray(y).reshape(-1,)
    loc = np.asarray(loc).reshape(-1,)
    scale = np.asarray(scale).reshape(-1,)
    u = []
    for y_sample, loc_sample, scale_sample in zip(y, loc, scale):
        rv = norm(loc=loc_sample, scale=scale_sample)
        x = np.linspace(rv.ppf(q=0.01), rv.ppf(q=0.99), 1000)
        u_sample = rv.cdf(y_sample)
        u.append(u_sample)
    u = np.asarray(u)
    return u

def interval_sharpness(yt, yp, confint=0.6827):
    yt = np.asarray(yt).flatten()
    yp_mean, yp_lower, yp_upper = yp
    yp_mean = np.asarray(yp_mean).reshape(-1,)
    yp_lower = np.asarray(yp_lower).reshape(-1,)
    yp_upper = np.asarray(yp_upper).reshape(-1,)
    yt = cdf_normdist(yt, loc=yp_mean, scale=0.5*(yp_upper-yp_lower))
    alpha = 1-confint
    yp_lower = np.ones_like(yp_lower)*(0.5-confint/2)
    yp_upper = np.ones_like(yp_upper)*(0.5+confint/2)
    yp_mean = np.ones_like(yp_mean)*0.5
    delta_alpha = yp_upper - yp_lower
    intsharp = np.nanmean(np.greater_equal(yt, yp_upper)*(-2*alpha*delta_alpha - 4*(yt - yp_upper)) + np.greater_equal(yp_lower, yt)*(-2*alpha*delta_alpha - 4*(yp_lower - yt)) + -2*alpha*delta_alpha)
    return intsharp




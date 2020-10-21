
import numpy as np
EPS = 1e-6


#### error metrics #####
#deterministic metrics ######
def rmse(yt, yp):
    yt = np.asarray(yt).flatten()
    yp = np.asarray(yp).flatten()
    iqr = (np.quantile(yt, 0.95) - np.quantile(yt, 0.05)) + EPS   
    return np.sqrt(np.mean((yt-yp)**2))#/np.mean(yt**2))

def mae(yt, yp):
    yt = np.asarray(yt).flatten()
    yp = np.asarray(yp).flatten()
    iqr = (np.quantile(yt, 0.95) - np.quantile(yt, 0.05)) + EPS   
    #return np.mean(abs(yt-yp))/iqr
    return np.mean(abs(yt-yp))#/np.mean(abs(np.mean(yt)))

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


def be(yt, yp):
    yt = np.asarray(yt).flatten()
    yp = np.asarray(yp).flatten()
    iqr = (np.quantile(yt, 0.95) - np.quantile(yt, 0.05)) + EPS   
    #return np.mean(yp-yt)/iqr
    return np.mean(yp-yt)#/np.mean(yt + EPS)

# probabilistic metrics ###
def ace(yt, yp, confint=0.6827):
    yt = np.asarray(yt).flatten()
    yp_mean, yp_lower, yp_upper = yp
    yp_mean = np.asarray(yp_mean).reshape(-1,)
    yp_lower = np.asarray(yp_lower).reshape(-1,)
    yp_upper = np.asarray(yp_upper).reshape(-1,)
    alpha = 1-confint
    c = np.equal(np.greater_equal(yt, yp_lower), np.less_equal(yt, yp_upper))
    ace_alpha = np.mean(c) - (1-alpha)
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

def interval_sharpness(yt, yp, confint=0.6827):
    yt = np.asarray(yt).flatten()
    yp_mean, yp_lower, yp_upper = yp
    yp_mean = np.asarray(yp_mean).reshape(-1,)
    yp_lower = np.asarray(yp_lower).reshape(-1,)
    yp_upper = np.asarray(yp_upper).reshape(-1,)
    alpha = 1-confint
    delta_alpha = yp_upper - yp_lower
    intsharp = np.greater_equal(yt, yp_upper)*(2*alpha*delta_alpha + 4*(yt - yp_upper)) + np.less_equal(yt, yp_lower)*(2*alpha*delta_alpha + 4*(yp_lower - yt)) + np.equal(np.less_equal(yt, yp_upper), np.greater_equal(yt, yp_lower))*(2*alpha*delta_alpha)
    iqr = (np.quantile(intsharp, 0.95) - np.quantile(intsharp, 0.05)) + EPS  
    intsharp_norm = np.mean((intsharp - np.min(intsharp))/(np.max(intsharp)-np.min(intsharp)))
    intsharp_norm = np.mean(intsharp/iqr)
    return intsharp_norm




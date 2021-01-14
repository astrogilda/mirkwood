
#https://intelpython.github.io/daal4py/sklearn.html
#import daal4py.sklearn
#daal4py.sklearn.patch_sklearn()

import pandas as pd
import numpy as np

#from itertools import chain, combination
from tpot import TPOTRegressor
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics.scorer import make_scorer
import seaborn as sns

from sklearn.preprocessing import PowerTransformer as pt, MinMaxScaler as mms, StandardScaler as ss, RobustScaler as rs

import regressor_mod_trees
import re

from pprint import pprint

import warnings

import pickle

import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams.update({
    "savefig.facecolor": "w",
    "figure.facecolor" : 'w',
    "figure.figsize" : (10,8),
    "text.color": "k",
    "legend.fontsize" : 20,
    "font.size" : 30,
    "axes.edgecolor": "k",
    "axes.labelcolor": "k",
    "axes.linewidth": 3,
    "xtick.color": "k",
    "ytick.color": "k",
    "xtick.labelsize" : 25,
    "ytick.labelsize" : 25,
    "ytick.major.size" : 12,
    "xtick.major.size" : 12,
    "ytick.major.width" : 2,
    "xtick.major.width" : 2,
    "font.family": "Times",
    "mathtext.fontset" : "cm"
})
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D # for legend purposes
import colors
import matplotlib.cm as cm

EPS = 1e-6
#import seaborn as sns

'''
###### deterministic metrics ######
def nrmse(yt, yp):
    yt = np.asarray(yt).flatten()
    yp = np.asarray(yp).flatten()
    iqr = (np.quantile(yt, 0.95) - np.quantile(yt, 0.05)) + EPS   
    return np.sqrt(np.mean((yt-yp)**2)/np.mean(yt**2))

def nmae(yt, yp):
    yt = np.asarray(yt).flatten()
    yp = np.asarray(yp).flatten()
    iqr = (np.quantile(yt, 0.95) - np.quantile(yt, 0.05)) + EPS   
    #return np.mean(abs(yt-yp))/iqr
    return np.mean(abs(yt-yp))/np.mean(abs(np.mean(yt)))

def mape(yt, yp):
    yt = np.asarray(yt).flatten()
    return np.mean(abs((yt-yp)/(yt + EPS)))

def bias(yt, yp):
    yt = np.asarray(yt).flatten()
    yp = np.asarray(yp).flatten()
    return np.mean(yp>yt)


def nmbe(yt, yp):
    yt = np.asarray(yt).flatten()
    yp = np.asarray(yp).flatten()
    iqr = (np.quantile(yt, 0.95) - np.quantile(yt, 0.05)) + EPS   
    #return np.mean(yp-yt)/iqr
    return np.mean(yp-yt)/np.mean(yt + EPS)

# probabilistic metrics ###
def ace(yt, yp, confint=0.67):
    yt = np.asarray(yt).flatten()
    yp_mean, yp_lower, yp_upper = yp
    alpha = 1-confint
    c = np.equal(np.greater_equal(yt, yp_lower), np.greater_equal(yp_upper, yt))
    ace_alpha = np.mean(c) - (1-alpha)
    return ace_alpha

def pinaw(yt, yp, confint=0.67):
    yt = np.asarray(yt).flatten()
    yp_mean, yp_lower, yp_upper = yp
    alpha = 1-confint
    pinaw = np.mean(yp_upper - yp_lower)/(np.max(yt) - np.min(yt))
    return pinaw

from scipy.stats import norm
def cdf_normdist(y, loc=0, scale=1):
    y = np.asarray(y).reshape(-1,)
    loc = np.asarray(loc).reshape(-1,)
    scale = np.asarray(scale).reshape(-1,)
    u = []
    for y_sample, loc_sample, scale_sample in zip(y, loc, scale):
        rv = norm(loc=loc_sample, scale=scale_sample)
        x = np.linspace(rv.ppf(q=0.001), rv.ppf(q=0.999), 1000)
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
    intsharp = np.mean(np.greater_equal(yt, yp_upper)*(-2*alpha*delta_alpha - 4*(yt - yp_upper)) + np.greater_equal(yp_lower, yt)*(-2*alpha*delta_alpha - 4*(yp_lower - yt)) + -2*alpha*delta_alpha)
    return intsharp
'''

def logcosh(true,pred):
    loss = np.log(np.cosh(pred-true))
    return np.sum(loss)

logcosh_scorer = make_scorer(logcosh, greater_is_better=False)

def chunkify(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def noise_maker(df, df_noise, df_y, iter_per_samp=10):
    num_of_features = df.shape[1]
    df_noisy = []
    for i,j in zip(df, df_noise):
        s = np.random.default_rng().normal(i, j, (iter_per_samp, num_of_features))
        df_noisy.append(s)
    df_noisy = np.asarray(df_noisy)
    df_noisy = pd.DataFrame(np.vstack(df_noisy), columns=list(df))
    df_y_noisy = np.repeat(df_y, iter_per_samp)
    return df_noisy, df_y_noisy


### reading in files ###
simba_seds = 'simba_ml_SEDs_z0.pkl'
simba_seds_fir = 'simba_ml_SEDs_z0.0_extendedFIR.pkl'
simba_props = 'simba_ml_props_z0.pkl'

eagle_seds_fir = 'eagle_ml_SEDs_z0.0_extendedFIR.pkl'
eagle_seds2 = 'eagle_ml_SEDs_z0_pt2.pkl'
eagle_seds1 = 'eagle_ml_SEDs_z0_pt1.pkl'
eagle_props2 = 'eagle_ml_props_z0_pt2.pkl'
eagle_props1 = 'eagle_ml_props_z0_pt1.pkl'

tng_seds = 'tng_ml_SEDs_z0.0.pkl'
tng_seds_fir = 'tng_ml_SEDs_z0.0_alma_scuba.pkl'
tng_props = 'tng_ml_props_z0.0.pkl'

#simba_seds_z2 = 'simba_ml_SEDs_z2.pkl'
simba_seds_z2 = 'simba_ml_SEDs_z2.0_obs_frame.pkl'
simba_props_z2 = 'simba_ml_props_z2.pkl'

#eagle_seds_z2 = 'eagle_ml_SEDs_z2.pkl'
eagle_seds_z2 = 'eagle_ml_SEDs_z2.0_obs_frame.pkl'
eagle_props_z2 = 'eagle_ml_props_z2.pkl'

#tng_seds_z2 = 'tng_ml_SEDs_z2.pkl'
tng_seds_z2 = 'tng_ml_SEDs_z2.0_obs_frame.pkl'
tng_props_z2 = 'tng_ml_props_z2.pkl'

### EAGLE ###
X_eagle = pd.concat((pd.read_pickle(eagle_seds1), pd.read_pickle(eagle_seds2)), axis=0)
X_eagle_fir = pd.read_pickle(eagle_seds_fir)
y_eagle = pd.concat((pd.read_pickle(eagle_props1), pd.read_pickle(eagle_props2)), axis=0)

X_eagle.reset_index(drop=True, inplace=True)
X_eagle_fir.reset_index(drop=True, inplace=True)
y_eagle.reset_index(drop=True, inplace=True)

common_ids, common_idx0, common_idx1 = np.intersect1d(X_eagle['ID'].values, y_eagle['ID'].values, assume_unique=True, return_indices=True)
y_eagle = y_eagle.loc[common_idx1,:].copy()
X_eagle = X_eagle.loc[common_idx0, :].copy()
X_eagle.reset_index(inplace=True)
X_eagle.drop(columns=['index', 'ID'], inplace=True)
X_eagle_fir = X_eagle_fir.loc[common_idx0, :].copy()
X_eagle_fir.reset_index(inplace=True)
X_eagle_fir.drop(columns=['index', 'ID'], inplace=True)

titles = X_eagle['Filters'][0]
titles_fir = X_eagle_fir['Filters'][0]
fluxes_eagle = []
err_eagle = []
fluxes_eagle_fir = []
err_eagle_fir = []
for i in range(X_eagle.shape[0]):
    fluxes_eagle.append(X_eagle['Flux [Jy]'][i])
    err_eagle.append(X_eagle['Flux Err'][i])
    fluxes_eagle_fir.append(X_eagle_fir['Flux [Jy]'][i])
    err_eagle_fir.append(X_eagle_fir['Flux Err'][i])

fluxes_eagle = np.asarray(fluxes_eagle)
err_eagle = np.asarray(err_eagle)
fluxes_eagle_fir = np.asarray(fluxes_eagle_fir)
err_eagle_fir = np.asarray(err_eagle_fir)

X_eagle = pd.DataFrame(fluxes_eagle, columns=titles)
X_err_eagle = pd.DataFrame(err_eagle, columns=titles)
X_eagle_fir = pd.DataFrame(fluxes_eagle_fir, columns=titles_fir)
X_err_eagle_fir = pd.DataFrame(err_eagle_fir, columns=titles_fir)

X_eagle = pd.concat((X_eagle, X_eagle_fir), axis=1)
X_err_eagle = pd.concat((X_err_eagle, X_err_eagle_fir), axis=1)


y_eagle_id = y_eagle.pop('ID')
for prop in list(y_eagle):
    q = y_eagle[prop].values.copy() #it's a numpy array of singular lists
    idx_zerosfr = [i for i in range(X_eagle.shape[0]) if len(q[i])>1]
    for i in idx_zerosfr:
        q[i] = np.array([0.], ndmin=2)
    w = [q[i][0].item() for i in range(X_eagle.shape[0])]
    y_eagle[prop] = np.asarray(w)   


### fixing the metallicity. 08/21/20. Earlier they weren't mass weighted.
y_eagle['metallicity'] = pd.read_pickle('eagle_metallicities.pkl')['metallicity'].copy()
y_eagle['metallicity'] = 10**y_eagle['metallicity']


## finding any nans:
print('number of nans in Stellar Mass in Eagle = %d' %np.where(pd.isnull(y_eagle['stellar_mass']))[0].shape[0])
print('number of nans in SFR in Eagle = %d' %np.where(pd.isnull(y_eagle['sfr']))[0].shape[0])
print('number of nans in Metallicity in Eagle = %d' %np.where(pd.isnull(y_eagle['metallicity']))[0].shape[0])
print('number of nans in Dust Mass in Eagle = %d' %np.where(pd.isnull(y_eagle['dust_mass']))[0].shape[0])
####


logmass_eagle = np.log10(y_eagle['stellar_mass'].values)
logsfr_eagle = np.log10(1+y_eagle['sfr'].values)
logmet_eagle = np.log10(y_eagle['metallicity']).values
logdustmass_eagle = np.log10(1+y_eagle['dust_mass']).values

logmass_eagle[logmass_eagle<EPS] = 0
logsfr_eagle[logsfr_eagle<EPS] = 0
#logmet_eagle[logmet_eagle<EPS] = 0
logdustmass_eagle[logdustmass_eagle<EPS] = 0

###########################
### z = 2 ####

X_eagle2 = pd.read_pickle(eagle_seds_z2)
y_eagle2 = pd.read_pickle(eagle_props_z2)

common_ids, common_idx0, common_idx1 = np.intersect1d(X_eagle2['ID'].values, y_eagle2['ID'].values, assume_unique=True, return_indices=True)
y_eagle2 = y_eagle2.loc[common_idx1,:].copy()
X_eagle2 = X_eagle2.loc[common_idx0, :].copy()
X_eagle2.reset_index(inplace=True)
X_eagle2.drop(columns=['index', 'ID'], inplace=True)

fluxes_eagle = []
for i in range(X_eagle2.shape[0]):
    fluxes_eagle.append(X_eagle2['Flux [Jy]'][i])

fluxes_eagle = np.asarray(fluxes_eagle)
X_eagle2 = pd.DataFrame(fluxes_eagle, columns=titles+titles_fir)

y_eagle_id = y_eagle2.pop('ID')

for prop in list(y_eagle2):
    q = y_eagle2[prop].values.copy() #it's a numpy array of singular lists
    w = []
    for i in range(X_eagle2.shape[0]):
        try:
            if len(q[i])>1:
                print('whoohoo')
                w.append(0.)
            elif len(q[i])==1:
                w.append(q[i][0].item())
        except TypeError:
            w.append(q[i])
    y_eagle2[prop] = np.asarray(w)   


y_eagle2['metallicity'] = 10**y_eagle2['metallicity']

logmass_eagle2 = np.log10(y_eagle2['stellar_mass'].values)
logsfr_eagle2 = np.log10(1+y_eagle2['sfr'].values)
logmet_eagle2 = np.log10(y_eagle2['metallicity']).values
logdustmass_eagle2 = np.log10(1+y_eagle2['dust_mass']).values

logmass_eagle2[logmass_eagle2<EPS] = 0
logsfr_eagle2[logsfr_eagle2<EPS] = 0
#logmet_eagle2[logmet_eagle2<EPS] = 0
logdustmass_eagle2[logdustmass_eagle2<EPS] = 0

###########################

#########################################################
### SIMBA ###############################################
X_simba = pd.read_pickle(simba_seds)
X_simba_fir = pd.read_pickle(simba_seds_fir)
y_simba = pd.read_pickle(simba_props)

X_simba.reset_index(drop=True, inplace=True)
X_simba_fir.reset_index(drop=True, inplace=True)
y_simba.reset_index(drop=True, inplace=True)


common_ids, common_idx0, common_idx1 = np.intersect1d(X_simba['ID'].values, y_simba['ID'].values, assume_unique=True, return_indices=True)
y_simba = y_simba.loc[common_idx1,:].copy()
X_simba = X_simba.loc[common_idx0, :].copy()
X_simba.reset_index(drop=True, inplace=True)
X_simba.drop(columns=['ID'], inplace=True)
X_simba_fir = X_simba_fir.loc[common_idx0, :].copy()
X_simba_fir.drop(columns=['ID'], inplace=True)
X_simba_fir.reset_index(drop=True, inplace=True)


titles = X_simba['Filters'].loc[0].copy()
titles_fir = X_simba_fir['Filters'].loc[0].copy()
fluxes_simba = []
err_simba = []
fluxes_simba_fir = []
err_simba_fir = []
for i in range(X_simba.shape[0]):
    fluxes_simba.append(X_simba['Flux [Jy]'][i])
    err_simba.append(X_simba['Flux Err'][i])
    fluxes_simba_fir.append(X_simba_fir['Flux [Jy]'][i])
    err_simba_fir.append(X_simba_fir['Flux Err'][i])

fluxes_simba = np.asarray(fluxes_simba)
err_simba = np.asarray(err_simba)
fluxes_simba_fir = np.asarray(fluxes_simba_fir)
err_simba_fir = np.asarray(err_simba_fir)

X_simba = pd.DataFrame(fluxes_simba, columns=titles)
X_err_simba = pd.DataFrame(err_simba, columns=titles)
X_simba_fir = pd.DataFrame(fluxes_simba_fir, columns=titles_fir)
X_err_simba_fir = pd.DataFrame(err_simba_fir, columns=titles_fir)

X_simba = pd.concat((X_simba, X_simba_fir), axis=1)
X_err_simba = pd.concat((X_err_simba, X_err_simba_fir), axis=1)

#X_simba = np.log10(X_simba)

y_simba.drop(columns=['ID'], inplace=True)
for prop in list(y_simba):
    q = y_simba[prop].values.copy() #it's a numpy array of singular lists
    idx_zerosfr = [i for i in range(y_simba.shape[0]) if len(q[i])>1]
    for i in idx_zerosfr:
        q[i] = np.array([0.], ndmin=2)
    w = [q[i][0].item() for i in range(y_simba.shape[0])]
    y_simba[prop] = np.asarray(w)   


### fixing the metallicity. 08/31/20. Earlier they weren't mass weighted.
y_simba['metallicity'] = pd.read_pickle('simba_metallicities.pkl')['metallicity'].copy()
y_simba['metallicity'] = 10**y_simba['metallicity']


## finding any nans:
print('number of nans in Stellar Mass in Simba = %d' %np.where(pd.isnull(y_simba['stellar_mass']))[0].shape[0])
print('number of nans in SFR in Simba = %d' %np.where(pd.isnull(y_simba['sfr']))[0].shape[0])
print('number of nans in Metallicity in Simba = %d' %np.where(pd.isnull(y_simba['metallicity']))[0].shape[0])
print('number of nans in Dust Mass in Simba = %d' %np.where(pd.isnull(y_simba['dust_mass']))[0].shape[0])
####


logmass_simba = np.log10(y_simba['stellar_mass'].values)
logsfr_simba = np.log10(1+y_simba['sfr'].values)
logmet_simba = np.log10(y_simba['metallicity']).values
logdustmass_simba = np.log10(1+y_simba['dust_mass']).values

logmass_simba[logmass_simba<EPS] = 0
logsfr_simba[logsfr_simba<EPS] = 0
#logmet_simba[logmet_simba<EPS] = 0
logdustmass_simba[logdustmass_simba<EPS] = 0


###########################
### z = 2 ####

X_simba2 = pd.read_pickle(simba_seds_z2)
y_simba2 = pd.read_pickle(simba_props_z2)

common_ids, common_idx0, common_idx1 = np.intersect1d(X_simba2['ID'].values, y_simba2['ID'].values, assume_unique=True, return_indices=True)
y_simba2 = y_simba2.loc[common_idx1,:].copy()
X_simba2 = X_simba2.loc[common_idx0, :].copy()
X_simba2.reset_index(inplace=True)
X_simba2.drop(columns=['index', 'ID'], inplace=True)

fluxes_simba = []
for i in range(X_simba2.shape[0]):
    fluxes_simba.append(X_simba2['Flux [Jy]'][i])

fluxes_simba = np.asarray(fluxes_simba)
X_simba2 = pd.DataFrame(fluxes_simba, columns=titles+titles_fir)

y_simba_id = y_simba2.pop('ID')

for prop in list(y_simba2):
    q = y_simba2[prop].values.copy() #it's a numpy array of singular lists
    w = []
    for i in range(X_simba2.shape[0]):
        try:
            if len(q[i])>1:
                print('whoohoo')
                w.append(0.)
            elif len(q[i])==1:
                w.append(q[i][0].item())
        except TypeError:
            w.append(q[i])
    y_simba2[prop] = np.asarray(w)   


y_simba2['metallicity'] = 10**y_simba2['metallicity']

logmass_simba2 = np.log10(y_simba2['stellar_mass'].values)
logsfr_simba2 = np.log10(1+y_simba2['sfr'].values)
logmet_simba2 = np.log10(y_simba2['metallicity']).values
logdustmass_simba2 = np.log10(1+y_simba2['dust_mass']).values

logmass_simba2[logmass_simba2<EPS] = 0
logsfr_simba2[logsfr_simba2<EPS] = 0
#logmet_simba2[logmet_simba2<EPS] = 0
logdustmass_simba2[logdustmass_simba2<EPS] = 0

###########################

#########################################################
### TNG ###############################################
X_tng = pd.read_pickle(tng_seds)
X_tng_fir = pd.read_pickle(tng_seds_fir)
y_tng = pd.read_pickle(tng_props)
y_tng_fixed = pd.read_pickle('tng_ml_props_0.0_fixed_SFR_sample.pkl')

X_tng.reset_index(drop=True, inplace=True)
X_tng_fir.reset_index(drop=True, inplace=True)
y_tng.reset_index(drop=True, inplace=True)
y_tng_fixed.reset_index(drop=True, inplace=True)

common_ids, common_idx0, common_idx1 = np.intersect1d(X_tng['ID'].values, y_tng['ID'].values, assume_unique=True, return_indices=True)
y_tng = y_tng.loc[common_idx1,:]
X_tng = X_tng.loc[common_idx0, :]
X_tng.reset_index(drop=True, inplace=True)
X_tng.drop(columns=['ID'], inplace=True)
X_tng_fir = X_tng_fir.loc[common_idx0, :]
X_tng_fir.drop(columns=['ID'], inplace=True)
X_tng_fir.reset_index(drop=True, inplace=True)


titles = X_tng['Filters'].loc[0]
titles_fir = X_tng_fir['Filters'].loc[0]
fluxes_tng = []
err_tng = []
fluxes_tng_fir = []
err_tng_fir = []
for i in range(X_tng.shape[0]):
    fluxes_tng.append(X_tng['Flux [Jy]'][i])
    err_tng.append(X_tng['Flux Err'][i])
    fluxes_tng_fir.append(X_tng_fir['Flux [Jy]'][i])
    err_tng_fir.append(X_tng_fir['Flux Err'][i])

fluxes_tng = np.asarray(fluxes_tng)
err_tng = np.asarray(err_tng)
fluxes_tng_fir = np.asarray(fluxes_tng_fir)
err_tng_fir = np.asarray(err_tng_fir)

X_tng = pd.DataFrame(fluxes_tng, columns=titles)
X_err_tng = pd.DataFrame(err_tng, columns=titles)
X_tng_fir = pd.DataFrame(fluxes_tng_fir, columns=titles_fir)
X_err_tng_fir = pd.DataFrame(err_tng_fir, columns=titles_fir)

X_tng = pd.concat((X_tng, X_tng_fir), axis=1)
X_err_tng = pd.concat((X_err_tng, X_err_tng_fir), axis=1)

y_tng_ID = y_tng.pop('ID')

for prop in list(y_tng):
    q = y_tng[prop].values.copy() #it's a numpy array of singular lists
    idx_zerosfr = [i for i in range(y_tng.shape[0]) if len(q[i])>1]
    for i in idx_zerosfr:
        q[i] = np.array([0.], ndmin=2)
    w = [q[i][0].item() for i in range(y_tng.shape[0])]
    y_tng[prop] = np.asarray(w)   


## replace the old SFRs with the new ones
common_idx = np.asarray([np.where(y_tng_ID==i)[0] for i in y_tng_fixed['ID'].values]).reshape(-1,)
for i,j in enumerate(common_idx):
    y_tng.loc[j, 'sfr'] = y_tng_fixed.loc[i, 'sfr'][0].copy().item()


### fixing the metallicity. 08/21/20. Earlier they weren't mass weighted.
y_tng['metallicity'] = pd.read_pickle('tng_metallicities.pkl')['metallicity'].copy()
y_tng['metallicity'] = 10**y_tng['metallicity']

## finding any nans:
print('number of nans in Stellar Mass in tng = %d' %np.where(pd.isnull(y_tng['stellar_mass']))[0].shape[0])
print('number of nans in SFR in tng = %d' %np.where(pd.isnull(y_tng['sfr']))[0].shape[0])
print('number of nans in Metallicity in tng = %d' %np.where(pd.isnull(y_tng['metallicity']))[0].shape[0])
print('number of nans in Dust Mass in tng = %d' %np.where(pd.isnull(y_tng['dust_mass']))[0].shape[0])
####

# this is temporary
idx_sfr_is_nan = np.where(pd.isnull(y_tng['sfr']))[0]
y_tng.drop(idx_sfr_is_nan, axis=0, inplace=True)
y_tng.reset_index(drop=True, inplace=True)
X_tng.drop(idx_sfr_is_nan, axis=0, inplace=True)
X_tng.reset_index(drop=True, inplace=True)


logmass_tng = np.log10(y_tng['stellar_mass'].values)
logsfr_tng = np.log10(1+y_tng['sfr'].values)
logmet_tng = np.log10(y_tng['metallicity']).values
logdustmass_tng = np.log10(1+y_tng['dust_mass']).values

logmass_tng[logmass_tng<EPS] = 0
logsfr_tng[logsfr_tng<EPS] = 0
#logmet_tng[logmet_tng<EPS] = 0
logdustmass_tng[logdustmass_tng<EPS] = 0


###########################
### z = 2 ####

X_tng2 = pd.read_pickle(tng_seds_z2)
y_tng2 = pd.read_pickle(tng_props_z2)

common_ids, common_idx0, common_idx1 = np.intersect1d(X_tng2['ID'].values, y_tng2['ID'].values, assume_unique=True, return_indices=True)
y_tng2 = y_tng2.loc[common_idx1,:].copy()
X_tng2 = X_tng2.loc[common_idx0, :].copy()
X_tng2.reset_index(inplace=True)
X_tng2.drop(columns=['index', 'ID'], inplace=True)

fluxes_tng = []
for i in range(X_tng2.shape[0]):
    fluxes_tng.append(X_tng2['Flux [Jy]'][i])

fluxes_tng = np.asarray(fluxes_tng)
X_tng2 = pd.DataFrame(fluxes_tng, columns=titles + titles_fir)

y_tng_id = y_tng2.pop('ID')

for prop in list(y_tng2):
    q = y_tng2[prop].values.copy() #it's a numpy array of singular lists
    w = []
    for i in range(X_tng2.shape[0]):
        try:
            if len(q[i])>1:
                print('whoohoo')
                w.append(0.)
            elif len(q[i])==1:
                w.append(q[i][0].item())
        except TypeError:
            w.append(q[i])
    y_tng2[prop] = np.asarray(w)   


y_tng2['metallicity'] = 10**y_tng2['metallicity']

logmass_tng2 = np.log10(y_tng2['stellar_mass'].values)
logsfr_tng2 = np.log10(1+y_tng2['sfr'].values)
logmet_tng2 = np.log10(y_tng2['metallicity']).values
logdustmass_tng2 = np.log10(1+y_tng2['dust_mass']).values

logmass_tng2[logmass_tng2<EPS] = 0
logsfr_tng2[logsfr_tng2<EPS] = 0
#logmet_tng2[logmet_tng2<EPS] = 0
logdustmass_tng2[logdustmass_tng2<EPS] = 0


###########################

## dropping NaNs, in case ###
filternames = list(X_simba)
propnames = list(y_simba)

q = pd.concat([X_eagle2, y_eagle2], axis=1)
q.dropna(inplace=True)
X_eagle2, y_eagle2 = q.loc[:, filternames].copy(), q.loc[:, propnames].copy()

############## combine SIMBA and EAGLE and TNG ##################
y_simba['z'] = 0.
y_simba2['z'] = 2.
y_eagle['z'] = 0.
y_eagle2['z'] = 2.
y_tng['z'] = 0.
y_tng2['z'] = 2.

mulfac = 1e15

dataset_dict = {'simba': (X_simba, y_simba), 'eagle': (X_eagle, y_eagle), 'tng': (X_tng, y_tng), 'simba2': (X_simba2, y_simba2), 'eagle2': (X_eagle2, y_eagle2), 'tng2': (X_tng2, y_tng2)}

def get_data(train_data, dataset_dict=dataset_dict):
    X = pd.DataFrame()
    y = pd.DataFrame()
    for i in train_data:
        X = pd.concat((X, dataset_dict[i][0]), axis=0).reset_index().drop('index', axis=1)
        y = pd.concat((y, dataset_dict[i][1]), axis=0).reset_index().drop('index', axis=1)
    #
    #redshift = y['z'].values
    logmass = np.log10(y['stellar_mass'].values)
    logdustmass = np.log10(1+y['dust_mass']).values
    logmet = np.log10(y['metallicity']).values
    logsfr = np.log10(1+y['sfr'].values)
    #
    logmass[logmass<EPS] = 0
    logsfr[logsfr<EPS] = 0
    #logmet[logmet<EPS] = 0
    logdustmass[logdustmass<EPS] = 0
    return X*mulfac, (logmass, logdustmass, logmet, logsfr)

X, y = get_data(['simba'])
from sedpy.observate import load_filters
filters = load_filters(list(X), directory='./sedpy/data/filters')
filt_mean_wave = dict()
for filt in filters:
    #mean_wave in microns
    #filt_mean_wave[filt.name]= str([round(filt.wave_mean/10000,2), round(filt.effective_width/10000,3)])
    filt_mean_wave[filt.name]= str(round(filt.wave_mean/10000,2))

central_wav_list = [filt_mean_wave.get(i) for i in list(X)]

train_data = ['simba', 'simba2', 'eagle', 'eagle2', 'tng', 'tng2']
q=get_data(train_data)


def custom_cv(y, n_folds=10):
    np.random.seed(10)
    to_return = []
    folds =  [[] for i in range(n_folds)]
    #
    y_idx = np.argsort(y)
    n_bins = np.ceil(len(y)/n_folds)
    #
    q = chunkify(y_idx, n_bins)
    #
    for sub_arr in q:
        sub_arr_shuffled = np.random.choice(sub_arr, size=len(sub_arr), replace=False)
        for i in range(len(sub_arr)):
            folds[i].append(sub_arr_shuffled[i])
    #
    for i in range(n_folds):
        q = list(np.arange(n_folds))
        test_idx_meta = q.pop(i)
        train_idx_meta = q
        train_idx = []
        test_idx = folds[i]
        for j in train_idx_meta:
            train_idx.extend(folds[j])
        to_return.append((train_idx, test_idx))
    return to_return



'''
logmass = y['log(Mstar/Msun)'].values
logmet = y['log(Z/Zsun)'].values
logmet = np.asarray([float(i) for i in logmet])
logsfr = np.log10(1+y['SFR_100Myr']).values
logsfr[logsfr<EPS] = 0
'''

'''
#X = pd.read_csv('ml_SEDs_all.csv',index_col=0)
#y = pd.read_csv('ml_props_all.csv',index_col=0)

logmass = np.log10(y['Mstar'].values)
logsfr = np.log10(1+y['SFR'].values)
logmet = np.log10(1+y['Metallicity']).values
'''

'''
dustmass = pd.read_pickle('ml_dustmass_z0.pkl')
common_ids, common_idx0, common_idx1 = np.intersect1d(X['ID'].values, dustmass['ID'].values, assume_unique=True, return_indices=True)
dustmass = dustmass.iloc[common_idx1,:]
X = X.iloc[common_idx0, :]
X.drop(columns=['ID'], inplace=True)

reg = re.compile('.+'+'err')
list_of_err_cols = [i for i in X.columns if bool(re.match(reg, i))]
X.drop(columns=list_of_err_cols, inplace=True)

X = np.log10(X)

logdustmass = np.log10(1+dustmass['dust_mass']).values
logdustmass[logdustmass<EPS] = 0

### reduced filter set ##
galex = ['galex_FUV', 'galex_NUV']
hst_wfc3_uv  = ['wfc3_uvis_f275w', 'wfc3_uvis_f336w', 'wfc3_uvis_f475w','wfc3_uvis_f555w', 'wfc3_uvis_f606w', 'wfc3_uvis_f814w']
sdss = ['sdss_i0']
hst_wfc3_ir = ['wfc3_ir_f105w', 'wfc3_ir_f125w', 'wfc3_ir_f140w', 'wfc3_ir_f160w']
spitzer_irac = ['spitzer_irac_ch1']
spitzer_mips = ['spitzer_mips_24']
herschel_pacs = ['herschel_pacs_70', 'herschel_pacs_100', 'herschel_pacs_160']
herschel_spire = ['herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500']
jwst_miri = ['jwst_f560w', 'jwst_f770w', 'jwst_f1000w', 'jwst_f1130w', 'jwst_f1280w', 'jwst_f1500w', 'jwst_f1800w']
jwst_nircam = ['jwst_f070w', 'jwst_f090w', 'jwst_f115w', 'jwst_f150w', 'jwst_f200w', 'jwst_f277w']

reduced_filters = galex.copy()
reduced_filters.extend(hst_wfc3_uv)
reduced_filters.extend(sdss)
reduced_filters.extend(hst_wfc3_ir)
reduced_filters.extend(spitzer_irac)
reduced_filters.extend(spitzer_mips)
reduced_filters.extend(herschel_pacs)
reduced_filters.extend(herschel_spire)
reduced_filters.extend(jwst_miri)
reduced_filters.extend(jwst_nircam)
'''
#list(X) = reduced_filters
#######################################

#label1 = y.values
#import statsmodels
#statsmodels.genmod.generalized_linear_model.GLM.estimate_tweedie_power


#################################33
### shap #########################
import shap
from xgboost import XGBRegressor
from pprint import pprint
from sklearn.pipeline import make_pipeline, make_union
from tpot.export_utils import set_param_recursive

#### 06/14/2020 trying out beta distribution on the output. from here-https://github.com/guyko81/ngboost/blob/beta-distribution/examples/BetaBernoulli/NGBoost%20Beta.ipynb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from ngboost.distns import Normal, LogNormal#, Beta
from ngboost.scores import LogScore, CRPScore
import ngboost as ngb
#from learners import default_tree_learner, default_linear_learner
import multiprocessing as mp
import time

#learner = DecisionTreeRegressor(criterion='friedman_mse', max_depth=5)
learner = DecisionTreeRegressor(
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=31,
        max_depth=3,
        splitter='best')

def ngb_pipeline():
    base_model = ngb.NGBRegressor(
                Dist=LogNormal, 
                Score=LogScore, 
                Base=learner, 
                n_estimators=500, 
                learning_rate=0.04,
                col_sample=1.0,
                minibatch_frac=1.0,
                verbose=False,
                natural_gradient=True)
    return base_model




label_list=['Mass', 'Dust Mass', 'Metallicity', 'Star Formation Rate']
label_dict = {'Mass':logmass,
              'Dust Mass':logdustmass,
              'Metallicity':logmet,
              'Star Formation Rate':logsfr}
label_rev_func = {'Mass': lambda x: 10**x,
                  'Dust Mass': lambda x: 10**x - 1,
                  'Metallicity': lambda x: 10**x,
                  'Star Formation Rate': lambda x: 10**x - 1}


label_true_pd = pd.DataFrame()
label_pred_pd = pd.DataFrame()

CHAIN_FLAG = False

# %noise
x_noise=0.10
if x_noise is None:
    x_noise = 0.05

#generate_images(obs_noise=x_noise)

#def generate_images(obs_noise):
for label_str in range(len(label_list)):
    print('iteration %d started' %label_str)
    label1 = label_dict[label_list[label_str]]
    if label_str==0:
        train_idx, test_idx = custom_cv(label1, n_folds=10)[0]
        x_train, x_test = X.values[train_idx], X.values[test_idx]
        #
        x_train = np.log10(1+x_train)
        x_test = np.log10(1+x_test)
        #
        x_test_df = pd.DataFrame(x_test, columns=list(X))#mean_wave)
        x_train_df = pd.DataFrame(x_train, columns=list(X))#mean_wave)
        #
        #x_tr = pt().fit(x_train)
        #x_train = x_tr.transform(x_train)
        #x_test = x_tr.transform(x_test)
    else:
        if CHAIN_FLAG:
            x_test = np.concatenate((x_test, model.predict(x_test).reshape(-1,1)), axis=1)
            x_train = np.concatenate((x_train, model.predict(x_train).reshape(-1,1)), axis=1)
            x_test_df = pd.DataFrame(x_test, columns=list(x_test_df)+[label_list[label_str-1]])#mean_wave)
            x_train_df = pd.DataFrame(x_train, columns=list(x_train_df)+[label_list[label_str-1]])#mean_wave)
    '''
    x_tr = pt().fit(x_train)
    x_train = x_tr.transform(x_train)
    x_test = x_tr.transform(x_test)
    '''
    y_train, y_test = label1[train_idx], label1[test_idx]    
    #'''
    y_tr = mms().fit(y_train.reshape(-1,1))
    y_train = y_tr.transform(y_train.reshape(-1,1)).reshape(-1,)
    y_test = y_tr.transform(y_test.reshape(-1,1)).reshape(-1,)
    #'''
    # Fix random state for all the steps in exported pipeline
    exported_pipeline = make_pipeline(ngb_pipeline())
    #exported_pipeline = make_pipeline(XGBRegressor(colsample_bylevel=0.8, colsample_bynode=0.9, colsample_bytree=0.65, learning_rate=0.1, max_depth=9, min_child_weight=5, n_estimators=300, nthread=1, num_parallel_tree=2, objective="reg:squarederror", subsample=0.6))
    set_param_recursive(exported_pipeline.steps, 'random_state', 10)
    model = exported_pipeline[-1]
    model = model.fit(x_train, y_train, X_noise=x_noise)
    #
    #'''
    y_train_plt = y_tr.inverse_transform(y_train.reshape(-1,1)).reshape(-1,)
    y_test_plt = y_tr.inverse_transform(y_test.reshape(-1,1)).reshape(-1,)
    y_train_pred_plt = y_tr.inverse_transform(model.predict(x_train).reshape(-1,1)).reshape(-1,)
    y_test_pred_plt = y_tr.inverse_transform(model.predict(x_test).reshape(-1,1)).reshape(-1,)
    label_plt= label_list[label_str]#.split('log')[-1]
    reversify_func = label_rev_func[label_list[label_str]]
    y_train_plt = reversify_func(y_train_plt)
    y_test_plt = reversify_func(y_test_plt)
    y_train_pred_plt = reversify_func(y_train_pred_plt)
    y_test_pred_plt = reversify_func(y_test_pred_plt)
    plt.close()
    fig, ax = plt.subplots()
    ax.scatter(y_test_plt, y_test_pred_plt)#,'x')
    ax.plot(y_test_plt, y_test_plt, '-', color='black')
    ax.set_xscale('log')
    ax.set_yscale('log')    
    ax.set_xlabel('True %s'%label_plt)#, EAGLE+SIMBA')
    ax.set_ylabel('Predicted %s'%label_plt)#, EAGLE+SIMBA')
    # set the y-axis limits equal to x-axis limits
    ax.set_ylim(bottom=ax.get_xlim()[0], top=ax.get_xlim()[1])
    label_nrmse = nrmse(y_test_plt, y_test_pred_plt)
    label_nmae = nmae(y_test_plt, y_test_pred_plt)
    label_mape = mape(y_test_plt, y_test_pred_plt)
    label_bias = nmbe(y_test_plt, y_test_pred_plt)
    textstr = '\n'.join((
    r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse, ),
    r'$\mathrm{NMAE}=%.2f$' % (label_nmae, ),
    r'$\mathrm{MAPE}=%.2f$' % (label_mape, ),
    r'$\mathrm{NMBE}=%.2f$' % (label_bias, )))
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props)
    #plt.title('Trained on EAGLE+SIMBA, z=0')
    ax.set_title(label_plt)
    savename_add = '_scatter_chain_ngb' if CHAIN_FLAG else '_scatter_ngb'
    savename_add2 = savename_add + '_SNR_%d.png'%int(1/x_noise)
    savename_add += '_SNR_%d.eps'%int(1/x_noise)
    plt.savefig(label_list[label_str]+savename_add, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.savefig(label_list[label_str]+savename_add2, bbox_inches='tight', pad_inches=0.1, dpi=300)
    #plt.show()
    #'''
    #
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(model, model_output=0)#, data=x_test_df)# data=shap.kmeans(x_test_df, 100))#, l1_reg=0)#, check_additivity=False)
        ind=np.random.permutation(x_train.shape[0]) 
        shap_values_test = explainer.shap_values(x_test, check_additivity=False)
        shap_values_train = explainer.shap_values(x_train, check_additivity=False)
        #shap_values_train = explainer.shap_values(x_train[ind][:], check_additivity=False)
        '''
        shap_interaction_values_train = explainer.shap_interaction_values(x_train[ind][:])
        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]
        print(f"Explainer expected value: {expected_value}")
        '''
    #
    central_wav_list = [filt_mean_wave.get(i) if filt_mean_wave.get(i) is not None else i for i in list(x_test_df)]
    df_shap_test = pd.DataFrame(shap_values_test, columns=central_wav_list)
    df_shap_train = pd.DataFrame(shap_values_train, columns=central_wav_list)
    #
    x_test_df.columns = central_wav_list
    x_train_df.columns = central_wav_list
    # files for desika so he can plot as he wishes
    #x_train_df.to_csv(label_list[label_str]+'_x_train_df_onlyfilters_todesika.csv')
    #df_shap_train.to_csv(label_list[label_str]+'_df_shap_train_onlyfilters_todesika.csv')
    # summary plot
    sort=False
    plt.close()
    q = shap.summary_plot(df_shap_test.values, x_test_df, max_display=40, sort=sort)#, plot_type="bar")
    # files for desika so he can plot as he wishes
    #with open(label_list[label_str]+'_summary_plot_todesika.pickle', 'wb') as f:
    #    pickle.dump(q, f)
    ###################################################
    ### horizontal plotting as opposed the default vertical plotting
    plt.close('all')
    fig = plt.figure(figsize=(15,10))
    for i,j,k,l,m in zip(*q):
        _ = plt.scatter(j, i, cmap=colors.red_blue, vmin=k, vmax=l, s=16, c=m, alpha=1, zorder=3, rasterized=len(i) > 500)#,linewidth=2
    #
    labels = {
        'MAIN_EFFECT': "SHAP main effect value for\n%s",
        'INTERACTION_VALUE': "SHAP interaction value",
        'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
        'VALUE': "SHAP value (impact on model output)",
        'GLOBAL_VALUE': "mean(|SHAP value|) (average impact on model output magnitude)",
        'VALUE_FOR': "SHAP value for\n%s",
        'PLOT_FOR': "SHAP plot for %s",
        'FEATURE': "Feature %s",
        'FEATURE_VALUE': "Feature value",
        'FEATURE_VALUE_LOW': "Low",
        'FEATURE_VALUE_HIGH': "High",
        'JOINT_VALUE': "Joint SHAP value",
        'MODEL_OUTPUT': "Model output value"
    }
    #
    color_bar_label=labels["FEATURE_VALUE"]
    m = cm.ScalarMappable(cmap=colors.red_blue)
    m.set_array([0, 1])
    cb = plt.colorbar(m, ticks=[0, 1], aspect=1000)
    cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
    cb.set_label(color_bar_label, size=25, labelpad=0)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.9) * 20)
    #
    axis_color="#333333"
    max_display = 40
    feature_names = x_test_df.columns
    num_features = len(feature_names)
    max_display = min(len(feature_names), max_display)
    #
    if sort:
        feature_order = np.argsort(np.sum(np.abs(df_shap_test.values), axis=0))
        feature_order = feature_order[-max_display:]
    else:
        feature_order = np.flip(np.arange(max_display), 0)
    #
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().tick_params(color=axis_color, labelcolor=axis_color)
    #
    # flipping x and y, and adding 'rotation'
    _ = plt.xticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=15, rotation=90)
    plt.gca().tick_params('x', length=20, width=0.5, which='major')
    plt.gca().tick_params('y', labelsize=15)
    plt.xlim(-1, len(feature_order))
    plt.ylabel(labels['VALUE'], fontsize=25)
    plt.xlabel(r'Wavelength ($\mu$m)',fontsize=24)
    plt.gca().invert_xaxis()
    plt.title(label_list[label_str], fontsize=24)
    savename_add = '_shap_chain_ngb' if CHAIN_FLAG else '_shap_ngb'
    savename_add2 = savename_add + '_SNR_%d.png'%int(1/x_noise)
    savename_add += '_SNR_%d.eps'%int(1/x_noise)
    plt.savefig(label_list[label_str]+savename_add, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.savefig(label_list[label_str]+savename_add2, bbox_inches='tight', pad_inches=0.1, dpi=300)
    #plt.margins(0,0)
    #plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #plt.tight_layout()
    #plt.show()
    #return
    ##########################################
    ###########################################


# partial dependence plot
plt.close()
shap.dependence_plot('158.58', shap_values_train, x_train_df)

# shap interaction values
shap_interaction_values_train = explainer.shap_interaction_values(x_train_df)
#shap.summary_plot(shap_interaction_values_train, x_train_df, max_display=5)

tmp = np.abs(shap_interaction_values_train).sum(0)
for i in range(tmp.shape[0]):
    tmp[i,i] = 0

plt.close()
inds = np.argsort(-tmp.sum(0))[:50]
tmp2 = tmp[inds,:][:,inds]
_ = plt.figure(figsize=(12,12))
_ = plt.imshow(tmp2)
_ = plt.colorbar()
_ = plt.yticks(range(tmp2.shape[0]), x_train_df.columns[inds], rotation=0, horizontalalignment="right", fontsize=12)
_ = plt.xticks(range(tmp2.shape[0]), x_train_df.columns[inds], rotation=50.4, horizontalalignment="left", fontsize=12)
plt.gca().xaxis.tick_top()
plt.show()

#force plot
# https://slundberg.github.io/shap/notebooks/tree_explainer/Census%20income%20classification%20with%20LightGBM.html
plt.close()
shap.force_plot(explainer.expected_value, shap_values_train[:1000,:], matplotlib=True)#, X_display.iloc[:1000,:])



for j in list(x_train_df):
    plt.close()
    shap.dependence_plot(
        ("158.58", j),
        shap_interaction_values_train, x_train_df,
        show=True
    )


#simplified summary plot, code from https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d
def ABS_SHAP(df_shap, df, onx='var', top_k=None, xlabel=None, ylabel=None):
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable', 'Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    if onx=='wav':
        central_wav_list = [filt_mean_wave.get(i) if filt_mean_wave.get(i) is not None else i for i in list(feature_list)]
        corr_df = pd.concat([corr_df, pd.Series(central_wav_list)], axis=1)
        cols_corr_df = list(corr_df)
        cols_corr_df[-1] = '[Central Wavelength, Width]'
        corr_df.columns = cols_corr_df
    # Plot it
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    if top_k is not None:
        top_k = int(top_k)
        k2 = k2.loc[-top_k:,:].copy()
    colorlist = k2['Sign']
    if onx=='wav':
        xaxis='[Central Wavelength, Width]'
    else:
        xaxis = 'Variable' 
    ax = k2.plot.barh(x=xaxis,y='SHAP_abs',color = colorlist, figsize=(5,6),legend=False)
    if xlabel is None:
        ax.set_xlabel("Importance Value (Red = Positive Impact)")
    else:
        ax.set_xlabel(str(xlabel))
    if ylabel is not None:
        ax.set_ylabel(str(ylabel))
    return k2

plt.close()
k2_abs_shap = ABS_SHAP(shap_values_train, x_train_df, onx='wav', top_k=20) 
plt.show()


# partial dependence plot
plt.close()
shap.dependence_plot('WINDSPED', shap_values_train, x_train_df)

# j will be the record we explain
for j in range(10):
    plt.close()
    shap.decision_plot(base_value=explainer.expected_value, shap_values=shap_values_train[j], features=x_train_df.loc[j])#, matplotlib=True)


### 3.6  Identify typical prediction paths
# https://slundberg.github.io/shap/notebooks/plots/decision_plot.html
y_pred_train = model.predict(x_train)  # Get predictions on the probability scale.
T = x_train_df[y_pred_train >= np.log10(1)]
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sh = explainer.shap_values(T)

r = shap.decision_plot(explainer.expected_value, sh, T, feature_order='hclust', return_objects=True, ignore_warnings=True)#, feature_display_range=slice(-1,-sh.shape[1],-1))

################################################################33
##### END OF SHAP ##############################################





### reading in tpot_logsfr_fancy_3zs ###

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import OneHotEncoder, StackingEstimator, ZeroCount
from xgboost import XGBRegressor

# Average CV score on the training set was:-0.0007762360120959576
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(learning_rate=1.0, max_depth=10, min_child_weight=14, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.1)),
    ZeroCount(),
    OneHotEncoder(minimum_fraction=0.25, sparse=False, threshold=10),
    OneHotEncoder(minimum_fraction=0.1, sparse=False, threshold=10),
    MinMaxScaler(),
    ExtraTreesRegressor(bootstrap=False, max_features=0.8, min_samples_leaf=1, min_samples_split=6, n_estimators=100)
)


#### reading in tpot_logsfr ######
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# Average CV score on the training set was: -0.0711025343786716
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=False)),
    XGBRegressor(learning_rate=0.1, max_depth=7, min_child_weight=2, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.55)
)


##### reading in tpot_logsfr_fancy ########


import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# Average CV score on the training set was: -0.0003745881141605864
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        make_pipeline(
            FastICA(tol=0.55),
            FastICA(tol=0.2)
        )
    ),
    XGBRegressor(learning_rate=0.1, max_depth=4, min_child_weight=1, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.6500000000000001)
)


#### reading in tpot_logsfr_logcosh_fancy ####

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# Average CV score on the training set was: -0.0003745881141605864
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        make_pipeline(
            FastICA(tol=0.55),
            FastICA(tol=0.2)
        )
    ),
    XGBRegressor(learning_rate=0.1, max_depth=4, min_child_weight=1, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.6500000000000001)
)



####################################
## reading in tpot_logmass ##


from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LassoLarsCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


# Average CV score on the training set was: -0.0024021978312827
exported_pipeline = make_pipeline(
    RBFSampler(gamma=0.05),
    MinMaxScaler(),
    LassoLarsCV(normalize=False)
)
#######################################
## reading in fancy tpot_logmass ###

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer, PolynomialFeatures, StandardScaler
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy


# Average CV score on the training set was: -0.0014782449717589403
exported_pipeline_fancy = make_pipeline(
    make_union(
        make_union(
            FunctionTransformer(copy),
            make_union(
                make_pipeline(
                    make_union(
                        FunctionTransformer(copy),
                        make_union(
                            make_union(
                                make_union(
                                    make_pipeline(
                                        make_union(
                                            FunctionTransformer(copy),
                                            make_union(
                                                make_union(
                                                    make_union(
                                                        FunctionTransformer(copy),
                                                        FunctionTransformer(copy)
                                                    ),
                                                    FunctionTransformer(copy)
                                                ),
                                                FunctionTransformer(copy)
                                            )
                                        ),
                                        Normalizer(norm="max"),
                                        StandardScaler()
                                    ),
                                    FunctionTransformer(copy)
                                ),
                                make_union(
                                    FunctionTransformer(copy),
                                    make_union(
                                        make_union(
                                            make_union(
                                                FunctionTransformer(copy),
                                                FunctionTransformer(copy)
                                            ),
                                            VarianceThreshold(threshold=0.005)
                                        ),
                                        FunctionTransformer(copy)
                                    )
                                )
                            ),
                            make_union(
                                FunctionTransformer(copy),
                                make_union(
                                    make_union(
                                        PCA(iterated_power=5, svd_solver="randomized"),
                                        make_union(
                                            FunctionTransformer(copy),
                                            FunctionTransformer(copy)
                                        )
                                    ),
                                    FunctionTransformer(copy)
                                )
                            )
                        )
                    ),
                    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
                ),
                FunctionTransformer(copy)
            )
        ),
        make_union(
            FunctionTransformer(copy),
            make_union(
                FunctionTransformer(copy),
                FunctionTransformer(copy)
            )
        )
    ),
    RidgeCV()
)



scores = cross_val_score(exported_pipeline, X, label1, cv=custom_cv(), scoring='neg_mean_squared_error')

label1_pred = cross_val_predict(exported_pipeline, X, label1, n_jobs=-1, cv = custom_cv(n_folds=10))

label1_pred_fancy = cross_val_predict(exported_pipeline_fancy, X, label1, n_jobs=-1, cv = custom_cv())

pd.DataFrame(np.hstack(((10**label1 -1).reshape(-1,1), (10**label1_pred-1).reshape(-1,1))), columns=['dustmass', 'predicted dustmass']).to_csv('dustmass2.csv')

label1 = label1_pt.inverse_transform(label1.reshape(-1,1)).reshape(-1,)
label1_pred = label1_pt.inverse_transform(label1_pred.reshape(-1,1)).reshape(-1,)

label1_plt = label1.copy()
label1_pred_plt = label1_pred.copy()

LOG=True
if LOG:
    label1_plt = 10**label1 - 1
    label1_pred_plt = 10**label1_pred - 1

q = pd.DataFrame(np.hstack((label1_plt.reshape(-1,1), label1_pred_plt.reshape(-1,1))), columns=['true dust mass', 'pred dust mass'])
q.to_csv('dust_mass_trained_on_eagle_pred_on_simba_z=0.csv')


'''
plt.plot(label1, label1_pred - label1, 'rx', alpha=0.5, linewidth=2)
plt.plot(label1, label1_pred_fancy - label1, 'gx')
plt.hlines(0, label1.min(), label1.max())
plt.show()
'''

####### SFR plotting #####
LOG=True
plt.close()
if LOG:
    plt.plot(10**label1 - 1, 10**label1_pred - 1, 'rx')
    plt.plot(10**label1 - 1, 10**label1 - 1, 'k-')
    plt.xlabel('True mass, EAGLE')
    plt.ylabel('Predicted mass, EAGLE')
    plt.title('Trained on SIMBA, pred on EAGLE')
    plt.xscale('log')
    plt.yscale('log')    
    plt.show()
else:
    plt.plot(label1, label1_pred, 'rx')
    plt.plot(label1, label1, 'k-')
    plt.xlabel('True log dust mass')
    plt.ylabel('Predicted log dust mass')
    plt.show()


mass_pred = 10**label1_pred_fancy
pd.Series(mass_pred).to_pickle('mass_pred_fancy.pkl')





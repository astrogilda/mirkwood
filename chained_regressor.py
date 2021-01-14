
import pandas as pd
import numpy as np

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
def chunkify(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

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

simba_seds_z2 = 'simba_ml_SEDs_z2.0_obs_frame.pkl'
simba_props_z2 = 'simba_ml_props_z2.pkl'

eagle_seds_z2 = 'eagle_ml_SEDs_z2.0_obs_frame.pkl'
eagle_props_z2 = 'eagle_ml_props_z2.pkl'

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

############## combine SIMBA and EAGLE and TNG ##################
y_simba['z'] = 0.
y_eagle['z'] = 0.
y_tng['z'] = 0.

mulfac = 1e15

dataset_dict = {'simba': (X_simba, y_simba), 'eagle': (X_eagle, y_eagle), 'tng': (X_tng, y_tng)}

def get_data(train_data, dataset_dict=dataset_dict):
    X = pd.DataFrame()
    y = pd.DataFrame()
    for i in train_data:
        X = pd.concat((X, dataset_dict[i][0]), axis=0).reset_index().drop('index', axis=1)
        y = pd.concat((y, dataset_dict[i][1]), axis=0).reset_index().drop('index', axis=1)
    #
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
    filt_mean_wave[filt.name]= str(round(filt.wave_mean/10000,2))

central_wav_list = [filt_mean_wave.get(i) for i in list(X)]


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


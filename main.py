
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import time
from datetime import datetime
import gc
from pathlib import Path

from sklearn.model_selection import KFold, train_test_split, cross_val_predict, GridSearchCV, RandomizedSearchCV, ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.utils import resample
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from ngboost.distns import Normal, LogNormal#, Beta
from ngboost.scores import LogScore
from ngboost.ngboost import NGBoost
import ngboost as ngb
import pathos.multiprocessing as mp
# pathos instead of in-built python so we can pickle our 'reversify' functions. see https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function

from astropy.stats import mad_std, median_absolute_deviation as mad
import shap
from scipy import optimize
from metrics import *
from sedpy.observate import load_filters

X_simba, y_simba = pd.read_pickle('X_simba.pkl'), pd.read_pickle('y_simba.pkl')
X_eagle, y_eagle = pd.read_pickle('X_eagle.pkl'), pd.read_pickle('y_eagle.pkl')
X_tng, y_tng = pd.read_pickle('X_tng.pkl'), pd.read_pickle('y_tng.pkl')

dataset_dict = {'simba': (X_simba, y_simba), 'eagle': (X_eagle, y_eagle), 'tng': (X_tng, y_tng)}

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
filters = load_filters(list(X), directory='./sedpy/data/filters')
filt_mean_wave = dict()
for filt in filters:
    filt_mean_wave[filt.name]= str(round(filt.wave_mean/10000,2))

central_wav_list = [filt_mean_wave.get(i) for i in list(X)]

def chunkify(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

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


np.random.seed(1)

EPS = 1e-6

### Bootstrap loop. Every time 'bootstrap_func_mp' is called, it creates one bootstrap bag. 'mp' means it is called by a multithreader.
NUM_BS = 22*2
NUM_BS = max(2, NUM_BS)

# to chain or not to chain, that is the question
CHAIN_FLAG = True

def bootstrap_func_mp(estimator, x, y, x_val, x_noise=None, x_transformer=None, y_transformer=None, max_samples_best=1.0, weight_bins=10, iteration_num=1, reversifyfn=None, property_name=None, testfoldnum=0, fitting_mode=True):
    np.random.seed() #this is absolutely essential to ensure different bags pick different indices
    estimator = ngb_pipeline()
    indices = np.arange(x.shape[0])
    idx_res = resample(indices, n_samples=int(max_samples_best*len(indices)))#, random_state=np.random.randint(10000))
    #below is the line to modify to take into account observational errors
    x_res, y_res = x[idx_res], y[idx_res]
    y_res_weights = np.ones_like(y_res)
    #####################
    if x_transformer is not None:
        x_transformer = x_transformer.fit(x_res)
        x_res = x_transformer.transform(x_res)
        x_val = x_transformer.transform(x_val)
    if y_transformer is not None:
        list_of_fitted_transformers = []
        for ytr in y_transformer:
            ytr = ytr.fit(y_res.reshape(-1,1))
            y_res = ytr.transform(y_res.reshape(-1,1)).reshape(-1,)
            list_of_fitted_transformers.append(ytr)
    posixpath_strcomponent = 'ngb_prop=%s_fold=%d_bag=%d.pkl'%(property_name, testfoldnum, iteration_num)
    posixpath_shapstrcomponent = 'shap_prop=%s_fold=%d_bag=%d.pkl'%(property_name, testfoldnum, iteration_num)
    file_path = Path.home()/'desika'/posixpath_strcomponent
    shap_file_path = Path.home()/'desika'/posixpath_shapstrcomponent
    if fitting_mode:
        fitted_estimator = estimator.fit(x_res, y_res, X_noise=x_noise, sample_weight=y_res_weights)
        with file_path.open('wb') as f:
            pickle.dump(fitted_estimator, f)
    else:
        with file_path.open("rb") as f:
            estimator = pickle.load(f)
    y_pred = estimator.pred_dist(x_val)
    y_pred_mean = y_pred.loc.reshape(-1,)
    y_pred_std = y_pred.scale.reshape(-1,)
    y_pred_upper = (y_pred_mean + y_pred_std).reshape(-1,)
    y_pred_lower = (y_pred_mean - y_pred_std).reshape(-1,)
    # SHAP Summary Plots
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fitting_mode:
            explainer_mean = shap.TreeExplainer(estimator, data=shap.kmeans(x_res, 100), model_output=0)
            #print('started calculating mean shap values for bag %d'%iteration_num)
            shap_values_mean = explainer_mean.shap_values(x_val, check_additivity=False)
            with shap_file_path.open('wb') as f:
                pickle.dump(shap_values_mean, f)
        else:
            with shap_file_path.open("rb") as f:
                shap_values_mean = pickle.load(f)
    ### removing samples with high std dev.
    y_pred_upper = np.ma.masked_where(y_pred_upper>=2*y_pred_mean, y_pred_upper)
    mask = y_pred_upper.mask
    y_pred_upper = np.ma.array(y_pred_upper, mask=mask)
    y_pred_lower = np.ma.array(y_pred_lower, mask=mask)
    y_pred_mean = np.ma.array(y_pred_mean, mask=mask)
    y_pred_std = np.ma.array(y_pred_std, mask=mask)
    shap_values_mean_new = np.zeros((len(y_pred_mean), shap_values_mean.shape[1]))
    for i in range(shap_values_mean.shape[1]):
        shap_values_mean_new[:,i] = np.ma.array(shap_values_mean[:,i], mask=mask).reshape(-1,)
    shap_values_mean = shap_values_mean_new.copy()
    ##################
    if y_transformer is not None:
        for ytr in reversed(list_of_fitted_transformers):
            y_pred_upper = ytr.inverse_transform(y_pred_upper.reshape(-1,1)).reshape(-1,)
            y_pred_lower = ytr.inverse_transform(y_pred_lower.reshape(-1,1)).reshape(-1,)
            y_pred_mean = ytr.inverse_transform(y_pred_mean.reshape(-1,1)).reshape(-1,)
    ##################
    if reversifyfn is not None:
        y_pred_upper = reversifyfn(y_pred_upper)
        y_pred_lower = reversifyfn(y_pred_lower)
        y_pred_mean = reversifyfn(y_pred_mean)
    ##################
    y_pred_std = (np.ma.masked_invalid(y_pred_upper)-np.ma.masked_invalid(y_pred_lower))/2
    return np.ma.masked_invalid(y_pred_mean), np.ma.masked_invalid(y_pred_std), np.ma.masked_invalid(y_pred_lower), np.ma.masked_invalid(y_pred_upper), np.ma.masked_invalid(shap_values_mean)#, np.ma.masked_invalid(shap_values_std)

def simple_train_predict(estimator, x_df, y, cv_to_use, x_noise=None, x_transformer=None, y_transformer=None, save_boot=False, max_samples_best=0.8, weight_bins=10, reversifyfn=None, property_name=None, testfoldnum=0, fitting_mode=True):#, grid_search_best_params=None):
    yval_pred_mean_list = list()
    yval_pred_lower_list = list()
    yval_pred_upper_list = list()
    yval_pred_std_list = list()
    yval_pred_std_epis_list = list()
    yval_list = list()
    yval_shap_mean_list = list()
    yval_shap_std_list = list()
    #
    estimator_best = estimator
    for i, (train_idx, val_idx) in enumerate(cv_to_use):
        print('CV fold %d of %d'%(i+1, np.shape(cv_to_use)[0]))
        xtrain_temp = x_df.loc[train_idx].copy().reset_index(drop='index').values
        xval_temp = x_df.loc[val_idx].copy().reset_index(drop='index').values
        ytrain_temp = y[train_idx]
        yval_temp = y[val_idx]
        ytrain_weights = weightify(ytrain_temp, n_bins=weight_bins) if WEIGHT_FLAG else np.ones_like(ytrain_temp)
        ###################
        with mp.Pool() as p:
            concat_output = p.starmap(bootstrap_func_mp, [(estimator_best, xtrain_temp, ytrain_temp, xval_temp, x_noise, x_transformer, y_transformer, max_samples_best, weight_bins, i, reversifyfn, property_name, testfoldnum, fitting_mode) for i in np.arange(NUM_BS)])
            _ = gc.collect()
        mu_array = list()
        std_array = list()
        lower_array = list()
        upper_array = list()
        shap_mu_array = list()
        for i in range(NUM_BS):
            mu_array.append(concat_output[i][0])
            std_array.append(concat_output[i][1])
            lower_array.append(concat_output[i][2])
            upper_array.append(concat_output[i][3])
            shap_mu_array.append(concat_output[i][4])            
        # avoid infs. from std_array. repeat for mu_array just in case.
        mu_array = np.ma.masked_invalid(mu_array)
        std_array = np.ma.masked_invalid(std_array)
        lower_array = np.ma.masked_invalid(lower_array)
        upper_array = np.ma.masked_invalid(upper_array)
        shap_mu_array = np.ma.masked_invalid(shap_mu_array)
        yval_pred_mean = np.ma.mean(mu_array, axis=0)
        yval_pred_std = np.ma.sqrt(np.ma.mean(std_array**2, axis=0))
        yval_pred_std_epis = np.ma.std(mu_array, axis=0)
        yval_pred_lower = yval_pred_mean - yval_pred_std
        yval_pred_upper = yval_pred_mean + yval_pred_std
        yval_shap_mean = np.ma.mean(shap_mu_array, axis=0)
        if reversifyfn is not None:
            yval_temp = reversifyfn(yval_temp)
        yval_pred_mean_list.extend(yval_pred_mean)
        yval_pred_lower_list.extend(yval_pred_lower)
        yval_pred_upper_list.extend(yval_pred_upper)
        yval_pred_std_list.extend(yval_pred_std)
        yval_pred_std_epis_list.extend(yval_pred_std_epis)
        yval_list.extend(yval_temp)
        yval_shap_mean_list.extend(yval_shap_mean)
    return yval_pred_mean_list, yval_pred_std_list, yval_pred_lower_list, yval_pred_upper_list, yval_pred_std_epis_list, yval_list, yval_shap_mean_list#, yval_shap_std_list

def ngb_nobs_cv(estimator, x_df, y, x_noise=None, x_transformer=None, y_transformer=None, save_boot=False, max_samples_best=0.8, weight_bins=10, reversifyfn=None, n_folds=5):
    cv_val = custom_cv(y, n_folds=n_folds)
    max_samples_best=0.8
    grid_search_best_params = dict()
    grid_search_best_params = estimator.get_params()
    estimator_best = estimator.set_params(**grid_search_best_params)
    yval_pred_mean_list, yval_pred_std_list, yval_pred_lower_list, yval_pred_upper_list, yval_pred_std_epis_list, yval_list, yval_shap_mean_list, yval_shap_std_list = simple_train_predict(estimator=estimator_best, x_df=x_df, y=y, cv_to_use=cv_val, x_transformer=x_transformer, y_transformer=y_transformer, x_noise=x_noise, save_boot=save_boot, max_samples_best=max_samples_best, weight_bins=weight_bins, reversifyfn=reversifyfn)
    return yval_pred_mean_list, yval_pred_std_list, yval_pred_lower_list, yval_pred_upper_list, yval_pred_std_epis_list, yval_list, estimator_best, max_samples_best, yval_shap_mean_list, yval_shap_std_list

learner = DecisionTreeRegressor(
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=31,
        #max_depth=3,
        splitter='best')

def ngb_pipeline():
    base_model = ngb.NGBRegressor(
                Dist=Normal, 
                Score=LogScore, 
                Base=learner, 
                n_estimators=500, 
                learning_rate=0.04,
                col_sample=1.0,
                minibatch_frac=1.0,
                verbose=False,
                natural_gradient=True)
    return base_model

###################################################################

label_list=['Mass', 'Dust', 'Z', 'SFR']

label_dict = {'Mass':logmass,
              'Dust Mass':logdustmass,
              'Metallicity':logmet,
              'Star Formation Rate':logsfr}

label_rev_func = {'Mass': lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=20)),
                  'Dust': lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=20)) - 1,
                  'Z': lambda x: np.float_power(10, np.clip(x, a_min=-1e1, a_max=1e1)),
                  'SFR': lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=1e2)) - 1,
                  }

label_func = {'Mass': lambda x: np.log10(x),
              'Dust': lambda x: np.log10(x+1),
              'Z': lambda x: np.log10(x),
              'SFR': lambda x: np.log10(x+1),
              }

############################################################3

## number of cross validation folds
n_folds_test = 5

x_transformer = None # don't modify x_transformer at all, it'll break the addition of x_noise
y_transformer = [rs(), mms()]

test_data = None#['simba']
train_data = ['simba']#, 'eagle', 'tng']
X_simba, y_simba = get_data(['simba'])
sizeofsimba = X_simba.shape[0]

timestr = time.strftime("%Y%m%d")#-%H%M%S")
ytest_filename = 'test_metrics_' + timestr + '.csv'
val_filename = 'validation_metrics_' + timestr + '.csv'

ytest_cal_filename = 'test_cal_metrics_' + timestr + '.csv'

ytestattrib_filename = 'test_shap_' + timestr + '.pkl'
valattrib_filename = 'validation_shap_' + timestr + '.pkl'

ytestattrib_std_filename = 'test_shap_std_' + timestr + '.pkl'
valattrib_std_filename = 'validation_shap_std_' + timestr + '.pkl'

trained_model_filename = 'trained_ngboost_' + timestr + '.pkl'

##########################################
##########################################
time_start = time.time()
for x_noise in [1/5, 1/10, 1/20]:
    for label_str in [0,1,2,3]:
        y_tests = list()
        y_test_preds_std_epis = list()
        y_test_preds_std = list()
        y_test_preds_mean = list()
        y_test_preds_lower = list()
        y_test_preds_upper = list()
        #### shap ####
        y_test_shaps_mean = list()
        #### input fluxes ###
        x_trains = list()
        x_tests = list()
        #
        plt.close('all')
        snr = 1000 if x_noise==0 else 1/x_noise
        print('Label=%s, SNR=%d'%(label_list[label_str], int(snr)))
        #
        if (test_data is not None) and ('simba' not in train_data):
            print('training on %s and testing on %s'%(train_data, test_data))
            X_train, logy_train = get_data(train_data)
            X_test, logy_test = get_data(test_data)
            X = pd.concat((X_train, X_test), axis=0).reset_index(drop=True)
            if CHAIN_FLAG:
                label1_train = list()
                label1_test = list()
                for i in range(0, label_str+1):
                    label1_train.append(logy_train[i])
                    label1_test.append(logy_test[i])
                label1_train = np.transpose(np.asarray(label1_train))
                label1_test = np.transpose(np.asarray(label1_test))
                label1 = np.append(label1_train, label1_test, axis=0)
                train_val_idxs, test_idxs = zip(*[(list(np.arange(len(label1_train))), list(np.arange(len(label1_train), len(label1))))])
            else:
                label1_train = logy_train[label_str]
                label1_test = logy_test[label_str]
                label1 = np.append(label1_train, label1_test)
                train_val_idxs, test_idxs = zip(*[(list(np.arange(len(label1_train))), list(np.arange(len(label1_train), len(label1))))])
        elif (test_data is not None) and ('simba' in train_data):
            print('training and testing on %s'%train_data)
            X, logy = get_data(train_data)
            if CHAIN_FLAG:
                label1 = list()
                label1_simba = list()
                label1_rest = list()
                for i in range(0, label_str+1):
                    label1.append(logy[i])
                    label1_simba.append(logy[i][:sizeofsimba])
                    label1_rest.append(logy[i][sizeofsimba:])
                label1 = np.transpose(np.asarray(label1))
                label1_simba = np.transpose(np.asarray(label1_simba))
                label1_rest = np.transpose(np.asarray(label1_rest))
                train_val_idxs, test_idxs = zip(*custom_cv(label1_simba[:,-1], n_folds=n_folds_test)[0:])
                _ = [i.extend(list(np.arange(sizeofsimba, len(label1_simba)+len(label1_rest)))) for i in train_val_idxs]
            else:
                label1_simba = logy[label_str][:sizeofsimba]
                label1_rest = logy[label_str][sizeofsimba:]
                train_val_idxs, test_idxs = zip(*custom_cv(label1_simba, n_folds=n_folds_test)[0:])
                _ = [i.extend(list(np.arange(sizeofsimba, len(label1_simba)+len(label1_rest)))) for i in train_val_idxs]
        #
        elif test_data is None:
            print('training and testing on %s'%train_data)
            X, logy = get_data(train_data)
            if CHAIN_FLAG:
                label1 = list()
                for i in range(0, label_str+1):
                    label1.append(logy[i])
                label1 = np.transpose(np.asarray(label1))
                train_val_idxs, test_idxs = zip(*custom_cv(label1[:,-1], n_folds=n_folds_test)[0:])
            else:
                train_val_idxs, test_idxs = zip(*custom_cv(label1, n_folds=n_folds_test)[0:])
        for n_fold_test_inuse, (train_val_idx, test_idx) in enumerate(zip(train_val_idxs, test_idxs)):
            ytest_filename_thisfold = ytest_filename.split('.csv')[0] + '_' + label_list[label_str] + str(n_fold_test_inuse+1) + 'of' + str(min(n_folds_test,np.shape(train_val_idxs)[0])) + '.csv'
            #
            ytestattrib_filename_thisfold = ytestattrib_filename.split('.pkl')[0] + '_' + label_list[label_str] + '_' + str(n_fold_test_inuse+1) + 'of' + str(min(n_folds_test,np.shape(train_val_idxs)[0])) + '.pkl'
            ytestattrib_std_filename_thisfold = ytestattrib_std_filename.split('.pkl')[0] + '_' + label_list[label_str] + str(n_fold_test_inuse+1) + 'of' + str(min(n_folds_test,np.shape(train_val_idxs)[0])) + '.pkl'
            #
            trained_model_filename_thisfold = trained_model_filename.split('.pkl')[0] + '_' + label_list[label_str] + '_' + str(n_fold_test_inuse+1) + 'of' + str(min(n_folds_test,np.shape(train_val_idxs)[0])) + '.pkl'
            #
            X_train_val = X.loc[train_val_idx].copy().reset_index(drop='index')
            y_train_val = label1[train_val_idx]
            #
            label_str_orig = label_str
            lowerlim = not(CHAIN_FLAG)*label_str_orig if label_str_orig!=0 else 0
            x_df = np.log10(1+X)
            x_noise_arr = x_noise * np.ones_like(X)[0]
            for label_str_iter in range(lowerlim, label_str_orig+1):
                prop = label_list[label_str_iter]
                print('starting predictions on test set, for propertry %s'%prop)
                weight_bins = int(np.sqrt(len(y_train_val)/4))#50
                max_samples_best = 0.8
                reversify_func = label_rev_func[label_list[label_str_iter]]
                y = label1[:,label_str_iter]
                fitting_mode = label_str_iter==label_str_orig
                if fitting_mode:
                    print('Fitting and not loading')
                else:
                    print('loading saved model')
                y_test_pred_mean, y_test_pred_std, y_test_pred_lower, y_test_pred_upper, y_test_pred_std_epis, y_test, y_test_shap_mean= simple_train_predict(estimator=ngb_pipeline(), x_df=x_df, y=y, cv_to_use=[(train_val_idx, test_idx)], x_transformer=x_transformer, y_transformer=y_transformer, x_noise=x_noise_arr, save_boot=False, max_samples_best=max_samples_best, weight_bins=weight_bins, reversifyfn=reversify_func, property_name=prop, testfoldnum=n_fold_test_inuse, fitting_mode=fitting_mode)#, y_test_shap_std 
                y_test_pred_mean = np.asarray(y_test_pred_mean).reshape(-1,1)
                y_test_pred_std = np.asarray(y_test_pred_std).reshape(-1,1)
                y_test_pred_lower = np.asarray(y_test_pred_lower).reshape(-1,1)
                y_test_pred_upper = np.asarray(y_test_pred_upper).reshape(-1,1)
                y_test_pred_std_epis = np.asarray(y_test_pred_std_epis).reshape(-1,1)
                y_test = np.asarray(y_test).reshape(-1,1)
                y_test_pred_std_final = np.sqrt(y_test_pred_std**2 + y_test_pred_std_epis**2).reshape(-1,1)
                y_test_shap_mean = np.array(y_test_shap_mean)
                #############################################################
                x_df[label_list[label_str_iter]] = np.log10(1 + np.append(reversify_func(label1[train_val_idx, label_str_iter]), y_test_pred_mean))
                x_noise_arr = np.append(x_noise_arr, 0.)
            #
            pd.DataFrame(np.hstack((y_test, y_test_pred_mean, y_test_pred_lower, y_test_pred_upper, y_test_pred_std_epis)), columns=['true', 'pred_mean', 'pred_lower', 'pred_upper', 'pred_epis_std']).to_csv(ytest_filename_thisfold)
            #
            pd.DataFrame(y_test_shap_mean).to_pickle(ytestattrib_filename_thisfold)
            #
            print('inner for loop complete. total run time = %.1f minutes'%((time.time() - time_start)/60))
            # appending results of individual folds to lists:
            y_tests.extend(y_test)
            y_test_preds_std_epis.extend(y_test_pred_std_epis)
            y_test_preds_std.extend(y_test_pred_std)
            y_test_preds_mean.extend(y_test_pred_mean)
            y_test_preds_lower.extend(y_test_pred_lower)
            y_test_preds_upper.extend(y_test_pred_upper)
            y_test_shaps_mean.extend(y_test_shap_mean)
            x_trains.extend(x_df.loc[train_val_idx,list(x_df)[:-1]].copy().values)
            x_tests.extend(x_df.loc[test_idx,list(x_df)[:-1]].copy().values)
        #convert lists to arrays and reshape
        y_tests = np.asarray(y_tests).reshape(-1,1)
        y_test_preds_std_epis = np.asarray(y_test_preds_std_epis).reshape(-1,1)
        y_test_preds_std = np.asarray(y_test_preds_std).reshape(-1,1)
        y_test_preds_mean = np.asarray(y_test_preds_mean).reshape(-1,1)
        y_test_preds_lower = np.asarray(y_test_preds_lower).reshape(-1,1)
        y_test_preds_upper = np.asarray(y_test_preds_upper).reshape(-1,1)
        y_test_shaps_mean = np.asarray(y_test_shaps_mean)
        #y_test_shaps_std = np.asarray(y_test_shaps_std)
        x_trains = np.asarray(x_trains)
        x_tests = np.asarray(x_tests)
        ### saving results ###
        a = pd.DataFrame(np.hstack((y_tests, y_test_preds_mean, y_test_preds_lower, y_test_preds_upper, y_test_preds_std_epis)), columns=['true', 'pred_mean', 'pred_lower', 'pred_upper', 'pred_std_epis'])
        a.to_csv('uncal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(CHAIN_FLAG))+timestr+'.csv')
        b = pd.DataFrame(y_test_shaps_mean, columns=list(x_df)[:-1])
        b.to_csv('shapmea_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(CHAIN_FLAG))+timestr+'.csv')
        xtrn = pd.DataFrame(10**(x_trains)-1, columns=list(x_df)[:-1])
        xtrn.to_csv('xtrain_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(CHAIN_FLAG))+timestr+'.csv')
        xtst = pd.DataFrame(10**(x_tests)-1, columns=list(x_df)[:-1])
        xtst.to_csv('xtest_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(CHAIN_FLAG))+timestr+'.csv')
        print('run time = %.1f minutes'%((time.time() - time_start)/60))
############################################################


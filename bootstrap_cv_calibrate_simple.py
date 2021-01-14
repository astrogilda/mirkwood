
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from ngboost.distns import Normal, NormalFixedVar
from ngboost.ngboost import NGBoost
from ngboost.scores import MLE, CRPS
#from examples.experiments.loggers import RegressionLogger

from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.metrics import mean_squared_error
import time
from sklearn.model_selection import KFold

np.random.seed(1)

from datetime import datetime

'''
from ngboost_rongba import NGBoost#NGBRegressor
from distns import Normal, NormalFixedVar
from scores import MLE, CRPS
#from sklearn.datasets import load_boston
'''

import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from scipy import optimize
import gc
#from calibrate import *
from metrics import *

EPS = 1e-6


'''
#### error metrics #####
#deterministic metrics ######
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
    return np.mean((yp>=yt)*1. + (yp<yt)*-1.)

def nbe(yt, yp):
    yt = np.asarray(yt).flatten()
    yp = np.asarray(yp).flatten()
    iqr = (np.quantile(yt, 0.95) - np.quantile(yt, 0.05)) + EPS   
    #return np.mean(yp-yt)/iqr
    return np.mean(yp-yt)/np.mean(yt + EPS)

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
'''

'''
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
'''
########################

from balanced_binning import BalancedBinningReference
def weightify(y_train, n_bins=50):
    # reweight samples by inverse square root of frequency.
    # 09/03/2020. below is taken from https://juan0001.github.io/Balanced_bining_reference_visualizer/
    # this takes as input the number of bins, and finds splits such that all classes are equally balanced.
    # not exactly what we need. we want to take as input number of bins, and use some method to find splits such that some metric (say entropy) is minimized. or take a fixed number of bins.
    '''
    visualizer = BalancedBinningReference(bins=n_bins)
    splits = visualizer.get_vline_value(y_train)
    # bin the data based on the reference using numpy.digitize
    # set the binning point
    # you can copy the binning reference values and paste here
    bins = np.insert(np.asarray(splits), 0, y_train.min()-1e-5) 
    bins = np.append(bins, y_train.max()+1e-5) 
    samples_idx = np.digitize(y_train, bins,right=True)
    samples_per_bin = np.unique(samples_idx, return_counts=True)[1]
    '''
    #####################################################################    
    #'''
    samples_per_bin, bin_edges = np.histogram(y_train, bins=n_bins)
    bin_edges[0] = bin_edges[0] - 1e-5
    bin_edges[-1] = bin_edges[-1] + 1e-5
    samples_idx = np.digitize(y_train, bins=bin_edges)
    #'''
    ########################################################################
    num_samples = len(y_train)
    #trying the new way from here: https://towardsdatascience.com/handling-class-imbalanced-data-using-a-loss-specifically-made-for-it-6e58fd65ffab
    beta = 0.9
    effective_num = 1.0 - np.power(beta, samples_per_bin)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights/np.sum(weights) * n_bins
    samples_weight = np.asarray([weights[i] for i in (samples_idx-1)])
    '''
    samples_freq = np.asarray([samples_per_bin[i]/num_samples for i in (samples_idx-1)])
    samples_freq = 1/samples_freq #1/np.sqrt(samples_freq)
    samples_weight = samples_freq/np.sum(samples_freq) * n_bins
    '''
    return samples_weight


### Bootstrap loop. Every time 'bootstrap_func_mp' is called, it creates one bootstrap bag. 'mp' means it is called by a multithreader.
# to save trained ngb models
from pathlib import Path
from sklearn.utils import resample
NUM_BS = 22*1
NUM_BS = max(2, NUM_BS)
WEIGHT_FLAG = False

def bootstrap_func_mp(estimator, x, y, x_val, x_noise=None, x_transformer=None, y_transformer=None, max_samples_best=1.0, weight_bins=10, iteration_num=1, reversifyfn=None, property_name=None, testfoldnum=0, fitting_mode=True):
    np.random.seed() #this is absolutely essential to ensure different bags pick different indices
    estimator = ngb_pipeline()
    indices = np.arange(x.shape[0])
    idx_res = resample(indices, n_samples=int(max_samples_best*len(indices)))#, random_state=np.random.randint(10000))
    #below is the line to modify to take into account observational errors
    x_res, y_res = x[idx_res], y[idx_res]
    y_res_weights = weightify(y_res, n_bins=weight_bins) if WEIGHT_FLAG else np.ones_like(y_res)
    #print('bootstrap bag min idx=%d, max idx=%d'%(idx_res.min(), idx_res.max()))
    #print('bootstrap bag min. weight=%.2f, max. weight=%.2f, median weight=%.2f'%(y_res_weights.min(), y_res_weights.max(), np.median(y_res_weights)))
    #x_res, x_val = np.log10(x_res), np.log10(x_val)
    #####################
    if x_transformer is not None:
        #print('fitting x_tr for bag #{}'.format(iteration_num))
        x_transformer = x_transformer.fit(x_res)
        x_res = x_transformer.transform(x_res)
        x_val = x_transformer.transform(x_val)
    if y_transformer is not None:
        #print('fitting y_tr for bag #{}'.format(iteration_num))
        list_of_fitted_transformers = []
        for ytr in y_transformer:
            ytr = ytr.fit(y_res.reshape(-1,1))
            y_res = ytr.transform(y_res.reshape(-1,1)).reshape(-1,)
            list_of_fitted_transformers.append(ytr)
    #print('ngboost fit started for bootstrap bag #{}'.format(iteration_num))
    posixpath_strcomponent = 'ngb_prop=%s_fold=%d_bag=%d.pkl'%(property_name, testfoldnum, iteration_num)
    posixpath_shapstrcomponent = 'shap_prop=%s_fold=%d_bag=%d.pkl'%(property_name, testfoldnum, iteration_num)
    file_path = Path.home()/'desika'/posixpath_strcomponent
    shap_file_path = Path.home()/'desika'/posixpath_shapstrcomponent
    if fitting_mode:
        #print('fitting mode engaged')
        fitted_estimator = estimator.fit(x_res, y_res, X_noise=x_noise, sample_weight=y_res_weights)
        with file_path.open('wb') as f:
            pickle.dump(fitted_estimator, f)
    else:
        with file_path.open("rb") as f:
            estimator = pickle.load(f)
    #
    #print('ngboost fit complete for bootstrap bag #{}'.format(iteration_num))
    y_pred = estimator.pred_dist(x_val)
    #print('predicted y_pdf for x_val for bootstrap bag #{}'.format(iteration_num))
    y_pred_mean = y_pred.loc.reshape(-1,)
    ##print('extracted y_mean for y_val for bootstrap bag #{}'.format(iteration_num))
    #print('bootstrap bag #%d: max, min of y_mean is %.2f, %.2f'%(iteration_num, np.ma.max(y_pred_mean), np.ma.min(y_pred_mean)))
    y_pred_std = y_pred.scale.reshape(-1,)
    ##print('extracted y_std for y_val for bootstrap bag #{}'.format(iteration_num))
    #print('bootstrap bag #%d: max, min of y_std is %.2f, %.2f'%(iteration_num, np.ma.max(y_pred_std), np.ma.min(y_pred_std)))
    y_pred_upper = (y_pred_mean + y_pred_std).reshape(-1,)
    y_pred_lower = (y_pred_mean - y_pred_std).reshape(-1,)
    #print('bootstrap bag #%d: max, min of y_pred_upper is %.2f, %.2f'%(iteration_num, np.ma.max(y_pred_upper), np.ma.min(y_pred_upper)))
    #print('bootstrap bag #%d: max, min of y_pred_lower is %.2f, %.2f'%(iteration_num, np.ma.max(y_pred_lower), np.ma.min(y_pred_lower)))
    #y_pred_std[y_pred_std>np.max(np.abs(y_pred_mean))] = np.max(np.abs(y_pred_mean))
    ##print('new max, min of y_std is %.2f, %.2f'%(y_pred_std.max(), y_pred_std.min()))
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
        #print('finished calculating mean shap values for bag %d'%iteration_num)
        ##print('bootstrap bag #%d: shap_values_mean calculated'%iteration_num)
        #print('calculating std shap values')
        #explainer_std = shap.TreeExplainer(estimator, data=shap.kmeans(x_res, 1000), model_output=1)
        #shap_values_std = explainer_std.shap_values(x_val, check_additivity=False)
        ##print('bootstrap bag #%d: shap_values_std calculated'%iteration_num)
    ### removing samples with high std dev.
    y_pred_upper = np.ma.masked_where(y_pred_upper>=2*y_pred_mean, y_pred_upper)
    mask = y_pred_upper.mask
    y_pred_upper = np.ma.array(y_pred_upper, mask=mask)
    y_pred_lower = np.ma.array(y_pred_lower, mask=mask)
    y_pred_mean = np.ma.array(y_pred_mean, mask=mask)
    y_pred_std = np.ma.array(y_pred_std, mask=mask)
    shap_values_mean_new = np.zeros((len(y_pred_mean), shap_values_mean.shape[1]))
    #shap_values_std_new = np.zeros((len(y_pred_mean), shap_values_std.shape[1]))
    for i in range(shap_values_mean.shape[1]):
        shap_values_mean_new[:,i] = np.ma.array(shap_values_mean[:,i], mask=mask).reshape(-1,)
        #shap_values_std_new[:,i] = np.ma.array(shap_values_std[:,i], mask=mask).reshape(-1,)
    shap_values_mean = shap_values_mean_new.copy()
    #shap_values_std = shap_values_std_new.copy()
    print(np.max(y_pred_upper), np.min(y_pred_upper), np.median(y_pred_upper))
    ###
    if y_transformer is not None:
        ##print('inverse fitting y_tr for bag #{}'.format(iteration_num))
        for ytr in reversed(list_of_fitted_transformers):
            y_pred_upper = ytr.inverse_transform(y_pred_upper.reshape(-1,1)).reshape(-1,)
            #print('bootstrap bag #%d: inverse fitting y_pred_upper successful, max=%.2f and min=%.2f are '%(iteration_num, np.ma.max(y_pred_upper), np.ma.min(y_pred_upper)))
            y_pred_lower = ytr.inverse_transform(y_pred_lower.reshape(-1,1)).reshape(-1,)
            #print('bootstrap bag #%d: inverse fitting y_pred_lower successful, max=%.2f and min=%.2f are '%(iteration_num, np.ma.max(y_pred_lower), np.ma.min(y_pred_lower)))
            y_pred_mean = ytr.inverse_transform(y_pred_mean.reshape(-1,1)).reshape(-1,)
            #print('bootstrap bag #%d: inverse fitting y_pred_mean successful, max=%.2f and min=%.2f are '%(iteration_num, np.ma.max(y_pred_mean), np.ma.min(y_pred_mean)))
        #print('after inverse y_transforming:', np.max(y_pred_upper), np.min(y_pred_upper), np.median(y_pred_upper))
    ##################3
    if reversifyfn is not None:
        #print('reversifying predictions')
        y_pred_upper = reversifyfn(y_pred_upper)
        #print('bootstrap bag #%d: max, min of y_pred_upper_reversified is %.2f, %.2f'%(iteration_num, np.ma.max(y_pred_upper), np.ma.min(y_pred_upper)))
        y_pred_lower = reversifyfn(y_pred_lower)
        #print('bootstrap bag #%d: max, min of y_pred_lower_reversified is %.2f, %.2f'%(iteration_num, np.ma.max(y_pred_lower), np.ma.min(y_pred_lower)))
        y_pred_mean = reversifyfn(y_pred_mean)
        #print('bootstrap bag #%d: max, min of y_pred_upper_reversified is %.2f, %.2f'%(iteration_num, np.ma.max(y_pred_mean), np.ma.min(y_pred_mean)))
        #print('after reversifying y:', np.max(y_pred_upper), np.min(y_pred_upper), np.median(y_pred_upper))
    #
    y_pred_std = (np.ma.masked_invalid(y_pred_upper)-np.ma.masked_invalid(y_pred_lower))/2
    return np.ma.masked_invalid(y_pred_mean), np.ma.masked_invalid(y_pred_std), np.ma.masked_invalid(y_pred_lower), np.ma.masked_invalid(y_pred_upper), np.ma.masked_invalid(shap_values_mean)#, np.ma.masked_invalid(shap_values_std)


MEDIANFLAG = True

from astropy.stats import mad_std, median_absolute_deviation as mad
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
    #if grid_search_best_params is not None:
    #    estimator_best = estimator.set_params(**grid_search_best_params['estimator'])
    #
    for i, (train_idx, val_idx) in enumerate(cv_to_use):
        print('CV fold %d of %d'%(i+1, np.shape(cv_to_use)[0]))
        xtrain_temp = x_df.loc[train_idx].copy().reset_index(drop='index').values
        xval_temp = x_df.loc[val_idx].copy().reset_index(drop='index').values
        ytrain_temp = y[train_idx]
        yval_temp = y[val_idx]
        #print(np.shape(yval_temp))
        ytrain_weights = weightify(ytrain_temp, n_bins=weight_bins) if WEIGHT_FLAG else np.ones_like(ytrain_temp)
        if BOOT_FLAG==False:
            if y_transformer is not None:
                ytrain_temp = y_transformer.fit_transform(y[train_idx].reshape(-1,1)).reshape(-1,)
            if x_transformer is not None:
                xtrain_temp = x_transformer.fit_transform(x_df.loc[train_idx].copy().reset_index(drop='index').values, y[train_idx])
                xval_temp = x_transformer.transform(x_df.loc[val_idx].copy().reset_index(drop='index').values)
            #
            estimator_best = estimator_best.fit(xtrain_temp, ytrain_temp, X_noise=x_noise, sample_weight=ytrain_weights)
            yval_pred = estimator_best.pred_dist(xval_temp)
            yval_pred_mean = yval_pred.loc.reshape(-1,)
            yval_pred_std = yval_pred.scale.reshape(-1,)
            yval_pred_std_epis = np.zeros_like(yval_pred_std)
            #
            if y_transformer is not None:
                yval_pred_mean = y_transformer.inverse_transform(yval_pred_mean.reshape(-1,1)).reshape(-1,)
                yval_pred_upper = y_transformer.inverse_transform((yval_pred_mean + yval_pred_std).reshape(-1,1)).reshape(-1,)
                yval_pred_lower = y_transformer.inverse_transform((yval_pred_mean - yval_pred_std).reshape(-1,1)).reshape(-1,)
                yval_pred_std = (yval_pred_upper-yval_pred_lower)/2
            #
        else:
            with mp.Pool() as p:
                #concat_output = p.starmap(bootstrap_func_mp, [(estimator_best, xtrain_temp, ytrain_temp, xval_temp, x_noise=x_noise, x_transformer=x_transformer, y_transformer=y_transformer, max_samples_best=max_samples_best) for i in np.arange(NUM_BS)])
                #print('Beginning bootstrappoing. Number of bootstrap bags = %d'%NUM_BS)
                concat_output = p.starmap(bootstrap_func_mp, [(estimator_best, xtrain_temp, ytrain_temp, xval_temp, x_noise, x_transformer, y_transformer, max_samples_best, weight_bins, i, reversifyfn, property_name, testfoldnum, fitting_mode) for i in np.arange(NUM_BS)])
                _ = gc.collect()
            #shape of concat_output = (num_bs, 2, x_val.shape[0])
            #need to change this to (2, num_bs, x_val.shape[0])
            #concat_output = np.einsum('ijk->jik', np.asarray(concat_output))
            #mu_array = np.asarray(concat_output[0])
            #std_array = np.asarray(concat_output[1])
            mu_array = list()
            std_array = list()
            lower_array = list()
            upper_array = list()
            shap_mu_array = list()
            #shap_std_array = list()
            for i in range(NUM_BS):
                mu_array.append(concat_output[i][0])
                std_array.append(concat_output[i][1])
                lower_array.append(concat_output[i][2])
                upper_array.append(concat_output[i][3])
                shap_mu_array.append(concat_output[i][4])            
                #shap_std_array.append(concat_output[i][5])            
            # avoid infs. from std_array. repeat for mu_array just in case.
            mu_array = np.ma.masked_invalid(mu_array)
            std_array = np.ma.masked_invalid(std_array)
            lower_array = np.ma.masked_invalid(lower_array)
            upper_array = np.ma.masked_invalid(upper_array)
            shap_mu_array = np.ma.masked_invalid(shap_mu_array)
            #shap_std_array = np.ma.masked_invalid(shap_std_array)
            #08/18/2020. replacing mean with median and std with MAD in below 3 lines
            yval_pred_mean = np.ma.mean(mu_array, axis=0)
            #yval_pred_lower = np.ma.mean(lower_array, axis=0)
            #yval_pred_upper = np.ma.mean(upper_array, axis=0)
            yval_pred_std = np.ma.sqrt(np.ma.mean(std_array**2, axis=0))
            yval_pred_std_epis = np.ma.std(mu_array, axis=0)
            yval_pred_lower = yval_pred_mean - yval_pred_std
            yval_pred_upper = yval_pred_mean + yval_pred_std
            yval_shap_mean = np.ma.mean(shap_mu_array, axis=0)
            #yval_shap_std = np.ma.mean(shap_std_array, axis=0)
            if MEDIANFLAG:
                yval_pred_mean = np.ma.median(mu_array, axis=0)
                yval_pred_std = np.ma.sqrt(np.ma.median(std_array**2, axis=0))
                yval_pred_std_epis = mad_std(mu_array, axis=0)
                #yval_pred_lower = np.ma.median(lower_array, axis=0)
                #yval_pred_upper = np.ma.median(upper_array, axis=0)
                yval_pred_lower = yval_pred_mean - yval_pred_std
                yval_pred_upper = yval_pred_mean + yval_pred_std
                yval_shap_mean = np.ma.median(shap_mu_array, axis=0)
                #yval_shap_std = np.ma.median(shap_std_array, axis=0)
            if reversifyfn is not None:
                yval_temp = reversifyfn(yval_temp)
            #print(np.shape(yval_pred_mean))
            '''
            ### removing samples with high std dev.
            yval_pred_upper = np.ma.masked_where(yval_pred_upper>=2*yval_pred_mean, yval_pred_upper)
            mask = yval_pred_upper.mask
            yval_pred_upper = yval_pred_upper.compressed()
            yval_pred_lower = np.ma.array(yval_pred_lower, mask=mask).compressed()
            yval_pred_mean = np.ma.array(yval_pred_mean, mask=mask).compressed()
            yval_pred_std = np.ma.array(yval_pred_std, mask=mask).compressed()
            yval_pred_std_epis = np.ma.array(yval_pred_std_epis, mask=mask).compressed()
            yval_temp = np.ma.array(yval_temp, mask=mask).compressed()
            yval_shap_mean_new = np.zeros((len(yval_temp), yval_shap_mean.shape[1]))
            yval_shap_std_new = np.zeros((len(yval_temp), yval_shap_std.shape[1]))
            for i in range(yval_shap_mean.shape[1]):
                yval_shap_mean_new[:,i] = np.ma.array(yval_shap_mean[:,i], mask=mask).compressed().reshape(-1,)
                yval_shap_std_new[:,i] = np.ma.array(yval_shap_std[:,i], mask=mask).compressed().reshape(-1,)
            yval_shap_mean = yval_shap_mean_new.copy()
            yval_shap_std = yval_shap_std_new.copy()
            ## checking whether std and std_epis are of the same shape
            assert yval_pred_std.shape==yval_pred_std_epis.shape
            print(yval_shap_mean.shape[0], yval_pred_std.shape[0])
            assert yval_pred_std.shape[0]==yval_shap_mean.shape[0]
            '''
        '''
        if save_boot:
            #change shape from (num_bs, x_val.shape[0]) to (x_val.shape[0], num_bs)
            pd.DataFrame(np.hstack((np.asarray(concat_output[0]).T, np.asarray(concat_output[1]).T, np.asarray(y[val_idx]).reshape(-1,1), np.asarray(val_idx).reshape(-1,1))), columns=['bag#_%d_mean'%i for i in range(NUM_BS)] + ['bag#_%d_alstd'%i for i in range(NUM_BS)] + ['truth'] + ['index']).to_csv('Bagging_index_low=%d_index_high=%d_time=%s.csv'%(np.asarray(val_idx).min(), np.asarray(val_idx).max(), datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
         '''
        #
        yval_pred_mean_list.extend(yval_pred_mean)
        yval_pred_lower_list.extend(yval_pred_lower)
        yval_pred_upper_list.extend(yval_pred_upper)
        yval_pred_std_list.extend(yval_pred_std)
        yval_pred_std_epis_list.extend(yval_pred_std_epis)
        yval_list.extend(yval_temp)
        yval_shap_mean_list.extend(yval_shap_mean)
        #yval_shap_std_list.extend(yval_shap_std)
        #print(np.shape(yval_pred_mean_list), np.shape(yval_shap_mean_list))
    return yval_pred_mean_list, yval_pred_std_list, yval_pred_lower_list, yval_pred_upper_list, yval_pred_std_epis_list, yval_list, yval_shap_mean_list#, yval_shap_std_list



HPO_FLAG = False


from sklearn.model_selection import cross_val_predict, GridSearchCV, RandomizedSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import BaggingRegressor
from tune_sklearn import TuneSearchCV


def ngb_nobs_cv(estimator, x_df, y, x_noise=None, x_transformer=None, y_transformer=None, save_boot=False, max_samples_best=0.8, weight_bins=10, reversifyfn=None, n_folds=5):
    #yval_pred_mean_list = list()
    #yval_preds_std_list = list()
    #yval_list = list()
    cv_val = custom_cv(y, n_folds=n_folds)
    max_samples_best=0.8
    #
    if HPO_FLAG: 
        # this is not the right way to transform y, causes data leakage. however, sklearn has no inbuilt way of doing this he right way. to do.
        pipeline = estimator#Pipeline([('estimator', estimator)])
        ytr = y.copy()
        #if x_transformer is not None:
        #    pipeline = Pipeline([('x_tr', x_transformer), ('estimator', estimator)])
        if y_transformer is not None:    
            ytr = y_transformer.fit_transform(y.reshape(-1,1)).reshape(-1,)
        #below is for sklearn randomized searchcv
        parameters = {
            'estimator__n_estimators': [500, 750, 1000],
            'estimator__minibatch_frac': [1.0, 0.8, 0.6],
            'estimator__learning_rate': [0.02, 0.04, 0.06],
            'estimator__Dist': [ngb.distns.Normal, ngb.distns.LogNormal],
            'estimator__Base__splitter': ['random', 'best'],
            'estimator__Base__criterion': ['friedman_mse', 'mae', 'mse'],
            'estimator__Base__max_leaf_nodes': [35, 30, 25],
            'estimator__Base': [b1, b2]
        }
        '''
        #below is for bayesian tunesearchcv
        parameters = {
            'estimator__n_estimators': (500, 1000),
            'estimator__minibatch_frac': (0.6, 1.0),
            'estimator__learning_rate': (0.02, 0.06),
            'estimator__Dist': [ngb.distns.Normal, ngb.distns.LogNormal],
            'estimator__Base__splitter': ['random', 'best'],
            'estimator__Base__criterion': ['friedman_mse', 'mae', 'mse'],
            'estimator__Base__max_leaf_nodes': (25, 35),
            'estimator__Base': [b1, b2]
        }
        '''
        parameters_bs = dict()
        for key in parameters:
            key_bs = key.split('__', maxsplit=1)
            key_bs = 'base_estimator__estimator__' + key_bs[-1]
            parameters_bs[key_bs] = parameters[key]
        parameters_bs.update({'max_samples' :  [0.4, 0.6, 0.8, 1.0]})
        if BOOT_FLAG:
            grid_search = RandomSearchCV(estimator=BaggingRegressor(base_estimator=pipeline, n_estimators=NUM_BS, n_jobs=-1), param_distributions=parameters_bs, n_iter=24, n_jobs=1, verbose=1, cv=cv_val)#, use_gpu=True)
            _ = grid_search.fit(x_df.values, ytr)
            grid_search_best_params = grid_search.best_params_
            # separte out just the ngboost parameters from the max_samples parameter for the bootstrap
            max_samples_best = grid_search_best_params['max_samples']
            grid_search_best_params_temp = dict()
            for key in grid_search_best_params:
                split_key = key.split('__',maxsplit=2)
                if len(split_key)==3 and split_key[0]=='base_estimator' and  split_key[1]=='estimator':
                    grid_search_best_params_temp[split_key[2]] = grid_search_best_params[key] 
            grid_search_best_params = grid_search_best_params_temp.copy()
            del grid_search_best_params_temp
        else:
            grid_search = RandomSearchCV(estimator=pipeline, param_distributions=parameters, n_iter=24, n_jobs=-1, verbose=1, cv=cv_val)#, use_gpu=True)
            _ = grid_search.fit(x_df.values, ytr)
            grid_search_best_params = grid_search.best_params_
    else:
        grid_search_best_params = dict()
        grid_search_best_params = estimator.get_params()
        #
    estimator_best = estimator.set_params(**grid_search_best_params)
    #    
    yval_pred_mean_list, yval_pred_std_list, yval_pred_lower_list, yval_pred_upper_list, yval_pred_std_epis_list, yval_list, yval_shap_mean_list, yval_shap_std_list = simple_train_predict(estimator=estimator_best, x_df=x_df, y=y, cv_to_use=cv_val, x_transformer=x_transformer, y_transformer=y_transformer, x_noise=x_noise, save_boot=save_boot, max_samples_best=max_samples_best, weight_bins=weight_bins, reversifyfn=reversifyfn)
    return yval_pred_mean_list, yval_pred_std_list, yval_pred_lower_list, yval_pred_upper_list, yval_pred_std_epis_list, yval_list, estimator_best, max_samples_best, yval_shap_mean_list, yval_shap_std_list




#### 06/14/2020 trying out beta distribution on the output. from here-https://github.com/guyko81/ngboost/blob/beta-distribution/examples/BetaBernoulli/NGBoost%20Beta.ipynb
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from ngboost.distns import Normal, LogNormal#, Beta
from ngboost.scores import LogScore, CRPScore
import ngboost as ngb
import pathos.multiprocessing as mp
# pathos instead of in-built python so we can pickle our 'reversify' functions. see https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
import time


b1 = DecisionTreeRegressor()#splitter='best')
b2 = ExtraTreeRegressor()#splitter='best')

#learner = DecisionTreeRegressor(criterion='friedman_mse', max_depth=5)
learner = DecisionTreeRegressor(
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=31,
        #max_depth=3,
        splitter='best')

NATGRAD_FLAG = True

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
                natural_gradient=NATGRAD_FLAG)
    return base_model


'''
estimator = exported_pipeline()
q = estimator.fit(X.values, label1)#, early_stopping_rounds=2)
pred_dist = estimator.pred_dist(X.values)
'''
###################################################################


#label_list=['Redshift', 'Mass', 'Dust Mass', 'Metallicity', 'SFR']
#label_list=['z', 'Mass', 'Dust', 'Z', 'SFR']
label_list=['Mass', 'Dust', 'Z', 'SFR']
#label_dict = {'Mass':logmass,
#              'Dust Mass':logdustmass,
#              'Metallicity':logmet,
#              'Star Formation Rate':logsfr}
label_rev_func = {'Mass': lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=20)),
                  'Dust': lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=20)) - 1,
                  'Z': lambda x: np.float_power(10, np.clip(x, a_min=-1e1, a_max=1e1)),
                  'SFR': lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=1e2)) - 1,
                  #'z': lambda x: np.clip(x, a_min=0, a_max=None)
                  }

label_func = {'Mass': lambda x: np.log10(x),
              'Dust': lambda x: np.log10(x+1),
              'Z': lambda x: np.log10(x),
              'SFR': lambda x: np.log10(x+1),
              #'z': lambda x: x
              }


############################################################3


EPIS_FLAG = BOOT_FLAG = True
CAL_COMB_FLAG = False
CALIBRATE_FLAG = False
CHAIN_FLAG = True

zfac = 1
x_noise = 0.5 #* np.ones((1, X.shape[1]))
#ngb = NGBoost(Base=default_tree_learner, Dist=Normal, Score=MLE, learning_rate=0.04, n_estimators=600, natural_gradient=True, verbose=False)

n_folds_test = 5
n_folds_val = 5
#10,3 results in train, val, test split of 60,30,10
run_num_test = 0

# x_transformer = mms() gives retarded results.
x_transformer = None # don't modify x_transformer at all, it'll break the addition of x_noise
y_transformer = [rs(), mms()]


test_data = None#['simba']
train_data = ['simba']#, 'eagle', 'tng']
X_simba, y_simba = get_data(['simba'])
sizeofsimba = X_simba.shape[0]

#CV_FOR_TEST = False

timestr = time.strftime("%Y%m%d")#-%H%M%S")
ytest_filename = 'test_metrics_' + timestr + '.csv'
val_filename = 'validation_metrics_' + timestr + '.csv'

ytest_cal_filename = 'test_cal_metrics_' + timestr + '.csv'

ytestattrib_filename = 'test_shap_' + timestr + '.pkl'
valattrib_filename = 'validation_shap_' + timestr + '.pkl'

ytestattrib_std_filename = 'test_shap_std_' + timestr + '.pkl'
valattrib_std_filename = 'validation_shap_std_' + timestr + '.pkl'

trained_model_filename = 'trained_ngboost_' + timestr + '.pkl'

num_nearest_nbrs = 200
distpower = 0
OPTIMIZE_FLAG = False

##########################################
##########################################
time_start = time.time()
for x_noise in [1/5, 1/10, 1/20]:
    for label_str in [0,1,2,3]:
        # calibrated
        y_test_preds_cal_lower = list()
        y_test_preds_cal_upper = list()
        y_test_preds_cal_median = list()
        #### uncalibrated ####
        y_tests = list()
        y_test_preds_std_epis = list()
        y_test_preds_std = list()
        y_test_preds_mean = list()
        y_test_preds_lower = list()
        y_test_preds_upper = list()
        #### shap ####
        y_test_shaps_mean = list()
        y_test_shaps_std = list()
        #### input fluxes ###
        x_trains = list()
        x_tests = list()
        #### Calibration coeffs ###
        best_b = list()
        best_w = list()
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
                #if CV_FOR_TEST:
                #    test_train_idxs, test_idxs = zip(*custom_cv(label1_test[:,-1], n_folds=n_folds_test)[0:])
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
        #n_folds_test_inuse = np.shape(train_val_idxs)[0]
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
                #y_test_shap_std = np.array(y_test_shap_std)
                #############################################################
                x_df[label_list[label_str_iter]] = np.log10(1 + np.append(reversify_func(label1[train_val_idx, label_str_iter]), y_test_pred_mean))
                x_noise_arr = np.append(x_noise_arr, 0.)
            #
            pd.DataFrame(np.hstack((y_test, y_test_pred_mean, y_test_pred_lower, y_test_pred_upper, y_test_pred_std_epis)), columns=['true', 'pred_mean', 'pred_lower', 'pred_upper', 'pred_epis_std']).to_csv(ytest_filename_thisfold)
            #
            pd.DataFrame(y_test_shap_mean).to_pickle(ytestattrib_filename_thisfold)
            #pd.DataFrame(y_test_shap_std).to_pickle(ytestattrib_std_filename_thisfold)
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
            #y_test_shaps_std.extend(y_test_shap_std)
            x_trains.extend(x_df.loc[train_val_idx,list(x_df)[:-1]].copy().values)
            x_tests.extend(x_df.loc[test_idx,list(x_df)[:-1]].copy().values)
            #### Calibration #################################
            if CALIBRATE_FLAG:
                #'''
                print('beginning calibration')
                val_filename_thisfold = val_filename.split('.csv')[0] + '_' + label_list[label_str] + str(n_fold_test_inuse+1) + 'of' + str(n_folds_test) + '.csv'
                ytest_cal_filename_thisfold = ytest_cal_filename.split('.csv')[0] + '_' + label_list[label_str] + str(n_fold_test_inuse+1) + 'of' + str(n_folds_test) + '.csv'
                #
                valattrib_filename_thisfold = valattrib_filename.split('.pkl')[0] + '_' + label_list[label_str] + str(n_fold_test_inuse+1) + 'of' + str(n_folds_test) + '.pkl'
                valattrib_std_filename_thisfold = valattrib_std_filename.split('.pkl')[0] + '_' + label_list[label_str] + str(n_fold_test_inuse+1) + 'of' + str(n_folds_test) + '.pkl'
                #
                y_val_preds_mean, y_val_preds_std, y_val_preds_lower, y_val_preds_upper, y_val_preds_std_epis, y_val, final_model, max_samples_best, y_val_shap_mean, y_val_shap_std = ngb_nobs_cv(estimator=ngb_pipeline(), x_df=np.log10(1+X_train_val), y=y_train_val, x_transformer=x_transformer, y_transformer=y_transformer, x_noise=x_noise, reversifyfn=reversify_func, save_boot=False, weight_bins=weight_bins, n_folds=n_folds_val)
                y_val = np.asarray(y_val).reshape(-1,1)
                y_val_preds_mean = np.asarray(y_val_preds_mean).reshape(-1,1)
                y_val_preds_std = np.asarray(y_val_preds_std).reshape(-1,1)
                y_val_preds_lower = np.asarray(y_val_preds_lower).reshape(-1,1)
                y_val_preds_upper = np.asarray(y_val_preds_upper).reshape(-1,1)  
                y_val_preds_std_epis = np.asarray(y_val_preds_std_epis).reshape(-1,1)
                y_val_shap_mean = np.asarray(y_val_shap_mean)
                y_val_shap_std = np.asarray(y_val_shap_std)
                #
                pd.DataFrame(np.hstack((y_val, y_val_preds_mean, y_val_preds_std, y_val_preds_std_epis)), columns=['val_true', 'val_pred_mean', 'val_std_al', 'val_std_epis']).to_csv(val_filename_thisfold)
                #
                pd.DataFrame(y_test_shap_mean).to_pickle(valattrib_filename_thisfold)
                pd.DataFrame(y_test_shap_std).to_pickle(valattrib_std_filename_thisfold)
                #
                #predicted_cal_df = calibrate(val_filename=val_filename_thisfold, ytest_filename=ytest_filename_thisfold, savefilename=ytest_cal_filename_thisfold, verbose=1, valattrib_filename=valattrib_filename_thisfold, ytestattrib_filename=ytestattrib_filename_thisfold, numnn=num_nearest_nbrs, distpower=distpower, optimize_flag=OPTIMIZE_FLAG)
                predicted_cal_df = calibrate(val_filename=val_filename_thisfold, ytest_filename=ytest_filename_thisfold, savefilename=ytest_cal_filename_thisfold, verbose=1, valattrib_filename=None, ytestattrib_filename=ytestattrib_filename_thisfold, numnn=num_nearest_nbrs, distpower=distpower, optimize_flag=OPTIMIZE_FLAG)
                y_test_preds_cal_median.extend(predicted_cal_df['pred_mean_cal'].values)
                y_test_preds_cal_lower.extend(predicted_cal_df['pred_lower_cal'].values)
                y_test_preds_cal_upper.extend(predicted_cal_df['pred_upper_cal'].values)
                '''
                b_to_use = result.item()#result[0]
                w_to_use = 1#result[1]
                y_test_pred_upper_cal = inv_cdf(b=b_to_use, w=w_to_use, p=0.841, mean_val=y_val_preds_mean, std_val=y_val_preds_std, std_epis_val=y_val_preds_std_epis, orig_val=y_val, mean_test=y_test_pred_mean, std_test=y_test_pred_std, std_epis_test=y_test_pred_std_epis)[int(CAL_COMB_FLAG)]
                y_test_pred_lower_cal = inv_cdf(b=b_to_use, w=w_to_use, p=0.159, mean_val=y_val_preds_mean, std_val=y_val_preds_std, std_epis_val=y_val_preds_std_epis, orig_val=y_val, mean_test=y_test_pred_mean, std_test=y_test_pred_std, std_epis_test=y_test_pred_std_epis)[int(CAL_COMB_FLAG)]
                y_test_pred_median_cal = inv_cdf(b=b_to_use, w=w_to_use, p=0.500, mean_val=y_val_preds_mean, std_val=y_val_preds_std, std_epis_val=y_val_preds_std_epis, orig_val=y_val, mean_test=y_test_pred_mean, std_test=y_test_pred_std, std_epis_test=y_test_pred_std_epis)[int(CAL_COMB_FLAG)]
                #appending results of individual folds to lists:
                y_test_preds_cal_median.extend(y_test_pred_median_cal)
                y_test_preds_cal_lower.extend(y_test_pred_lower_cal)
                y_test_preds_cal_upper.extend(y_test_pred_upper_cal)
                '''
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
        #       
        if CALIBRATE_FLAG:
            y_test_preds_cal_median = np.asarray(y_test_preds_cal_median).reshape(-1,1)
            y_test_preds_cal_lower = np.asarray(y_test_preds_cal_lower).reshape(-1,1)
            y_test_preds_cal_upper = np.asarray(y_test_preds_cal_upper).reshape(-1,1)
        ######################
        ### saving results ###
        a = pd.DataFrame(np.hstack((y_tests, y_test_preds_mean, y_test_preds_lower, y_test_preds_upper, y_test_preds_std_epis)), columns=['true', 'pred_mean', 'pred_lower', 'pred_upper', 'pred_std_epis'])
        a.to_csv('uncal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv')
        b = pd.DataFrame(y_test_shaps_mean, columns=list(x_df)[:-1])
        b.to_csv('shapmea_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv')
        #c = pd.DataFrame(y_test_shaps_std, columns=list(x_df)[:-1])
        #c.to_csv('shapstd_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv')
        xtrn = pd.DataFrame(10**(x_trains)-1, columns=list(x_df)[:-1])
        #xtrn.drop_duplicates(inplace=True,  ignore_index=True)
        xtrn.to_csv('xtrain_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv')
        xtst = pd.DataFrame(10**(x_tests)-1, columns=list(x_df)[:-1])
        #xtst.drop_duplicates(inplace=True, ignore_index=True)
        xtst.to_csv('xtest_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv')
        print('run time = %.1f minutes'%((time.time() - time_start)/60))
        if CALIBRATE_FLAG:
            d = pd.DataFrame(np.hstack((y_tests, y_test_preds_cal_median, y_test_preds_cal_lower, y_test_preds_cal_upper, y_test_preds_std_epis)), columns=['true', 'pred_mean_cal', 'pred_lower_cal', 'pred_upper_cal', 'pred_std_epis'])
            d.to_csv('cal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_NATGRADFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(NATGRAD_FLAG), str(CHAIN_FLAG))+timestr+'.csv')
        #'''
############################################################
############################################################



# SNR 5
snr = 5
### get Prospector results from Sidney's files #############\
simba_prosp = pd.read_pickle('simba_snr%d.pkl'%snr)
eagle_prosp = pd.read_pickle('eagle_snr%d.pkl'%snr)
tng_prosp = pd.read_pickle('tng_snr%d.pkl'%snr)
simba_prosp_z2 = pd.read_pickle('simba_snr%d_z2.pkl'%snr)
simba_prosp['redshift']=0.
eagle_prosp['redshift']=0.
tng_prosp['redshift']=0.
simba_prosp_z2['redshift']=2.
combined_prosp = pd.concat((simba_prosp, eagle_prosp, tng_prosp, simba_prosp_z2), axis=0).reset_index(drop=True)



############################################################

def plot(errorbars):
    plt.close('all')
    plt.xticks(np.arange(4), [r'Delayed $\tau$', 'Non\nparametric', r'$\tau$ + Burst','Constant'], fontsize=25) 
    plt.bar(0, len(maskt)/len(tau_imass), width=1, color='darkblue')
    plt.bar(1, len(maskd)/len(dir_imass), width=1, color='darkorange', alpha=0.6)
    plt.bar(2, len(maskb)/len(burst_imass), width=1, color='deepskyblue')
    plt.bar(3, len(maskc)/len(const_imass), width=1, color='darkseagreen')
    plt.ylabel('Fraction of M$_{\mathrm{true}}$\n within 1$\sigma$ of M$_{\mathrm{model}}$')

from matplotlib import patches as mpatches
def kdeplot():
plt.close('all')
color_kde=['Oranges_r','Blues_r','Greens_r','Reds_r']
par_alpha = {'alpha': 0.8}
sns.kdeplot(data=np.log10(1+y_test).reshape(-1,), data2=np.log10(1+y_test_pred_mean).reshape(-1,), shade=True, shade_lowest=False, linestyles="--", cmap=color_kde[0], kernel='biw', zorder=3, **par_alpha)
label_patch = patches.Patch(color=sns.color_palette(color_kde[0])[2],label="Test")
plt.legend(handles=label_patches, frameon=False, loc=0, prop={'family':'Georgia', 'size':20})

plt.close('all')
rows, cols = 1, 3
#length_x_axis = 30
#length_y_axis = 10
#fig_height = 5.
#height = length_y_axis * rows
#width = length_x_axis  * cols
#plot_aspect_ratio= float(width)/float(height)
fig = plt.figure(figsize=(60,20))
gs = matplotlib.gridspec.GridSpec(nrows=rows, ncols=cols, width_ratios=[1,1,1], figure=fig)#, wspace=0)#, hspace=0)

ax00 = plt.subplot(gs[:,0], aspect='equal')
ax01 = plt.subplot(gs[:,1])
#ax11 = plt.subplot(gs[1,1])
ax02 = plt.subplot(gs[:,2], aspect='equal')

"""
# histogram
_ = sns.distplot(np.log10(lower_file['true_stellar_mass']), ax=ax01, axlabel=r'$\log$ (M$_{\rm true}$ / M$_{\odot}$)')
ax01.set_title('Histogram')
plt.show()
"""

sns.jointplot(data=gilda_file, x='true', y='pred_mean_cal', kind='scatter', ax=ax00)

ax00.plot(gilda_file['true'], gilda_file['true'], 'k--')
ax00.scatter(lower_file['true_stellar_mass'], lower_file['est_stellar_mass_50'], label='Non-Parametric', color='darkorange', alpha=0.5, marker='x')
ax00.scatter(gilda_file['true'], gilda_file['pred_mean_cal'], label='mirkwood', color='deepskyblue')
ax00.set_xscale('log', basex=10, nonposx='clip')
ax00.set_yscale('log', basey=10, nonposy='clip') 
ax00.set_ylim(bottom=ax00.get_xlim()[0], top=ax00.get_xlim()[1])
ax00.set_xlabel(r'$\log$ (M$_{\rm true}$ / M$_{\odot}$)', ha='center')
ax00.set_ylabel(r'$\log$ (M$_{\rm model}$ / M$_{\odot}$)', va='center', rotation='vertical')
"""
label_patch1 = Line2D([0], [0], marker='o', color='w', label='Val1 = 1 \nVal2 = 3 \nVal3 = 5',  markerfacecolor='r', markersize=10)
label_patch2 = Line2D([0], [0], marker='*', color='w', label='Val1 = 1 \nVal2 = 3 \nVal3 = 5',  markerfacecolor='b', markersize=10)
legend_elements = [label_patch1,label_patch2]
plt.legend(handles=legend_elements, frameon=False, loc=0, prop={'family':'Georgia', 'size':20})
"""
## plotting shap values
### horizontal plotting as opposed the default vertical plotting
import colors
import matplotlib.cm as cm
df_shap = pd.read_csv(shapmean_filename, index_col=0)
sort = False
q = shap.summary_plot(df_shap.values, df_shap, max_display=40, sort=sort)
for i,j,k,l,m in zip(*q):
    _ = ax02.scatter(j, i, cmap=colors.red_blue, vmin=k, vmax=l, s=16, c=m, alpha=1, zorder=3, rasterized=len(i) > 500)#,linewidth=2

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
cb = fig.colorbar(m, ticks=[0, 1], aspect=1000, ax=ax02)
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
feature_names = df_shap.columns
num_features = len(feature_names)
max_display = min(len(feature_names), max_display)
#
if sort:
    feature_order = np.argsort(np.sum(np.abs(df_shap.values), axis=0))
    feature_order = feature_order[-max_display:]
else:
    feature_order = np.flip(np.arange(max_display), 0)

ax02.xaxis.set_ticks_position('bottom')
ax02.yaxis.set_ticks_position('none')
ax02.spines['right'].set_visible(False)
ax02.spines['top'].set_visible(False)
ax02.spines['left'].set_visible(False)
ax02.tick_params(color=axis_color, labelcolor=axis_color)
#
# flipping x and y, and adding 'rotation'
ax02.set_xticks(range(len(feature_order)), [feature_names[i] for i in feature_order])#, rotation=90)#, fontsize=15
ax02.tick_params('x', length=20, width=0.5, which='major')
ax02.tick_params('y', labelsize=15)
ax02.set_xlim(-1, len(feature_order))
ax02.set_ylabel(labels['VALUE'], fontsize=25)
ax02.set_xlabel(r'Wavelength ($\mu$m)',fontsize=24)
ax02.invert_xaxis()







plt.show()


ax01.plot(y_test, y_test, 'k--')
ax01.scatter(y_test, y_test_pred_mean, color='deepskyblue')
ax01.scatter(y_test, y_test_pred_mean, color='darkorange', alpha=0.5)
ax01.set_xscale('log', basex=10, nonposx='clip')
ax01.set_yscale('log', basey=10, nonposy='clip') 
ax01.yaxis.tick_right()

ax00.set_ylim(bottom=ax00.get_xlim()[0], top=ax00.get_xlim()[1])
ax01.set_ylim(bottom=ax01.get_xlim()[0], top=ax01.get_xlim()[1])

fig.text(0.51, 0.04, r'$\log$ (M$_{\rm true}$ / M$_{\odot}$)', ha='center')
fig.text(0.07, 0.5, r'$\log$ (M$_{\rm model}$ / M$_{\odot}$)', va='center', rotation='vertical')
plt.legend(frameon=False, loc=0, prop={'family':'Georgia', 'size':20})
plt.show()


label_patch = patches.Patch(color=sns.color_palette(color_kde[0])[2], label="Test")
plt.legend(handles=label_patch, frameon=False, loc=0, prop={'family':'Georgia', 'size':20})


########################

plt.close('all')
plt.plot(y_test, y_test_pred_mean, 'o', c="r")#, label='a')
plt.plot(y_test, y_test_pred_mean ,'*', c="b")#, label='b')
label_patch1 = Line2D([0], [0], marker='o', color='w', label='Val1 = 1 \nVal2 = 3 \nVal3 = 5',  markerfacecolor='r', markersize=10)
label_patch2 = Line2D([0], [0], marker='*', color='w', label='Val1 = 1 \nVal2 = 3 \nVal3 = 5',  markerfacecolor='b', markersize=10)
legend_elements = [label_patch1,label_patch2]
leg = plt.legend(handles=legend_elements, frameon=False, loc=0, borderpad=1, ncol=2, prop={'family':'Georgia', 'size':20})

# To get labels below marker. Play around with the value to see what fits right 
for txt in leg.get_texts():
    txt.set_ha("left") # horizontal alignment of text item
    txt.set_va("top") # horizontal alignment of text item
    txt.set_x(-210) # x-position
    txt.set_y(60) # y-position

plt.show()




########################




columns=['Average Coverage Area', 'Prediction Interval Normalized Average Width', 'Interval Sharpness']
models = ['Non-Parametric', 'mirkwood', 'mirkwood Calibrated']
bar_plot_axes = []
for ax in bar_plot_axes:
    #df = pd.read_csv(s, index_col=0, delimiter=' ', skipinitialspace=True)
    bar_plot_df = pd.DataFrame(columns=['ace', 'pinaw', 'is'])
    for i, dataframe in enumerate(dataframes):
        q = pd.read_csv(dataframe, index_col=0)
        y_test, y_test_pred_mean, y_test_pred_lower, y_test_pred_upper = q.loc[:,0].values, q.loc[:,1].values, q.loc[:,2].values, q.loc[:,3].values
        label_ace = ace(y_test, (y_test_pred_mean, y_test_pred_lower, y_test_pred_upper))
        label_pinaw = pinaw(y_test, (y_test_pred_mean, y_test_pred_lower, y_test_pred_upper))
        label_is = interval_sharpness(y_test, (y_test_pred_mean, y_test_pred_lower, y_test_pred_upper))
        bar_plot_df.loc[i] = [label_ace, label_pinaw, label_is]
        bar_plot_df.rename(index={i:models[i]}, inplace=True)
    #
    width = 0.4
    ax2 = ax.twinx()
    bar_plot_df.amount.plot(kind='bar', color='red', ax=ax, width=width, position=1)
    bar_plot_df.price.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)
    ax.set_ylabel('ACE + 1$\sigma$ Confidence Interval')
    ax2.set_ylabel('Interval Sharpness')



plt.show()


'''
y_test_shap_mean_new = np.zeros((len(y_test), y_test_shap_mean.shape[1]))
y_test_shap_std_new = np.zeros((len(y_test), y_test_shap_mean.shape[1]))
for i in range(y_test_shap_mean.shape[1]):
    y_test_shap_mean_new[:,i] = np.ma.array(y_test_shap_mean[:,i], mask=mask).compressed().reshape(-1,)
    y_test_shap_std_new[:,i] = np.ma.array(y_test_shap_std[:,i], mask=mask).compressed().reshape(-1,)

y_test_shap_mean = y_test_shap_mean_new.copy()
y_test_shap_std = y_test_shap_std_new.copy()
'''

def plot_func(y_test, y_test_pred_mean, y_test_pred_lower, y_test_pred_upper, y_test_pred_std_epis):
    errorbars_al = np.asarray([y_test_pred_mean-y_test_pred_lower, y_test_pred_upper-y_test_pred_mean]).reshape(2,-1)
    errorbars_total = np.asarray([np.sqrt((y_test_pred_mean-y_test_pred_lower)**2 + y_test_pred_std_epis**2), np.sqrt((y_test_pred_upper-y_test_pred_mean)**2 + y_test_pred_std_epis**2)]).reshape(2,-1)
    plt.close('all')
    _ = plt.figure(figsize=(20, 20))
    plt.plot(y_test, y_test, label='1:1 Line', color='black', alpha=0.2)
    plt.errorbar(y_test, y_test_pred_mean, yerr=errorbars_total, fmt='o', color='blue', ecolor='lightblue', elinewidth=3, capsize=0, label='Epistemic Error')
    plt.errorbar(y_test, y_test_pred_mean, yerr=errorbars_al, fmt='o', color='blue', ecolor='orange', elinewidth=3, capsize=0, label='Aleatoric Error')
    '''
    plt.scatter(y_test_reversified, y_test_pred_mean_reversified, label='Predicted Mean', color='blue')
    plt.fill_between(y_test_reversified, y_test_pred_mean_reversified-2*y_test_pred_std_reversified, y_test_pred_mean_reversified+2*y_test_pred_std_reversified, label='Aleatoric Uncertainty', alpha=0.2, color='gray')
    plt.fill_between(y_test_reversified, y_test_pred_mean_reversified-2*y_test_pred_final_std_reversified, y_test_pred_mean_reversified-2*y_test_pred_std_reversified, label='Epistemic Uncertainty', alpha=0.2, color='orange')
    plt.fill_between(y_test_reversified, y_test_pred_mean_reversified+2*y_test_pred_final_std_reversified, y_test_pred_mean_reversified+2*y_test_pred_std_reversified, alpha=0.4, color='orange')
    '''
    label_nrmse = nrmse(y_test, y_test_pred_mean)
    label_nmae = nmae(y_test, y_test_pred_mean)
    label_mape = mape(y_test, y_test_pred_mean)
    label_bias = bias(y_test, y_test_pred_mean)
    label_ace = ace(y_test, (y_test_pred_mean, y_test_pred_lower, y_test_pred_upper))
    label_pinaw = pinaw(y_test, (y_test_pred_mean, y_test_pred_lower, y_test_pred_upper))
    label_is = interval_sharpness(y_test, (y_test_pred_mean, y_test_pred_lower, y_test_pred_upper))
    textstr = '\n'.join((
    r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse, ),
    r'$\mathrm{NMAE}=%.2f$' % (label_nmae, ),
    #r'$\mathrm{MAPE}=%.2f$' % (label_mape, ),
    r'$\mathrm{Bias}=%.2f$' % (label_bias, ),
    r'$\mathrm{ACE}=%.2f$' % (label_ace, ),
    r'$\mathrm{PINAW}=%.2f$' % (label_pinaw, ),
    r'$\mathrm{IS}=%.2f$' % (label_is, ),
    ))
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    plt.title('%s_SNR=%d'%(label_list[label_str], int(snr)))
    plt.ylabel('Predicted %s'%label_list[label_str])
    ax = plt.gca()
    #ax.set_xscale('log', basex=10, nonposx='clip')
    #ax.set_yscale('log', basey=10, nonposy='clip') 
    ax.set_ylim(bottom=ax.get_xlim()[0], top=ax.get_xlim()[1])
    ax.text(0.90, 0.22, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props)
    plt.legend()
    #plt.savefig('uncal_simba+eagle+tng_%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_NATGRADFLAG=%s_'%(label_list[label_str], int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(NATGRAD_FLAG))+timestr+'.eps', bbox_inches='tight', pad_inches=0.1, dpi=300)
    #ax = plt.gca()
    #ax.set_xscale('log', basex=10, nonposx='clip')
    #ax.set_yscale('log', basey=10, nonposy='clip') 
    #ax.set_ylim(bottom=ax.get_xlim()[0], top=ax.get_xlim()[1])
    #plt.savefig('uncal_simba+eagle+tng_logscale_%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_NATGRADFLAG=%s_'%(label_list[label_str], int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(NATGRAD_FLAG))+timestr+'.eps', bbox_inches='tight', pad_inches=0.1, dpi=300)
    #plt.savefig('uncal_results_tng_%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_'%(label_list[label_str], int(snr), NUM_BS, str(WEIGHT_FLAG))+timestr+'.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    #plt.savefig('uncal_results_tng_%s_SNR=%d__NUMBS=%d_WEIGHTFLAG=%s_'%(label_list[label_str], int(snr), NUM_BS, str(WEIGHT_FLAG))+timestr+'.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()

plot_func(y_test, y_test_pred_mean, y_test_pred_lower, y_test_pred_upper, y_test_pred_std_epis)
plot_func(y_test, y_test_pred_median_cal, y_test_pred_lower_cal, y_test_pred_upper_cal, y_test_pred_std_epis)

#######################################################33
##########################################################


for train_val_idx, test_idx in custom_cv(label1, n_folds=n_folds_test):
    run_num_test += 1 
    X_train_val = X.iloc[train_val_idx].reset_index(drop='index')
    y_train_val = label1[train_val_idx]
    #
    y_val_preds_mean, y_val_preds_std, y_val_preds_std_epis, y_val, final_model, max_samples_best = ngb_nobs_cv(estimator=ngb_pipeline(), x_df=X_train_val, y=y_train_val, x_transformer=x_transformer, y_transformer=y_transformer, x_noise=x_noise, n_folds=n_folds_val)
    y_test_pred_mean, y_test_pred_std, y_test_pred_std_epis, y_test = simple_train_predict(estimator=final_model, x_df=np.log10(X), y=label1, cv_to_use=[(train_val_idx, test_idx)], x_transformer=x_transformer, y_transformer=y_transformer, x_noise=x_noise, save_boot=False, max_samples_best=max_samples_best)
    #
    y_test_preds_mean.extend(y_test_pred_mean)
    y_test_preds_std.extend(y_test_pred_std)
    y_test_preds_std_epis.extend(y_test_pred_std_epis)
    #log uncalibrated results
    pd.DataFrame(np.hstack((y_test_pred_mean.reshape(-1,1), ytest_pred_std.reshape(-1,1), y_test_pred_std_epis.reshape(-1,1), y_test.reshape(-1,1), np.asarray(test_idx).reshape(-1,1))), columns=['pred_mean', 'pred_std_al', 'pred_std_epis', 'truth', 'test_idx']).to_csv('uncalibrated_predictictions_BS=%s_HP=%s_time=%s.csv'%(str(BOOT_FLAG), str(HPO_FLAG), datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    ############################################
    ############################################
    # Start Calibration
    #result = optimize.minimize(crude_shift, [0, 1], args=(y_val_preds_mean, y_val_preds_std, y_val_preds_std_al, y_val_preds_std_epis, y_val, y_test_preds_mean, y_test_preds_std, y_test_preds_std_al, y_test_preds_std_epis, label1[test_idx]), method='Powell')
    #x_to_use = result.x[0]
    #w_to_use = result.x[1]
    result = optimize.fmin_powell(crude_shift, [0], args=(y_val_preds_mean, y_val_preds_std, y_val_preds_std_epis, y_val, y_test_pred_mean, y_test_pred_std, y_test_pred_std_epis, y_test))
    b_to_use = result.item()#result[0]
    w_to_use = 1#result[1]
    #bounds = [(0, 1), (0.1, 10)]
    #result = optimize.differential_evolution(crude_shift, bounds, args=(y_val_preds_mean, y_val_preds_std, y_val_preds_std_al, y_val_preds_std_epis, y_val, y_test_preds_mean, y_test_preds_std, y_test_preds_std_al, y_test_preds_std_epis, label1[test_idx]))
    #b_to_use = result.x[0]
    #w_to_use = result.x[1]
    best_b.append(b_to_use)
    best_w.append(w_to_use)
    # epistemic + aleatoric
    y_test_pred_upper_iter = inv_cdf(b=b_to_use, w=w_to_use, p=0.841, mean_val=y_val_preds_mean, std_val=y_val_preds_std, std_epis_val=y_val_preds_std_epis, orig_val=y_val, mean_test=y_test_pred_mean, std_test=y_test_pred_std, std_epis_test=y_test_pred_std_epis)[int(CAL_COMB_FLAG)]
    y_test_pred_lower_iter = inv_cdf(b=b_to_use, w=w_to_use, p=0.159, mean_val=y_val_preds_mean, std_val=y_val_preds_std, std_epis_val=y_val_preds_std_epis, orig_val=y_val, mean_test=y_test_pred_mean, std_test=y_test_pred_std, std_epis_test=y_test_pred_std_epis)[int(CAL_COMB_FLAG)]
    y_test_pred_median_iter = inv_cdf(b=b_to_use, w=w_to_use, p=0.500, mean_val=y_val_preds_mean, std_val=y_val_preds_std, std_epis_val=y_val_preds_std_epis, orig_val=y_val, mean_test=y_test_pred_mean, std_test=y_test_pred_std, std_epis_test=y_test_pred_std_epis)[int(CAL_COMB_FLAG)]
    # alea + epis
    # calibrated upper, lower, median
    y_test_preds_upper.extend(y_test_pred_upper_iter)
    y_test_preds_lower.extend(y_test_pred_lower_iter)
    y_test_preds_median.extend(y_test_pred_median_iter)
    #log calibrated results
    pd.DataFrame(np.hstack((y_test_preds_median.reshape(-1,1), y_test_preds_upper.reshape(-1,1), y_test_preds_lower.reshape(-1,1), y_test_pred_std_epis.reshape(-1,1), y_test.reshape(-1,1), np.asarray(test_idx).reshape(-1,1))), columns=['pred_cal_median', 'pred_cal_upper', 'pred_cal_lower', 'pred_std_epis', 'truth', 'test_idx']).to_csv('calibrated_predictictions_BS=%s_HP=%s_COMBCAL=%s_time=%s.csv'%(str(BOOT_FLAG), str(HPO_FLAG), str(CAL_COMB_FLAG), datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    print('test set %d of %d done' %(run_num_test, n_folds_test))
    print('time spent this run = %.0f seconds' %(time.time()-time_start))
    time_start = time.time()







#.841, .159
#al + epis
y_test_preds_upper = np.asarray(y_test_preds_upper)#*15
y_test_preds_lower = np.asarray(y_test_preds_lower)#*15
y_test_preds_median = np.asarray(y_test_preds_median)#*15
# al
y_test_preds_upper_al = np.asarray(y_test_preds_upper_al)#*15
y_test_preds_lower_al = np.asarray(y_test_preds_lower_al)#*15
y_test_preds_median_al = np.asarray(y_test_preds_median_al)#*15
# elis
y_test_preds_upper_epis = np.asarray(y_test_preds_upper_epis)#*15
y_test_preds_lower_epis = np.asarray(y_test_preds_lower_epis)#*15
y_test_preds_median_epis = np.asarray(y_test_preds_median_epis)#*15

#means
#al + epis
y_test_preds_mean = np.asarray(y_test_preds_mean)#*15
y_test_preds_std = np.asarray(y_test_preds_std)#*15
y_test_preds_mean_upper = y_test_preds_mean + y_test_preds_std
y_test_preds_mean_lower = y_test_preds_mean - y_test_preds_std
# al
y_test_preds_std_al = np.asarray(y_test_preds_std_al)#*15
y_test_preds_mean_upper_al = y_test_preds_mean + y_test_preds_std_al
y_test_preds_mean_lower_al = y_test_preds_mean - y_test_preds_std_al
# epis
y_test_preds_std_epis = np.asarray(y_test_preds_std_epis)#*15
y_test_preds_mean_upper_epis = y_test_preds_mean + y_test_preds_std_epis
y_test_preds_mean_lower_epis = y_test_preds_mean - y_test_preds_std_epis


y_test = np.asarray(y_test)#*15


'''
y_test_preds_upper = label1_pt.inverse_transform(y_test_preds_upper.reshape(-1,1)).reshape(-1,)
y_test_preds_lower = label1_pt.inverse_transform(y_test_preds_lower.reshape(-1,1)).reshape(-1,)
y_test_preds_median = label1_pt.inverse_transform(y_test_preds_median.reshape(-1,1)).reshape(-1,)
y_test_preds_mean_upper = label1_pt.inverse_transform((y_test_preds_mean + y_test_preds_var).reshape(-1,1)).reshape(-1,)
y_test_preds_mean_lower = label1_pt.inverse_transform((y_test_preds_mean - y_test_preds_var).reshape(-1,1)).reshape(-1,)
y_test_preds_mean = label1_pt.inverse_transform(y_test_preds_mean.reshape(-1,1)).reshape(-1,)
y_test = label1_pt.inverse_transform(y_test.reshape(-1,1)).reshape(-1,)
'''

y_test_sort_idx = np.argsort(y_test)
y_test = y_test[y_test_sort_idx]
#epis + al
y_test_preds_upper = y_test_preds_upper[y_test_sort_idx]
y_test_preds_lower = y_test_preds_lower[y_test_sort_idx]
y_test_preds_median = y_test_preds_median[y_test_sort_idx]
#al
y_test_preds_upper_al = y_test_preds_upper_al[y_test_sort_idx]
y_test_preds_lower_al = y_test_preds_lower_al[y_test_sort_idx]
y_test_preds_median_al = y_test_preds_median_al[y_test_sort_idx]
#epis
y_test_preds_upper_epis = y_test_preds_upper_epis[y_test_sort_idx]
y_test_preds_lower_epis = y_test_preds_lower_epis[y_test_sort_idx]
y_test_preds_median_epis = y_test_preds_median_epis[y_test_sort_idx]

#uncalibrated
#epis + al
y_test_preds_mean = y_test_preds_mean[y_test_sort_idx]
y_test_preds_mean_upper = y_test_preds_mean_upper[y_test_sort_idx]
y_test_preds_mean_lower = y_test_preds_mean_lower[y_test_sort_idx]
#al
y_test_preds_mean_upper_al = y_test_preds_mean_upper_al[y_test_sort_idx]
y_test_preds_mean_lower_al = y_test_preds_mean_lower_al[y_test_sort_idx]
#epis
y_test_preds_mean_upper_epis = y_test_preds_mean_upper_epis[y_test_sort_idx]
y_test_preds_mean_lower_epis = y_test_preds_mean_lower_epis[y_test_sort_idx]


num_to_plt = len(y_test)
#idx = np.random.choice(np.arange(len(y_test)), num_to_plt)
idx=np.arange(num_to_plt)
plt.close()
plt.plot(y_test[idx], y_test[idx], '-r')
plt.plot(y_test[idx], y_test_preds_median[idx], 'ok', alpha=0.4)
plt.fill_between(y_test[idx], y_test_preds_upper[idx], y_test_preds_lower[idx], color='gray', alpha=0.4)
#plt.fill_between(y_test[idx], y_test_preds_mean[idx] + y_test_preds_var[idx], y_test_preds_mean[idx] - y_test_preds_var[idx], color='blue', alpha=0.2)
plt.xlabel('True dustmass')#, EAGLE+SIMBA')
plt.ylabel('Predicted dustmass')#, EAGLE+SIMBA')
plt.title('Trained on SIMBA')
#plt.xscale('log')
#plt.yscale('log')    
plt.show()

bool_upper = y_test<=y_test_preds_upper
bool_lower = y_test>=y_test_preds_lower
(bool_upper*bool_lower).mean()

np.mean(np.abs(y_test-y_test_preds_median))


num_to_plt = len(y_test)
#idx = np.random.choice(np.arange(len(y_test)), num_to_plt)
idx=np.arange(num_to_plt)
plt.close()
plt.plot(y_test[idx], y_test[idx], '-r')
plt.plot(y_test[idx], y_test_preds_mean[idx], 'ok', alpha=0.4)
plt.fill_between(y_test[idx], y_test_preds_mean_upper[idx], y_test_preds_mean_lower[idx], color='gray', alpha=0.4)
#plt.fill_between(y_test[idx], y_test_preds_mean[idx] + y_test_preds_var[idx], y_test_preds_mean[idx] - y_test_preds_var[idx], color='blue', alpha=0.2)
plt.xlabel('True dustmass')#, EAGLE+SIMBA')
plt.ylabel('Predicted dustmass')#, EAGLE+SIMBA')
plt.title('Trained on EAGLE+SIMBA')
#plt.xscale('log')
#plt.yscale('log')    
plt.show()

bool_upper = y_test<=y_test_preds_mean_upper
bool_lower = y_test>=y_test_preds_mean_lower
(bool_upper*bool_lower).mean()

np.mean(np.abs(y_test-y_test_preds_mean))
##########################################################
##########################################################



def crude_shift_plot(x, mean_val, var_val, orig_val, mean_test, var_test, orig_test):
    orig_test = np.asarray(orig_test)
    p_true = np.arange(0,1.001,0.01)
    p_pred = list()
    for p_j in p_true:
        p_pred.append(np.mean(orig_test<inv_cdf(x, p=p_j, mean_val=mean_val, var_val=var_val, mean_test=mean_test, var_test=var_test, orig_val=orig_val)))
    p_pred = np.asarray(p_pred)
    return p_true, p_pred
    #return np.sqrt(mse(p_pred, p_true))


p_true, p_pred = crude_shift_plot(result.x.item(), y_val_preds_mean, y_val_preds_var, y_val, y_test_preds_mean, y_test_preds_var, y_test)



# test Mean Squared Error
test_MSE = mean_squared_error(y_test_preds, y_test)
print('Test MSE', test_MSE)

# test Negative Log Likelihood
test_NLL = -y_test_dists.logpdf(y_test.flatten()).mean()
print('Test NLL', test_NLL)


plt.close()
plt.errorbar(y_test, y_test_preds, y_test_dists.scale, ecolor='red')
plt.plot(y_test, y_test, color='black')
plt.show()

plt.close()
plt.errorbar(y_train, y_train_preds, y_train_dists.scale, ecolor='red', ls='none', marker='s')
plt.plot(y_train, y_train, color='black')
plt.show()


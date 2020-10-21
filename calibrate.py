########## calibration ######################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from ngboost.distns import Normal, LogNormal
from ngboost.scores import LogScore
import ngboost as ngb
import pathos.multiprocessing as mp
import gc
from pynndescent import NNDescent
import pandas as pd
import numpy as np
from metrics import *
from scipy.stats import norm


def calibrate(val_filename, ytest_filename, savefilename=None, verbose=1, valattrib_filename=None, ytestattrib_filename=None, numnn=10, distpower=2, optimize_flag=True):
    val_df = pd.read_csv(val_filename, index_col=0)
    y_test_df = pd.read_csv(ytest_filename, index_col=0)
    indexattrib = None
    y_test_attrib = None
    if valattrib_filename is not None:
        val_attrib = pd.read_pickle(valattrib_filename).values
        y_test_attrib = pd.read_pickle(ytestattrib_filename).values
        #https://github.com/lmcinnes/pynndescent/blob/master/doc/how_to_use_pynndescent.ipynb.
        # these settings give accurate indices, but take time to prepare
        indexattrib = NNDescent(val_attrib, n_neighbors=200, diversify_prob=0.0, pruning_degree_multiplier=3.0, metric='manhattan')
        indexattrib.prepare()
    val_true, val_mean, val_std_al, val_std_epis = val_df['val_true'].values, val_df['val_pred_mean'].values, val_df['val_std_al'].values, val_df['val_std_epis'].values
    y_test, y_test_pred_mean, y_test_pred_std_al, y_test_pred_std_epis = y_test_df['true'].values, y_test_df['pred_mean'].values, 0.5*(y_test_df['pred_upper'].values - y_test_df['pred_lower'].values), y_test_df['pred_epis_std'].values
    #
    #https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    def weighted_quantile(values, quantiles, sample_weight=None, 
                          values_sorted=False, old_style=False):
        """ Very close to numpy.percentile, but supports weights.
        NOTE: quantiles should be in [0, 1]!
        :param values: numpy.array with data
        :param quantiles: array-like with many quantiles needed
        :param sample_weight: array-like of the same length as `array`
        :param values_sorted: bool, if True, then will avoid sorting of
            initial array
        :param old_style: if True, will correct output to be consistent
            with numpy.percentile.
        :return: numpy.array with computed quantiles.
        """
        values = np.array(values)
        quantiles = np.array(quantiles)
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        sample_weight = np.array(sample_weight)
        assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
            'quantiles should be in [0, 1]'
        #
        if not values_sorted:
            sorter = np.argsort(values)
            values = values[sorter]
            sample_weight = sample_weight[sorter]
        #
        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        if old_style:
            # To be convenient with numpy.percentile
            weighted_quantiles -= weighted_quantiles[0]
            weighted_quantiles /= weighted_quantiles[-1]
        else:
            weighted_quantiles /= np.sum(sample_weight)
        return np.interp(quantiles, weighted_quantiles, values)
    #
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
    #
    def inv_cdf(b, w, p, val_mean, val_std_al, val_std_epis, val_orig, test_mean, test_std_al, test_std_epis, val_weights=None):
        if p>1:
            p=p/100
        val_mean = np.asarray(val_mean)
        val_std_al = np.asarray(val_std_al)
        val_std_epis = np.asarray(val_std_epis)
        val_std_total = np.sqrt(val_std_al**2 + w*(val_std_epis)**2)
        val_orig = np.asarray(val_orig)
        test_mean = np.asarray(test_mean)
        test_std_al = np.asarray(test_std_al)
        test_std_epis = np.asarray(test_std_epis)
        test_std_total = np.sqrt(test_std_al**2 + w*(test_std_epis)**2)
        if val_weights is None:
            val_weights = np.ones(len(val_mean))
        val_weights = np.asarray(val_weights)
        #test_cal = (test_mean + b) + np.sqrt(test_std_al**2 + w*(test_std_epis)**2)*np.quantile(((val_mean + b) - val_orig)/np.sqrt(val_std_al**2 + w*(val_std_epis)**2), p)
        #test_cal = (test_mean + b) + np.sqrt(test_std_al**2 + w*(test_std_epis)**2)*weighted_quantile(((val_mean + b) - val_orig)/np.sqrt(val_std_al**2 + w*(val_std_epis)**2), quantiles=p, sample_weight=val_weights)
        test_quantile_p = weighted_quantile(cdf_normdist(y=val_orig, loc=val_mean, scale=val_std_total), quantiles=p, sample_weight=val_weights)
        #print('test_quantile_p for p=%.2f is %.2f'%(p, test_quantile_p))
        test_rv = norm(loc=test_mean, scale=test_std_total)
        test_x = np.linspace(test_rv.ppf(q=0.001), test_rv.ppf(q=0.999), 1000)
        test_cal = weighted_quantile(values=test_x, quantiles=test_quantile_p, sample_weight=None, values_sorted=True)
        return test_cal
    #
    def crude_shift(x, val_mean, val_std_al, val_std_epis, val_orig, val_wt=None):
        val_orig = np.asarray(val_orig)
        p_true = np.arange(0,1.0001,0.001)
        p_pred = list()
        b = x[0]
        w = x[1]
        for p_j in p_true:
            inv_cdf_op = inv_cdf(b, w, p=p_j, val_mean=val_mean, val_std_al=val_std_al, val_std_epis=val_std_epis, val_orig=val_orig, test_mean=val_mean, test_std_al=val_std_al, test_std_epis=val_std_epis, val_weights=val_wt)
            p_pred.append(np.mean(val_orig<inv_cdf_op))
        p_pred = np.asarray(p_pred)
        return np.sqrt(mse(p_pred, p_true))
    #
    def fminpowelloptimize(fun, x0, bounds, val_mean, val_std_al, val_std_epis, val_true, y_test_features_sample=None, indexattrib=None, numnn=10, distpower=0, optimize_flag=True):
        if indexattrib is not None:
            # https://github.com/lmcinnes/pynndescent/blob/master/doc/how_to_use_pynndescent.ipynb
            q = indexattrib.query(y_test_features_sample, k=numnn, epsilon=0.2)
            val_idx = q[0][0]
            val_dist = q[1][0]
            val_wt = 1/(val_dist +1e-6)**distpower
            val_wt = val_wt/np.sum(val_wt)
        else:
            val_idx=np.arange(val_mean.shape[0])
            val_wt = np.ones_like(val_idx)
            val_wt = val_wt/np.sum(val_wt)
        if optimize_flag:
            op = optimize.minimize(fun=fun, x0=x0, bounds=bounds, method='Powell', args=(val_mean[val_idx], val_std_al[val_idx], val_std_epis[val_idx], val_true[val_idx], val_wt))
            b,w = op.x
        else:
            b,w = 0.,1.
        return [b,w], val_idx, val_wt
    def calibratefof(result_sample, val_mean, val_std_al, val_std_epis, val_true, y_test_pred_mean_sample, y_test_pred_std_al_sample, y_test_pred_std_epis_sample):
        b_to_use, w_to_use = result_sample[0]
        val_idx = result_sample[1]
        val_wt = result_sample[2]
        #
        calibrated_upper = inv_cdf(b=b_to_use, w=w_to_use, p=0.841, val_mean=val_mean[val_idx], val_std_al=val_std_al[val_idx], val_std_epis=val_std_epis[val_idx], val_orig=val_true[val_idx], test_mean=y_test_pred_mean_sample, test_std_al=y_test_pred_std_al_sample, test_std_epis=y_test_pred_std_epis_sample, val_weights=val_wt)
        #
        calibrated_lower = inv_cdf(b=b_to_use, w=w_to_use, p=0.159, val_mean=val_mean[val_idx], val_std_al=val_std_al[val_idx], val_std_epis=val_std_epis[val_idx], val_orig=val_true[val_idx], test_mean=y_test_pred_mean_sample, test_std_al=y_test_pred_std_al_sample, test_std_epis=y_test_pred_std_epis_sample, val_weights=val_wt)
        #
        calibrated_median = inv_cdf(b=b_to_use, w=w_to_use, p=0.50, val_mean=val_mean[val_idx], val_std_al=val_std_al[val_idx], val_std_epis=val_std_epis[val_idx], val_orig=val_true[val_idx], test_mean=y_test_pred_mean_sample, test_std_al=y_test_pred_std_al_sample, test_std_epis=y_test_pred_std_epis_sample, val_weights=val_wt)
        return calibrated_median, calibrated_lower, calibrated_upper, b_to_use, w_to_use
    #
    print('CRUDE calibration')
    calibrated_median_list = list()
    calibrated_lower_list = list()
    calibrated_upper_list = list()
    calibrated_lower_total_list = list()
    calibrated_upper_total_list = list()
    bestb_list, bestw_list = list(), list()
    ytestidxtouse = np.arange(np.shape(y_test)[0])
    #ytestidxtouse = np.arange(-1,-481,-1)
    #ytestidxtouse = np.arange(480)
    with mp.Pool() as p:
        if valattrib_filename is not None:
            result = p.starmap(fminpowelloptimize ,[(crude_shift, [0., 1.], [(-0.1,0.1), (.99,1.01)], val_mean, val_std_al, val_std_epis, val_true, y_test_attrib[i].reshape(1,-1), indexattrib, numnn, distpower, optimize_flag) for i in ytestidxtouse])
        else:
            result = p.starmap(fminpowelloptimize ,[(crude_shift, [0., 1.], [(0.,0.01), (0.99,1.01)], val_mean, val_std_al, val_std_epis, val_true, None, None, numnn, distpower, optimize_flag) for i in ytestidxtouse])
        _ = gc.collect()
    with mp.Pool() as p:
        result = p.starmap(calibratefof, [(result[i], val_mean, val_std_al, val_std_epis, val_true, y_test_pred_mean[i], y_test_pred_std_al[i], y_test_pred_std_epis[i]) for i in ytestidxtouse])
        _ = gc.collect()
    for i in ytestidxtouse:
        calibrated_median_list.append(result[i][0])
        calibrated_lower_list.append(result[i][1])
        calibrated_upper_list.append(result[i][2])
        calibrated_lower_total_list.append(result[i][1])
        calibrated_upper_total_list.append(result[i][2])
        bestb_list.append(result[i][3])
        bestw_list.append(result[i][4])
    calibrated_median_list = np.asarray(calibrated_median_list)
    calibrated_lower_list = np.asarray(calibrated_lower_list)    
    calibrated_upper_list = np.asarray(calibrated_upper_list)
    calibrated_lower_total_list = np.asarray(calibrated_lower_total_list)    
    calibrated_upper_total_list = np.asarray(calibrated_upper_total_list)
    bestb_list = np.asarray(bestb_list)
    bestw_list = np.asarray(bestw_list)
    #
    predicted_iq_df = pd.DataFrame(np.hstack((y_test[ytestidxtouse].reshape(-1,1), y_test_pred_mean[ytestidxtouse].reshape(-1,1), (y_test_pred_mean[ytestidxtouse]-y_test_pred_std_al[ytestidxtouse]).reshape(-1,1), (y_test_pred_mean[ytestidxtouse]+y_test_pred_std_al[ytestidxtouse]).reshape(-1,1), calibrated_median_list.reshape(-1,1), calibrated_lower_list.reshape(-1,1), calibrated_upper_list.reshape(-1,1), calibrated_lower_total_list.reshape(-1,1), calibrated_upper_total_list.reshape(-1,1), bestb_list.reshape(-1,1), bestw_list.reshape(-1,1))), columns=['true', 'pred_mean', 'pred_lower', 'pred_upper', 'pred_mean_cal', 'pred_lower_cal', 'pred_upper_cal', 'pred_lower_total_cal', 'pred_upper_total_cal', 'bestb', 'bestw'])
    if verbose!=0:
        print('MeanAE=%.3f'%np.mean(np.abs(predicted_iq_df['true']-predicted_iq_df['pred_mean'])))
        print('Calibrated MeanAE=%.3f'%np.mean(np.abs(predicted_iq_df['true']-predicted_iq_df['pred_mean_cal'])))
        print('MedianAE=%.3f'%np.median(np.abs(predicted_iq_df['true']-predicted_iq_df['pred_mean'])))
        print('Calibrated MedianAE=%.3f'%np.median(np.abs(predicted_iq_df['true']-predicted_iq_df['pred_mean_cal'])))
        print('ECE=%.3f'%ace(predicted_iq_df['true'], (predicted_iq_df['pred_mean'], predicted_iq_df['pred_lower'], predicted_iq_df['pred_upper'])))
        print('Calibrated ECE=%.3f'%ace(predicted_iq_df['true'], (predicted_iq_df['pred_mean_cal'], predicted_iq_df['pred_lower_cal'], predicted_iq_df['pred_upper_cal'])))
        print('IS=%.3f'%interval_sharpness(predicted_iq_df['true'], (predicted_iq_df['pred_mean'], predicted_iq_df['pred_lower'], predicted_iq_df['pred_upper'])))
        print('Calibrated IS=%.3f'%interval_sharpness(predicted_iq_df['true'], (predicted_iq_df['pred_mean_cal'], predicted_iq_df['pred_lower_cal'], predicted_iq_df['pred_upper_cal'])))
    if savefilename is not None:
        predicted_iq_df.to_csv(savefilename)
    return predicted_iq_df



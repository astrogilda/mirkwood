
#https://intelpython.github.io/daal4py/sklearn.html
#import daal4py.sklearn
#daal4py.sklearn.patch_sklearn()

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
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

EPS = 1e-6
#import seaborn as sns

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


### EAGLE ###
X_eagle = pd.concat((pd.read_pickle(eagle_seds1), pd.read_pickle(eagle_seds2)), axis=0)
X_eagle_fir = pd.read_pickle(eagle_seds_fir)
y_eagle = pd.concat((pd.read_pickle(eagle_props1), pd.read_pickle(eagle_props2)), axis=0)


common_ids, common_idx0, common_idx1 = np.intersect1d(X_eagle['ID'].values, y_eagle['ID'].values, assume_unique=True, return_indices=True)
y_eagle = y_eagle.iloc[common_idx1,:]
X_eagle = X_eagle.iloc[common_idx0, :]
X_eagle.reset_index(inplace=True)
X_eagle.drop(columns=['index', 'ID'], inplace=True)
X_eagle_fir = X_eagle_fir.iloc[common_idx0, :]
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


#X_eagle = np.log10(X_eagle)

y_eagle.drop(columns=['ID'], inplace=True)
for prop in list(y_eagle):
    q = y_eagle[prop].values #it's a numpy array of singular lists
    w = [q[i][0].item() for i in range(X_eagle.shape[0])]
    y_eagle[prop] = np.asarray(w)   

logmass_eagle = np.log10(y_eagle['stellar_mass'].values)
logsfr_eagle = np.log10(1+y_eagle['sfr'].values)
logmet_eagle = np.log10(1+y_eagle['metallicity']).values
logdustmass_eagle = np.log10(1+y_eagle['dust_mass']).values

logmass_eagle[logmass_eagle<EPS] = 0
logsfr_eagle[logsfr_eagle<EPS] = 0
logmet_eagle[logmet_eagle<EPS] = 0
logdustmass_eagle[logdustmass_eagle<EPS] = 0


#########################################################
### SIMBA ###############################################
X_simba = pd.read_pickle(simba_seds)
X_simba_fir = pd.read_pickle(simba_seds_fir)
y_simba = pd.read_pickle(simba_props)

common_ids, common_idx0, common_idx1 = np.intersect1d(X_simba['ID'].values, y_simba['ID'].values, assume_unique=True, return_indices=True)
y_simba = y_simba.iloc[common_idx1,:]
X_simba = X_simba.iloc[common_idx0, :]
X_simba.reset_index(inplace=True)
X_simba.drop(columns=['index', 'ID'], inplace=True)
X_simba_fir = X_simba_fir.iloc[common_idx0, :]
X_simba_fir.reset_index(inplace=True)
X_simba_fir.drop(columns=['index', 'ID'], inplace=True)

titles = X_simba['Filters'].iloc[0]
titles_fir = X_simba_fir['Filters'].iloc[0]
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
    q = y_simba[prop].values #it's a numpy array of singular lists
    w = [q[i][0].item() for i in range(X_simba.shape[0])]
    y_simba[prop] = np.asarray(w)   


logmass_simba = np.log10(y_simba['stellar_mass'].values)
logsfr_simba = np.log10(1+y_simba['sfr'].values)
logmet_simba = np.log10(1+y_simba['metallicity']).values
logdustmass_simba = np.log10(1+y_simba['dust_mass']).values

logmass_simba[logmass_simba<EPS] = 0
logsfr_simba[logsfr_simba<EPS] = 0
logmet_simba[logmet_simba<EPS] = 0
logdustmass_simba[logdustmass_simba<EPS] = 0


##########################################################
############### combine SIMBA and EAGLE ##################
X = pd.concat((X_simba, X_eagle), axis=0).reset_index().drop('index', axis=1)
X_err = pd.concat((X_err_simba, X_err_eagle), axis=0).reset_index().drop('index', axis=1)
y = pd.concat((y_simba, y_eagle), axis=0).reset_index().drop('index', axis=1)

logmass = np.log10(y['stellar_mass'].values)
logsfr = np.log10(1+y['sfr'].values)
logmet = np.log10(1+y['metallicity']).values
logdustmass = np.log10(1+y['dust_mass']).values

logmass[logmass<EPS] = 0
logsfr[logsfr<EPS] = 0
logmet[logmet<EPS] = 0
logdustmass[logdustmass<EPS] = 0

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

from sedpy.observate import load_filters
filters = load_filters(list(X), directory='./sedpy/data/filters')
filt_mean_wave = dict()
for filt in filters:
    #mean_wave in microns
    #filt_mean_wave[filt.name]= str([round(filt.wave_mean/10000,2), round(filt.effective_width/10000,3)])
    filt_mean_wave[filt.name]= str(round(filt.wave_mean/10000,2))



#label1 = y.values
import statsmodels
#statsmodels.genmod.generalized_linear_model.GLM.estimate_tweedie_power

def custom_cv(y, n_folds=10):
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
from ngboost.distns import Normal#, Beta
from ngboost.scores import LogScore, CRPScore
import ngboost as ngb
from learners import default_tree_learner, default_linear_learner
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




label_list=['logmass', 'logdustmass', 'logmet', 'logsfr']
label_dict = {'logmass':logmass,
              'logdustmass':logdustmass,
              'logmet':logmet,
              'logsfr':logsfr}
label_rev_func = {'logmass': lambda x: 10**x,
                  'logdustmass': lambda x: 10**x - 1,
                  'logmet': lambda x: 10**x - 1,
                  'logsfr': lambda x: 10**x - 1}

label_true_pd = pd.DataFrame()
label_pred_pd = pd.DataFrame()

for label_str in range(len(label_list)):
    print('iteration %d started' %label_str)
    label1 = label_dict[label_list[label_str]]
    if label_str==0:
        train_idx, test_idx = custom_cv(label1, n_folds=10)[0]
        x_train, x_test = X.values[train_idx], X.values[test_idx]
        #
        x_train = np.log10(x_train)
        x_test = np.log10(x_test)
        #
        x_test_df = pd.DataFrame(x_test, columns=list(X))#mean_wave)
        x_train_df = pd.DataFrame(x_train, columns=list(X))#mean_wave)
        #
        #x_tr = pt().fit(x_train)
        #x_train = x_tr.transform(x_train)
        #x_test = x_tr.transform(x_test)
    else:
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
    model = model.fit(x_train, y_train)
    #
    #'''
    y_train_plt = y_tr.inverse_transform(y_train.reshape(-1,1)).reshape(-1,)
    y_test_plt = y_tr.inverse_transform(y_test.reshape(-1,1)).reshape(-1,)
    y_train_pred_plt = y_tr.inverse_transform(model.predict(x_train).reshape(-1,1)).reshape(-1,)
    y_test_pred_plt = y_tr.inverse_transform(model.predict(x_test).reshape(-1,1)).reshape(-1,)
    label_plt= label_list[label_str].split('log')[-1]
    reversify_func = label_rev_func[label_list[label_str]]
    y_train_plt = reversify_func(y_train_plt)
    y_test_plt = reversify_func(y_test_plt)
    y_train_pred_plt = reversify_func(y_train_pred_plt)
    y_test_pred_plt = reversify_func(y_test_pred_plt)
    plt.close()
    plt.scatter(y_test_plt, y_test_pred_plt)#,'x')
    plt.plot(y_test_plt, y_test_plt, '-', color='black')
    plt.xscale('log')
    plt.yscale('log')    
    plt.xlabel('True %s'%label_plt)#, EAGLE+SIMBA')
    plt.ylabel('Predicted %s'%label_plt)#, EAGLE+SIMBA')
    plt.title('Trained on EAGLE+SIMBA, z=0')
    plt.savefig(label_list[label_str]+'_scatter_chain_ngb.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
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
    # summary plot
    sort=False
    plt.close()
    q = shap.summary_plot(df_shap_train.values, x_train_df, max_display=40, sort=sort)#, plot_type="bar")
    ###################################################
    ### horizontal plotting as opposed the default vertical plotting
    import colors
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    plt.close()
    fig = plt.figure(figsize=(15,10))
    for i,j,k,l,m in zip(*q):
        _ = plt.scatter(j, i, cmap=colors.red_blue, vmin=k, vmax=l, s=16, c=m, alpha=1, linewidth=0, zorder=3, rasterized=len(i) > 500)
    #
    labels = {
        'MAIN_EFFECT': "SHAP main effect value for\n%s",
        'INTERACTION_VALUE': "SHAP interaction value",
        'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
        'VALUE': "Shapley value (impact on model output)",
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
    feature_names = x_train_df.columns
    num_features = len(feature_names)
    max_display = min(len(feature_names), max_display)
    #
    if sort:
        feature_order = np.argsort(np.sum(np.abs(df_shap_train.values), axis=0))
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
    #
    plt.savefig(label_list[label_str]+'_shap_chain_ngb.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    #plt.margins(0,0)
    #plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #plt.tight_layout()
    #plt.show()
    ##########################################
    ###########################################



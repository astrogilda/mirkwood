

import seaborn as sns, numpy as np, matplotlib.pyplot as plt, pandas as pd

import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams.update({
    "savefig.facecolor": "w",
    "figure.facecolor" : 'w',
    "figure.figsize" : (10,8),
    "text.color": "k",
    "legend.fontsize" : 23,
    "font.size" : 25,
    "axes.edgecolor": "k",
    "axes.labelcolor": "k",
    "axes.linewidth": 4,
    "xtick.color": "k",
    "ytick.color": "k",
    "xtick.labelsize" : 25,
    "ytick.labelsize" : 25,
    "ytick.major.size" : 12,
    "xtick.major.size" : 12,
    "ytick.major.width" : 3,
    "xtick.major.width" : 3,
    "font.family": "STIXGeneral",
    "mathtext.fontset" : "cm"
})

from matplotlib.lines import Line2D # for legend purposes
EPS = 1e-5

plt.style.use('seaborn-bright')
# also including the facecolor argument in plt.savefig. remove for regular images.
timenow = time.strftime("%Y%m%d")

# also including the facecolor argument in plt.savefig. remove for regular images.
figsize = (8,8)

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


def update_filenames(label_str):
    uncal_filename = 'uncal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
    shapmean_filename = 'shapmea_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
    xtrn_filename = 'xtrain_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
    xtst_filename = 'xtest_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
    mean_plot_func = label_func[label_list[label_str]]
    gilda_only_flag = True if label_str==0 else False
    lower_mean_only_flag = True if label_str==2 else False
    return uncal_filename, shapmean_filename, xtrn_filename, xtst_filename, mean_plot_func, gilda_only_flag, lower_mean_only_flag


#'VALUE': "SHAP value\n(Impact on model output)"
labels = {
    'MAIN_EFFECT': "SHAP main effect value for\n%s",
    'INTERACTION_VALUE': "SHAP interaction value",
    'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
    'VALUE': "SHAP value",
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


def plot_shapsubplot(df_shap_test, df_features_test, axshap, figshap, ticksevenorodd='even'):
    axshap.set_facecolor('white')
    sort=False
    max_display = 40
    feature_names = df_features_test.columns
    features = df_features_test.values
    shap_values = df_shap_test.values
    #shap_values = np.flip(shap_values)
    num_features = len(feature_names)
    if sort:
        feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-max_display:]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)
    #ax.axvline(x=0, color="#999999", zorder=-1)
    for pos, i in enumerate(feature_order):
        row_height = 0.4
        alpha = 1
        axshap.axvline(x=pos, color="#cccccc", lw=1, dashes=(1, 5), zorder=-1)
        shaps = shap_values[:, i]
        values = features[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        values = values[inds]
        shaps = shaps[inds]
        colored_feature = True
        values = np.array(values, dtype=np.float64)  # make sure this can be numeric
        N = len(shaps)
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))
        # trim the color range, but prevent the color range from collapsing
        vmin = np.nanpercentile(values, 5)
        vmax = np.nanpercentile(values, 95)
        if vmin == vmax:
            vmin = np.nanpercentile(values, 1)
            vmax = np.nanpercentile(values, 99)
            if vmin == vmax:
                vmin = np.min(values)
                vmax = np.max(values)
        if vmin > vmax: # fixes rare numerical precision issues
            vmin = vmax
        # plot the nan values in the interaction feature as grey
        nan_mask = np.isnan(values)
        axshap.scatter(pos + ys[nan_mask], shaps[nan_mask], color="#777777", vmin=vmin, vmax=vmax, s=16, alpha=alpha, linewidth=0, zorder=3, rasterized=len(shaps) > 500)
        # plot the non-nan values colored by the trimmed feature value
        cvals = values[np.invert(nan_mask)].astype(np.float64)
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
        cvals[cvals_imp > vmax] = vmax
        cvals[cvals_imp < vmin] = vmin
        axshap.scatter(pos + ys[np.invert(nan_mask)], shaps[np.invert(nan_mask)], 
                   cmap=colors.red_blue, vmin=vmin, vmax=vmax, s=16,
                   c=cvals, alpha=alpha, linewidth=0,
                   zorder=3, rasterized=len(shaps) > 500)
    color_bar_label=''#labels["FEATURE_VALUE"]
    m = cm.ScalarMappable(cmap=colors.red_blue)
    m.set_array([0, 1])
    cb = plt.colorbar(m, ticks=[0, 1], ax=axshap)#, aspect=1000)
    cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
    cb.set_label(color_bar_label, size=25, labelpad=-25)
    cb.ax.tick_params(labelsize=25, length=0)
    #cb.set_alpha(1)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(figshap.dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.9) * 20)
    #
    axis_color="#333333"
    max_display = min(len(feature_names), max_display)
    axshap.xaxis.set_ticks_position('bottom')
    axshap.yaxis.set_ticks_position('none')
    axshap.spines['right'].set_visible(False)
    axshap.spines['top'].set_visible(False)
    axshap.spines['left'].set_visible(False)
    axshap.tick_params(color=axis_color, labelcolor=axis_color)
    _ = axshap.set_xticks(range(len(feature_order)))
    _ = axshap.set_xticklabels([feature_names[i] for i in feature_order], fontsize=25, rotation=90)
    axshap.grid(False)
    if ticksevenorodd!='both':
        for tick in axshap.xaxis.get_major_ticks()[int(ticksevenorodd=='even')::2]:
            tick.set_visible(False)#set_pad(100)
    axshap.tick_params('x', length=20, width=0.5, which='major')
    axshap.tick_params('y', labelsize=25)
    _ = axshap.set_xlim(left=-1, right=len(feature_order))
    _ = axshap.set_ylabel(labels['VALUE'], fontsize=25)
    _ = axshap.set_xlabel(r'Wavelength ($\mu$m)',fontsize=25)
    _ = axshap.invert_xaxis()
    plt.tight_layout()
    plt.savefig(figname+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
    plt.savefig(figname+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
    return

#https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas
def plot_clustered_stacked(dfall, figname, labels=None, title="multiple stacked bar plot",  H="x/O.", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""
    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    plt.close('all')
    fig = plt.figure(figsize=(16, 5))
    axe = plt.subplot(111)
    #mycolors = []
    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=1,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      color=['orange', 'blue'],
                      alpha=1,
                      **kwargs)  # make bar plots
    #
    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))
    #
    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title, fontsize=25)
    #
    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))
    #
    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    plt.tight_layout()
    plt.savefig(figname+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
    plt.savefig(figname+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
    return axe

# create fake dataframes
df1 = pd.DataFrame(np.random.rand(5, 2),
                   index=["RMSE", "MAE", "BE", "ACE", "IS"],
                   columns=["Mirkwood", "Traditional"])
df2 = pd.DataFrame(np.random.rand(5, 2),
                   index=["RMSE", "MAE", "BE", "ACE", "IS"],
                   columns=["Mirkwood", "Traditional"])
df3 = pd.DataFrame(np.random.rand(5, 2),
                   index=["RMSE", "MAE", "BE", "ACE", "IS"],
                   columns=["Mirkwood", "Traditional"])

# Then, just call :
plot_clustered_stacked(dfall=[df1, df2, df3], figname='Barplot_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow, labels=["SNR=5", "SNR=10", "SNR=20"], title='Mass')
plt.show()

snr=5
timestr = '20201101'
###########################################################################



### for redshift ####
label_str = 0
uncal_filename, shapmean_filename, xtrn_filename, xtst_filename, mean_plot_func, gilda_only_flag, lower_mean_only_flag = update_filenames(label_str)
gilda_file = pd.read_csv(uncal_filename, index_col=0)
gilda_file = gilda_file.apply(pd.to_numeric)

titletrue=r'$z_{\mathrm{true}}$'
titlemodel=r'$z_{\mathrm{model}}$'
title_den=None

### for mass ##########
label_str = 1
uncal_filename, shapmean_filename, xtrn_filename, xtst_filename, mean_plot_func, gilda_only_flag, lower_mean_only_flag = update_filenames(label_str)

gilda_file = pd.read_csv(uncal_filename, index_col=0)#.drop(columns=['pred_std_epis'])
lower_file = combined_prosp[['true_stellar_mass', 'est_stellar_mass_50', 'est_stellar_mass_16', 'est_stellar_mass_84']].copy()
lower_file.rename(columns={"true_stellar_mass": "true", "est_stellar_mass_50": "pred_mean", "est_stellar_mass_16": "pred_lower", "est_stellar_mass_84": "pred_upper"}, inplace=True)
idx0 = np.where(lower_file.true.values>=gilda_file.true.values.min())[0]
idx1 = np.where(lower_file.true.values<=gilda_file.true.values.max())[0]
idx_common = np.intersect1d(idx0, idx1)
lower_file = lower_file.loc[idx_common].copy()
lower_file.reset_index(inplace=True, drop=True)
lower_file.pred_mean = 10**lower_file.pred_mean
lower_file.pred_lower = 10**lower_file.pred_lower
lower_file.pred_upper = 10**lower_file.pred_upper
gilda_file = gilda_file.apply(pd.to_numeric)
lower_file = lower_file.apply(pd.to_numeric)

titletrue='M$^{\star}_{\mathrm{true}}$'
titlemodel='M$^{\star}_{\mathrm{model}}$'
title_den='M$_{\odot}$'

### for dust mass ##########
label_str = 2
uncal_filename, shapmean_filename, xtrn_filename, xtst_filename, mean_plot_func, gilda_only_flag, lower_mean_only_flag = update_filenames(label_str)

gilda_file = pd.read_csv(uncal_filename, index_col=0)#.drop(columns=['pred_std_epis'])
lower_file = combined_prosp[['true_dust_mass', 'est_dustmass']].copy()
lower_file.rename(columns={"true_dust_mass": "true", "est_dustmass": "pred_mean"}, inplace=True)
idx0 = np.where(lower_file.true.values>=gilda_file.true.values.min())[0]
idx1 = np.where(lower_file.true.values<=gilda_file.true.values.max())[0]
idx_common = np.intersect1d(idx0, idx1)
lower_file = lower_file.loc[idx_common].copy()
lower_file.reset_index(inplace=True, drop=True)

gilda_file = gilda_file.apply(pd.to_numeric)
lower_file = lower_file.apply(pd.to_numeric)

titletrue='M$_{\mathrm{dust, true}}$'
titlemodel='M$_{\mathrm{dust, model}}$'
title_den='M$_{\odot}$'

### for metallicity #####
label_str = 3
uncal_filename, shapmean_filename, xtrn_filename, xtst_filename, mean_plot_func, gilda_only_flag, lower_mean_only_flag = update_filenames(label_str)
gilda_file = pd.read_csv(uncal_filename, index_col=0)
lower_file = 10**(combined_prosp[['true_log(z/zsol)', 'est_log(z/zsol)_50', 'est_log(z/zsol)_16', 'est_log(z/zsol)_84']].copy())
lower_file.rename(columns={"true_log(z/zsol)": "true", "est_log(z/zsol)_50": "pred_mean", "est_log(z/zsol)_16": "pred_lower", "est_log(z/zsol)_84": "pred_upper"}, inplace=True)
idx0 = np.where(lower_file.loc[:,list(lower_file)[0]].copy().values>=gilda_file['true'].values.min())[0]
idx1 = np.where(lower_file.loc[:,list(lower_file)[0]].copy().values<=gilda_file['true'].values.max())[0]
idx_common = np.intersect1d(idx0, idx1)
lower_file = lower_file.loc[idx_common].copy()
lower_file.reset_index(inplace=True, drop=True)
gilda_file = gilda_file.apply(pd.to_numeric)
lower_file = lower_file.apply(pd.to_numeric)

titletrue='Z$^{\star}_{\mathrm{true}}$'
titlemodel='Z$^{\star}_{\mathrm{model}}$'
title_den='Z$_{\odot}$'

### for SFR ####
label_str = 4
uncal_filename, shapmean_filename, xtrn_filename, xtst_filename, mean_plot_func, gilda_only_flag, lower_mean_only_flag = update_filenames(label_str)
gilda_file = pd.read_csv(uncal_filename, index_col=0)
lower_file = combined_prosp[['true_sfr', 'est_sfr_50', 'est_sfr_16', 'est_sfr_84']].copy()
idx0 = np.where(lower_file['true_sfr'].values>=gilda_file['true'].values.min())[0]
idx1 = np.where(lower_file['true_sfr'].values<=gilda_file['true'].values.max())[0]
idx_common = np.intersect1d(idx0, idx1)
lower_file = lower_file.loc[idx_common].copy()
lower_file.reset_index(inplace=True, drop=True)
lower_file.rename(columns={"true_sfr": "true", "est_sfr_50": "pred_mean", "est_sfr_16": "pred_lower", "est_sfr_84": "pred_upper"}, inplace=True)
gilda_file = gilda_file.apply(pd.to_numeric)
lower_file = lower_file.apply(pd.to_numeric)

titletrue='SFR$_{\mathrm{100, true}}$'
titlemodel='SFR$_{\mathrm{100, model}}$'
title_den=None

############################################################################
######### COMMON CODE #########################################################3

plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
ax.plot(mean_plot_func(gilda_file.true.values), mean_plot_func(gilda_file.true.values), lw=2, color='k', ls='--')
if label_str!=0:
    _=sns.kdeplot(x=mean_plot_func(lower_file.true), y=mean_plot_func(lower_file.pred_mean), cmap="Blues", shade=False, ax=ax, linestyles="-", linewidths=4)#, thresh=0.1, levels=5

_=ax.scatter(x=mean_plot_func(gilda_file.true), y=mean_plot_func(gilda_file.pred_mean), color='orange', alpha=0.5)
#
ax.set_facecolor('white')
ax.grid(True)
_=ax.set_ylim(bottom=ax.get_xlim()[0], top=ax.get_xlim()[1])
if title_den is not None:
    _=ax.set_xlabel(r'$\log$ ('+titletrue +' / '+title_den +')', ha='center', size=25)
    _=ax.set_ylabel(r'$\log$ ('+titlemodel +' / '+title_den +')', va='center', size=25, labelpad=20)
else:
    _=ax.set_xlabel(r'$\log$ ('+titletrue +')', ha='center', size=25)
    _=ax.set_ylabel(r'$\log$ ('+titlemodel +')', va='center', size=25, labelpad=20)

label_nrmse_patch1 = nrmse(gilda_file.true.values, gilda_file.pred_mean.values)
label_nmae_patch1 = nmae(gilda_file.true.values, gilda_file.pred_mean.values)
label_mape_patch1 = mape(gilda_file.true.values, gilda_file.pred_mean.values)
label_bias_patch1 = nbe(gilda_file.true.values, gilda_file.pred_mean.values)
textstr_patch1 = '\n'.join((
r'$\bf{Mirkwood}$',
r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse_patch1, ),
r'$\mathrm{NMAE}=%.2f$' % (label_nmae_patch1, ),
r'$\mathrm{NBE}=%.2f$' % (label_bias_patch1, ),
))
label_patch1 = Line2D([0], [0], marker=None, color='orange', label=textstr_patch1, lw=4, linestyle='dotted')
if label_str!=0:
    label_nrmse_patch2 = nrmse(lower_file.true.values, lower_file.pred_mean.values)
    label_nmae_patch2 = nmae(lower_file.true.values, lower_file.pred_mean.values)
    label_mape_patch2 = mape(lower_file.true.values, lower_file.pred_mean.values)
    label_bias_patch2 = nbe(lower_file.true.values, lower_file.pred_mean.values)
    textstr_patch2 = '\n'.join((
    r'$\bf{Traditional}$',
    r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse_patch2, ),
    r'$\mathrm{NMAE}=%.2f$' % (label_nmae_patch2, ),
    r'$\mathrm{NBE}=%.2f$' % (label_bias_patch2, ),
    ))
    label_patch2 = Line2D([0], [0], marker=None, color='blue', label=textstr_patch2, lw=4, linestyle='solid')

if label_str==0:
    legend_elements = [label_patch1]
else:
    legend_elements = [label_patch1, label_patch2]

L1 = ax.legend(handles=legend_elements[0:1], loc=[0, 0.68], fancybox=True, framealpha=0.7, fontsize=18)
ax.add_artist(L1)
if label_str!=0:
    ax.legend(handles=legend_elements[1:], loc=[.58, 0.05], framealpha=0.7, fontsize=18)

plt.tight_layout()
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()

####### Calibration Plot ################################

u_gilda = cdf_normdist(y=gilda_file.true.values, loc=gilda_file.pred_mean.values, scale=.5*(gilda_file.pred_upper-gilda_file.pred_lower).values)
label_ace_mirk = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))
label_is_mirk = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))

if label_str not in [0,2]:
    u_lower = cdf_normdist(y=lower_file.iloc[:,0].values, loc=lower_file.iloc[:,1].values, scale=.5*(lower_file.iloc[:,3].values-lower_file.iloc[:,2].values))
    label_ace_lower = ace(lower_file.iloc[:,0].values, (lower_file.iloc[:,1].values, lower_file.iloc[:,2].values, lower_file.iloc[:,3].values))
    label_is_lower = interval_sharpness(lower_file.iloc[:,0].values, (lower_file.iloc[:,1].values, lower_file.iloc[:,2].values, lower_file.iloc[:,3].values))

plt.close('all')
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.set_facecolor='white'
ax.grid(True)
ax.plot(np.linspace(0, 1, len(u_gilda)), np.sort(u_gilda), ls='-', color='orange', lw=5, label='Mirkwood')
if label_str not in [0,2]:
    ax.plot(np.linspace(0, 1, len(u_lower)), np.sort(u_lower), ls='-', color='blue', lw=5, label='Traditional')

ax.plot(np.linspace(0, 1, len(u_gilda)), np.linspace(0, 1, len(u_gilda)), lw=4, color='k', ls='--')
ax.set_xlabel(r'Expected Confidence Level', ha='center', size=25)
ax.set_ylabel(r'Observed Confidence Level', ha='center', size=25)
#
textstr_patch1 = '\n'.join((
r'$\bf{Mirkwood}$',
r'$\mathrm{ACE}=%.2f$' % (label_ace_mirk, ),
r'$\mathrm{IS}=%.2f$' % (label_is_mirk, ),
))
label_patch1 = Line2D([0], [0], marker=None, color='orange', label=textstr_patch1, lw=4, linestyle='solid')
if label_str not in [0,2]:
    textstr_patch2 = '\n'.join((
    r'$\bf{Traditional}$',
    r'$\mathrm{ACE}=%.2f$' % (label_ace_lower, ),
    r'$\mathrm{IS}=%.2f$' % (label_is_lower, ),
    ))
    label_patch2 = Line2D([0], [0], marker=None, color='blue', label=textstr_patch2, lw=4, linestyle='solid')

if label_str not in [0,2]:
    legend_elements = [label_patch1, label_patch2]
else:
    legend_elements = [label_patch1]

L1 = ax.legend(handles=legend_elements[0:1], loc=[0, 0.68], fancybox=True, framealpha=0.7, fontsize=18)
ax.add_artist(L1)
if label_str not in [0,2]:
    ax.legend(handles=legend_elements[1:], loc=[.58, 0.05], framealpha=0.7, fontsize=18)

plt.tight_layout()
plt.savefig('Calibration_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig('Calibration_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()

###### SHAP values #######################################
df_test = pd.read_csv(xtst_filename, index_col=0)
df_test.columns = central_wav_list+['z','Mass','Dust','Z']
df_test = 10**(df_test)-1

df_train = pd.read_csv(xtrn_filename, index_col=0)
df_train.columns = central_wav_list+['z','Mass','Dust','Z']
df_train = 10**(df_train)-1

df_shap_test = pd.read_csv(shapmean_filename, index_col=0)
# this is temporary

col_idx = 0
q=get_data(train_data)
fluxcutoff = np.max(q[0].loc[np.where(q[1][0]==2.)[0], list(q[0])[col_idx]].copy().values/mulfac)

df_testz0 = df_test.loc[df_test.iloc[:,col_idx]/mulfac>=fluxcutoff,:].copy()
df_testz2 = df_test.loc[df_test.iloc[:,col_idx]/mulfac<fluxcutoff,:].copy()
df_shap_testz0 = df_shap_test.loc[df_test.iloc[:,col_idx]/mulfac>=fluxcutoff,:].copy()
df_shap_testz2 = df_shap_test.loc[df_test.iloc[:,col_idx]/mulfac<fluxcutoff,:].copy()

plt.close('all')
fig, ax = plt.subplots(1,1, figsize=(8,5))
figname = 'SHAPz0_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow
plot_shapsubplot(df_shap_testz0, df_testz0, ax, fig, ticksevenorodd='even')
plt.show()

plt.close('all')
fig, ax = plt.subplots(1,1, figsize=(8,5))
figname = 'SHAPz2_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow
plot_shapsubplot(df_shap_testz2, df_testz2, ax, fig, ticksevenorodd='odd')
plt.show()

plt.close('all')
fig, ax = plt.subplots(1,1, figsize=(16,5))
figname = 'SHAP_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow
plot_shapsubplot(df_shap_test, df_test, ax, fig, ticksevenorodd='both')
plt.show()


#####################################################################
### epistemic errors ################################################
'''
variables to change for different properties:
col_name, xtrn_col, x_tst_col
'''

q=get_data(train_data)
col_idxz0 = np.where(abs(df_shap_testz0).sum(axis=0)==max(abs(df_shap_testz0).sum(axis=0)))[0].item()
col_idxz2 = np.where(abs(df_shap_testz2).sum(axis=0)==max(abs(df_shap_testz2).sum(axis=0)))[0].item()

col_name = col_namez2 = list(q[0])[col_idxz2]
col_namez0 = list(q[0])[col_idxz0]

fluxcutoffz0 = np.max(q[0].loc[np.where(q[1][0]==2.)[0], col_namez0].copy().values/mulfac)
fluxcutoffz2 = np.max(q[0].loc[np.where(q[1][0]==2.)[0], col_namez2].copy().values/mulfac)

xtrn_colz0 = df_train.iloc[:,col_idxz0].copy().values/mulfac
xtst_colz0 = df_test.iloc[:,col_idxz0].copy().values/mulfac

xtrn_colz2 = df_train.iloc[:,col_idxz2].copy().values/mulfac
xtst_colz2 = df_test.iloc[:,col_idxz2].copy().values/mulfac

bool_trn_z0 = xtrn_colz0>=fluxcutoffz0
bool_trn_z2 = xtrn_colz2<fluxcutoffz2
bool_tst_z0 = xtst_colz0>=fluxcutoffz0
bool_tst_z2 = xtst_colz2<fluxcutoffz2

xtrn_colz0 = xtrn_colz0[bool_trn_z0]
xtrn_colz2 = xtrn_colz2[bool_trn_z2]
xtst_colz0 = xtst_colz0[bool_tst_z0]
xtst_colz2 = xtst_colz2[bool_tst_z2]

#quantile based cutoff to remove outliers
ll, ul = np.quantile(xtrn_colz0, 0.05), np.quantile(xtrn_colz0, 0.95)
xtrn_colz0 = xtrn_colz0[xtrn_colz0>=ll]
xtrn_colz0 = xtrn_colz0[xtrn_colz0<=ul]

ll, ul = np.quantile(xtst_colz0, 0.05), np.quantile(xtst_colz0, 0.95)
idxz0 =  np.intersect1d(np.where(xtst_colz0>=ll)[0], np.where(xtst_colz0<=ul)[0])
xtst_colz0 = xtst_colz0[idxz0]

ll, ul = np.quantile(xtrn_colz2, 0.05), np.quantile(xtrn_colz2, 0.95)
xtrn_colz2 = xtrn_colz2[xtrn_colz2>=ll]
xtrn_colz2 = xtrn_colz2[xtrn_colz2<=ul]

ll, ul = np.quantile(xtst_colz2, 0.05), np.quantile(xtst_colz2, 0.95)
idxz2 =  np.intersect1d(np.where(xtst_colz2>=ll)[0], np.where(xtst_colz2<=ul)[0])
xtst_colz2 = xtst_colz2[idxz2]

xtrn_colz0 = np.log10(1+xtrn_colz0)
xtrn_colz2 = np.log10(1+xtrn_colz2)
xtst_colz0 = np.log10(1+xtst_colz0)
xtst_colz2 = np.log10(1+xtst_colz2)

epis_err_rel = gilda_file.pred_std_epis.values.reshape(-1,)/(gilda_file.pred_mean.values+1e-5)

epis_err_relz0 = epis_err_rel[bool_tst_z0]
epis_err_relz2 = epis_err_rel[bool_tst_z2]

epis_err_relz0 = epis_err_relz0[idxz0]
epis_err_relz2 = epis_err_relz2[idxz2]

if label_str!=0:
    epis_err_relz0 = np.log10(1+epis_err_relz0)
    epis_err_relz2 = np.log10(1+epis_err_relz2)

##### z=0 ####
plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
ax.set_facecolor('white')
ax.grid(True)
#_ = ax.errorbar(xtst_col, np.log10(gilda_file.pred_mean.values), yerr=epis_err, lw=4, color='k', ls='--', fmt='none')
(counts_tst, bins_tst) = np.histogram(xtst_colz0, bins=99)#, density=True)
(counts_trn, bins_trn) = np.histogram(xtrn_colz0, bins=99)#, density=True)
#factor = 0.5*np.max(epis_err_relz0)/max(np.mean(counts_tst), np.mean(counts_trn))
factor_trn = 0.5*np.max(epis_err_relz0)/np.mean(counts_trn)
factor_tst = 0.5*np.max(epis_err_relz0)/np.mean(counts_tst)
_ = ax.hist(bins_tst[:-1], bins_tst, weights=factor_tst*counts_tst, color='red', alpha=0.3, label='Test Set')
_ = ax.hist(bins_trn[:-1], bins_trn, weights=factor_trn*counts_trn, color='blue', alpha=0.3, label='Training Set')
_ = ax.scatter(xtst_colz0, epis_err_relz0, color='black', alpha=0.5)
#ax.set_xlim(left=min(np.min(xtrn_col),np.min(xtst_col)), right=min(np.min(xtrn_col),np.min(xtst_col)))
ax.set_xlim(left=0.99*np.min(xtst_colz0), right=1.01*np.max(xtst_colz0))
ax.set_ylim(bottom=0.99*np.min(epis_err_relz0), top=1.1*np.max(epis_err_relz0))
ax.set_xlabel(r'$\log(1+$'+col_namez0+')', ha='center', size=25)
if label_str!=0:
    ax.set_ylabel(r'$\log(1+\sigma_{\rm e})$', va='center', size=25, labelpad=20)
else:
    ax.set_ylabel(r'$\sigma_{\rm e}$', va='center', size=25, labelpad=20)

plt.legend()
plt.tight_layout()
plt.savefig('Epis_z0_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig('Epis_z0_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()

### z=2 ###
plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
ax.set_facecolor('white')
ax.grid(True)
#_ = ax.errorbar(xtst_col, np.log10(gilda_file.pred_mean.values), yerr=epis_err, lw=4, color='k', ls='--', fmt='none')
(counts_tst, bins_tst) = np.histogram(xtst_colz2, bins=99)#, density=True)
(counts_trn, bins_trn) = np.histogram(xtrn_colz2, bins=99)#, density=True)
#factor = 0.5*np.max(epis_err_relz2)/max(np.mean(counts_tst), np.mean(counts_trn))
factor_trn = 0.4*np.max(epis_err_relz2)/np.mean(counts_trn)
factor_tst = 0.4*np.max(epis_err_relz2)/np.mean(counts_tst)
_ = ax.hist(bins_tst[:-1], bins_tst, weights=factor_tst*counts_tst, color='red', alpha=0.3, label='Test Set')
_ = ax.hist(bins_trn[:-1], bins_trn, weights=factor_trn*counts_trn, color='blue', alpha=0.3, label='Training Set')
_ = ax.scatter(xtst_colz2, epis_err_relz2, color='black', alpha=0.5)
#ax.set_xlim(left=min(np.min(xtrn_col),np.min(xtst_col)), right=min(np.min(xtrn_col),np.min(xtst_col)))
ax.set_xlim(left=0.99*np.min(xtst_colz2), right=1.01*np.max(xtst_colz2))
ax.set_ylim(bottom=0.99*np.min(epis_err_relz2), top=1.1*np.max(epis_err_relz2))
ax.set_xlabel(r'$\log(1+$'+col_namez2+')', ha='center', size=25)
if label_str!=0:
    ax.set_ylabel(r'$\log(1+\sigma_{\rm e})$', va='center', size=25, labelpad=20)
else:
    ax.set_ylabel(r'$\sigma_{\rm e}$', va='center', size=25, labelpad=20)

plt.legend()
plt.tight_layout()
plt.savefig('Epis_z2_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig('Epis_z2_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()







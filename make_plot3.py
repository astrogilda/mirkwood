

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

###########################################################################
################################# for mass ################################
snr=5
timestr = '20201031'
label_str = 1
uncal_filename = 'uncal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
shapmean_filename = 'shapmea_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
xtrn_filename = 'xtrain_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
xtst_filename = 'xtest_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'


gilda_file = pd.read_csv(uncal_filename, index_col=0)#.drop(columns=['pred_std_epis'])
lower_file = combined_prosp[['true_stellar_mass', 'est_stellar_mass_50', 'est_stellar_mass_16', 'est_stellar_mass_84', 'redshift']].copy()
idx0 = np.where(lower_file['true_stellar_mass'].values>=gilda_file['true'].values.min())[0]
idx1 = np.where(lower_file['true_stellar_mass'].values<=gilda_file['true'].values.max())[0]
idx_common = np.intersect1d(idx0, idx1)
lower_file = lower_file.loc[idx_common].copy()
lower_file.reset_index(inplace=True, drop=True)
lower_file['est_stellar_mass_50'] = 10**lower_file['est_stellar_mass_50']
lower_file['est_stellar_mass_16'] = 10**lower_file['est_stellar_mass_16']
lower_file['est_stellar_mass_84'] = 10**lower_file['est_stellar_mass_84']


gilda_file_just_means = pd.DataFrame()
gilda_file_just_means['true'] = np.log10(gilda_file['true'].copy())
gilda_file_just_means['pred_mean'] = np.log10(gilda_file['pred_mean'].copy())
gilda_file_just_means['flag']='mirkwood'
gilda_file_just_means['redshift'] = np.nan
lower_file_just_means = pd.DataFrame()
lower_file_just_means['true'] = np.log10(lower_file['true_stellar_mass'].copy())
lower_file_just_means['pred_mean'] = np.log10(lower_file['est_stellar_mass_50'].copy())
lower_file_just_means['flag']='Non-Parametric'
lower_file_just_means['redshift'] = lower_file['redshift'].copy()
joint_file=pd.concat([gilda_file_just_means, lower_file_just_means]).reset_index(drop=True)

titletrue='M$^{\star}_{\mathrm{true}}$'
titlemodel='M$^{\star}_{\mathrm{model}}$'
title_den='M$_{\odot}$'

plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
x, y, k = joint_file["true"], joint_file["pred_mean"], joint_file["flag"]
ax.plot(gilda_file_just_means.true.values, gilda_file_just_means.true.values, lw=4, color='k', ls='--')
#sns.kdeplot(x=gilda_file_just_means.true, y=gilda_file_just_means.pred_mean, cmap="Oranges", shade=False, thresh=0.1, levels=5, ax=ax, linestyles="--", label='mirkwood', linewidths=5)
sns.kdeplot(x=lower_file_just_means.true, y=lower_file_just_means.pred_mean, cmap="Blues", shade=False, ax=ax, linestyles="-", label='Traditional', linewidths=5)#, thresh=0.1, levels=5
ax.scatter(x=gilda_file_just_means.true, y=gilda_file_just_means.pred_mean, color='orange', alpha=0.5)
#
ax.set_facecolor('white')
ax.grid(True)
ax.set_ylim(bottom=ax.get_xlim()[0], top=ax.get_xlim()[1])
ax.set_xlabel(r'$\log$ ('+titletrue+' / '+title_den+')', ha='center', size=25)
ax.set_ylabel(r'$\log$ ('+titlemodel+' / '+title_den+')', va='center', size=25, labelpad=20)
#
label_nrmse_patch1 = nrmse(10**gilda_file_just_means['true'].values, 10**gilda_file_just_means['pred_mean'].values)
label_nmae_patch1 = nmae(10**gilda_file_just_means['true'].values, 10**gilda_file_just_means['pred_mean'].values)
label_mape_patch1 = mape(10**gilda_file_just_means['true'].values, 10**gilda_file_just_means['pred_mean'].values)
label_bias_patch1 = nbe(10**gilda_file_just_means['true'].values, 10**gilda_file_just_means['pred_mean'].values)
textstr_patch1 = '\n'.join((
r'$\bf{Mirkwood}$',
r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse_patch1, ),
r'$\mathrm{NMAE}=%.2f$' % (label_nmae_patch1, ),
r'$\mathrm{NBE}=%.2f$' % (label_bias_patch1, ),
))
label_nrmse_patch2 = nrmse(10**lower_file_just_means['true'].values, 10**lower_file_just_means['pred_mean'].values)
label_nmae_patch2 = nmae(10**lower_file_just_means['true'].values, 10**lower_file_just_means['pred_mean'].values)
label_mape_patch2 = mape(10**lower_file_just_means['true'].values, 10**lower_file_just_means['pred_mean'].values)
label_bias_patch2 = nbe(10**lower_file_just_means['true'].values, 10**lower_file_just_means['pred_mean'].values)
textstr_patch2 = '\n'.join((
r'$\bf{Traditional}$',
r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse_patch2, ),
r'$\mathrm{NMAE}=%.2f$' % (label_nmae_patch2, ),
r'$\mathrm{NBE}=%.2f$' % (label_bias_patch2, ),
))
#
label_patch1 = Line2D([0], [0], marker=None, color='orange', label=textstr_patch1, lw=4, linestyle='dotted')
label_patch2 = Line2D([0], [0], marker=None, color='blue', label=textstr_patch2, lw=4, linestyle='solid')
legend_elements = [label_patch1, label_patch2]
L1 = ax.legend(handles=legend_elements[0:1], loc=[0, 0.68], fancybox=True, framealpha=0.7, fontsize=18)
ax.add_artist(L1)
ax.legend(handles=legend_elements[1:], loc=[.58, 0.05], framealpha=0.7, fontsize=18)
#
plt.tight_layout()
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()


#####################################################################
### epistemic errors ################################################
'''
variables to change for different properties:
col_name, xtrn_col, x_tst_col
'''
col_name = 'jwst_f200w'
q=get_data(train_data)
fluxcutoff = np.max(q[0].loc[np.where(q[1][0]==2.)[0], col_name].copy().values/mulfac)
xtrn_col = pd.read_csv(xtrn_filename, index_col=0).loc[:,col_name].copy().values/mulfac
xtst_col = pd.read_csv(xtst_filename, index_col=0).loc[:,col_name].copy().values/mulfac
xtrn_colz0 = xtrn_col[xtrn_col>=fluxcutoff]
xtrn_colz2 = xtrn_col[xtrn_col<fluxcutoff]
xtst_colz0 = xtst_col[xtst_col>=fluxcutoff]
xtst_colz2 = xtst_col[xtst_col<fluxcutoff]

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

epis_err_rel = gilda_file.pred_std_epis.values.reshape(-1,)
#epis_err_rel = gilda_file.pred_std_epis.values.reshape(-1,)/gilda_file.pred_mean.values

epis_err_relz0 = epis_err_rel[xtst_col>=fluxcutoff]
epis_err_relz2 = epis_err_rel[xtst_col<fluxcutoff]

epis_err_relz0 = epis_err_relz0[idxz0]
epis_err_relz2 = epis_err_relz2[idxz2]

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
ax.set_xlabel(r'$\log(1+$'+col_name+')', ha='center', size=25)
ax.set_ylabel(r'$\log(1+\sigma_{\rm e})$', va='center', size=25, labelpad=20)
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
factor_trn = 0.5*np.max(epis_err_relz2)/np.mean(counts_trn)
factor_tst = 0.5*np.max(epis_err_relz2)/np.mean(counts_tst)
_ = ax.hist(bins_tst[:-1], bins_tst, weights=factor_tst*counts_tst, color='red', alpha=0.3, label='Test Set')
_ = ax.hist(bins_trn[:-1], bins_trn, weights=factor_trn*counts_trn, color='blue', alpha=0.3, label='Training Set')
_ = ax.scatter(xtst_colz2, epis_err_relz2, color='black', alpha=0.5)
#ax.set_xlim(left=min(np.min(xtrn_col),np.min(xtst_col)), right=min(np.min(xtrn_col),np.min(xtst_col)))
ax.set_xlim(left=0.99*np.min(xtst_colz2), right=1.01*np.max(xtst_colz2))
ax.set_ylim(bottom=0.99*np.min(epis_err_relz2), top=1.1*np.max(epis_err_relz2))
ax.set_xlabel(r'$\log(1+$'+col_name+')', ha='center', size=25)
ax.set_ylabel(r'$\log(1+\sigma_{\rm e})$', va='center', size=25, labelpad=20)
plt.legend()
plt.tight_layout()
plt.savefig('Epis_z2_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig('Epis_z2_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()

###### SHAP values #######################################
df_test = pd.read_csv(xtst_filename, index_col=0)
df_test.columns = central_wav_list
df_shap_test = pd.read_csv(shapmean_filename, index_col=0)

plt.close('all')
fig, ax = plt.subplots(1,1, figsize=(16,5))
figname = 'SHAP_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow
plot_shapsubplot(df_shap_test, df_test, ax, fig)
plt.show()

####### Calibration Plot ################################

u_gilda = cdf_normdist(y=gilda_file.true.values, loc=gilda_file.pred_mean.values, scale=.5*(gilda_file.pred_upper-gilda_file.pred_lower).values)
u_lower = cdf_normdist(y=lower_file.iloc[:,0].values, loc=lower_file.iloc[:,1].values, scale=.5*(lower_file.iloc[:,3].values-lower_file.iloc[:,2].values))
#
label_ace_mirk = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))
label_is_mirk = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))
label_ace_lower = ace(lower_file.iloc[:,0].values, (lower_file.iloc[:,1].values, lower_file.iloc[:,2].values, lower_file.iloc[:,3].values))
label_is_lower = interval_sharpness(lower_file.iloc[:,0].values, (lower_file.iloc[:,1].values, lower_file.iloc[:,2].values, lower_file.iloc[:,3].values))
#
plt.close('all')
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.set_facecolor='white'
ax.grid(True)
ax.plot(np.linspace(0, 1, len(u_gilda)), np.sort(u_gilda), ls='-', color='orange', lw=5, label='Mirkwood')
ax.plot(np.linspace(0, 1, len(u_lower)), np.sort(u_lower), ls='-', color='blue', lw=5, label='Traditional')
ax.plot(np.linspace(0, 1, len(u_lower)), np.linspace(0, 1, len(u_lower)), lw=4, color='k', ls='--')
ax.set_xlabel(r'Expected Confidence Level', ha='center', size=25)
ax.set_ylabel(r'Observed Confidence Level', ha='center', size=25)
#
textstr_patch1 = '\n'.join((
r'$\bf{Mirkwood}$',
r'$\mathrm{ACE}=%.2f$' % (label_ace_mirk, ),
r'$\mathrm{IS}=%.2f$' % (label_is_mirk, ),
))
textstr_patch2 = '\n'.join((
r'$\bf{Traditional}$',
r'$\mathrm{ACE}=%.2f$' % (label_ace_lower, ),
r'$\mathrm{IS}=%.2f$' % (label_is_lower, ),
))
#
label_patch1 = Line2D([0], [0], marker=None, color='orange', label=textstr_patch1, lw=4, linestyle='solid')
label_patch2 = Line2D([0], [0], marker=None, color='blue', label=textstr_patch2, lw=4, linestyle='solid')
legend_elements = [label_patch1, label_patch2]
L1 = ax.legend(handles=legend_elements[0:1], loc=[0, 0.68], fancybox=True, framealpha=0.7, fontsize=18)
ax.add_artist(L1)
ax.legend(handles=legend_elements[1:], loc=[.58, 0.05], framealpha=0.7, fontsize=18)
#
plt.tight_layout()
plt.savefig('Calibration_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig('Calibration_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()


##########################################################
#################### for redshift #######################
snr=5
timestr = '20201031'
label_str = 0
uncal_filename = 'uncal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
shapmean_filename = 'shapmea_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
xtrn_filename = 'xtrain_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
xtst_filename = 'xtest_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'


gilda_file = pd.read_csv(uncal_filename, index_col=0)#.drop(columns=['pred_std_epis'])
gilda_file_just_means = pd.DataFrame()
gilda_file_just_means['true'] = gilda_file['true']
gilda_file_just_means['pred_mean'] = gilda_file['pred_mean'].copy()

titletrue=r'$z_{\mathrm{true}}$'
titlemodel=r'$z_{\mathrm{model}}$'

plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
ax.plot(np.linspace(-0.1, 2.1, 100), np.linspace(-0.1, 2.1, 100), lw=2, color='k', ls='--')
ax.scatter(x=gilda_file_just_means.true, y=gilda_file_just_means.pred_mean, color='orange', alpha=1)
#
ax.set_facecolor('white')
ax.grid(True)
ax.set_ylim(bottom=ax.get_xlim()[0], top=ax.get_xlim()[1])
ax.set_xlabel(titletrue, ha='center', size=25)
ax.set_ylabel(titlemodel, va='center', size=25, labelpad=20)
#
label_nrmse_patch1 = nrmse(gilda_file_just_means['true'].values, gilda_file_just_means['pred_mean'].values)
label_nmae_patch1 = nmae(gilda_file_just_means['true'].values, gilda_file_just_means['pred_mean'].values)
label_mape_patch1 = mape(gilda_file_just_means['true'].values, gilda_file_just_means['pred_mean'].values)
label_bias_patch1 = nbe(gilda_file_just_means['true'].values, gilda_file_just_means['pred_mean'].values)
textstr_patch1 = '\n'.join((
r'$\bf{Mirkwood}$',
r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse_patch1, ),
r'$\mathrm{NMAE}=%.2f$' % (label_nmae_patch1, ),
r'$\mathrm{NBE}=%.2f$' % (label_bias_patch1, ),
))
#
label_patch1 = Line2D([0], [0], marker=None, color='orange', label=textstr_patch1, lw=4, linestyle='dotted')
legend_elements = [label_patch1]
L1 = ax.legend(handles=legend_elements[0:1], loc=[0, 0.68], fancybox=True, framealpha=0.7, fontsize=18)
ax.add_artist(L1)
#
plt.tight_layout()
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()


#####################################################################
### epistemic errors ################################################
'''
variables to change for different properties:
col_name, xtrn_col, x_tst_col
'''
q=get_data(train_data)
col_name = 'alma_band6'
fluxcutoff = np.max(q[0].loc[np.where(q[1][0]==2.)[0], col_name].copy().values/mulfac)
xtrn_col = pd.read_csv(xtrn_filename, index_col=0).loc[:,col_name].copy().values/mulfac
xtst_col = pd.read_csv(xtst_filename, index_col=0).loc[:,col_name].copy().values/mulfac
xtrn_colz0 = xtrn_col[xtrn_col>=fluxcutoff]
xtrn_colz2 = xtrn_col[xtrn_col<fluxcutoff]
xtst_colz0 = xtst_col[xtst_col>=fluxcutoff]
xtst_colz2 = xtst_col[xtst_col<fluxcutoff]

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

epis_err_rel = gilda_file.pred_std_epis.values.reshape(-1,)
#epis_err_rel = gilda_file.pred_std_epis.values.reshape(-1,)/gilda_file.pred_mean.values

epis_err_relz0 = epis_err_rel[xtst_col>=fluxcutoff]
epis_err_relz2 = epis_err_rel[xtst_col<fluxcutoff]

epis_err_relz0 = epis_err_relz0[idxz0]
epis_err_relz2 = epis_err_relz2[idxz2]

#epis_err_relz0 = np.log10(1+epis_err_relz0)
#epis_err_relz2 = np.log10(1+epis_err_relz2)

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
ax.set_xlabel(r'$\log(1+$'+col_name+')', ha='center', size=25)
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
ax.set_xlabel(r'$\log(1+$'+col_name+')', ha='center', size=25)
ax.set_ylabel(r'$\sigma_{\rm e}$', va='center', size=25, labelpad=20)
plt.legend()
plt.tight_layout()
plt.savefig('Epis_z2_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig('Epis_z2_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()

###### SHAP values #######################################
df_test = pd.read_csv(xtst_filename, index_col=0)
df_test.columns = central_wav_list
df_shap_test = pd.read_csv(shapmean_filename, index_col=0)

plt.close('all')
fig, ax = plt.subplots(1,1, figsize=(16,5))
figname = 'SHAP_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow
plot_shapsubplot(df_shap_test, df_test, ax, fig)
plt.show()

####### Calibration Plot ################################

u_gilda = cdf_normdist(y=gilda_file.true.values, loc=gilda_file.pred_mean.values, scale=.5*(gilda_file.pred_upper-gilda_file.pred_lower).values)
#
label_ace_mirk = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))
label_is_mirk = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))
#
plt.close('all')
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.set_facecolor='white'
ax.grid(True)
ax.plot(np.linspace(0, 1, len(u_gilda)), np.sort(u_gilda), ls='-', color='orange', lw=5, label='Mirkwood')
ax.plot(np.linspace(0, 1, len(u_lower)), np.linspace(0, 1, len(u_lower)), lw=4, color='k', ls='--')
ax.set_xlabel(r'Expected Confidence Level', ha='center', size=25)
ax.set_ylabel(r'Observed Confidence Level', ha='center', size=25)
#
textstr_patch1 = '\n'.join((
r'$\bf{Mirkwood}$',
r'$\mathrm{ACE}=%.2f$' % (label_ace_mirk, ),
r'$\mathrm{IS}=%.2f$' % (label_is_mirk, ),
))
#
label_patch1 = Line2D([0], [0], marker=None, color='orange', label=textstr_patch1, lw=4, linestyle='solid')
legend_elements = [label_patch1]
L1 = ax.legend(handles=legend_elements[0:1], loc=[0, 0.68], fancybox=True, framealpha=0.7, fontsize=18)
ax.add_artist(L1)
#
plt.tight_layout()
plt.savefig('Calibration_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig('Calibration_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()


##################################################################
############ For dust mass #######################################
snr=5
timestr = '20201031'
label_str = 2
uncal_filename = 'uncal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
shapmean_filename = 'shapmea_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
xtrn_filename = 'xtrain_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
xtst_filename = 'xtest_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'


gilda_file = pd.read_csv(uncal_filename, index_col=0)#.drop(columns=['pred_std_epis'])
lower_file = combined_prosp[['true_dust_mass', 'est_dustmass']].copy()
idx0 = np.where(lower_file['true_dust_mass'].values>=gilda_file['true'].values.min())[0]
idx1 = np.where(lower_file['true_dust_mass'].values<=gilda_file['true'].values.max())[0]
idx_common = np.intersect1d(idx0, idx1)
lower_file = lower_file.loc[idx_common].copy()
lower_file.reset_index(inplace=True, drop=True)


gilda_file_just_means = pd.DataFrame()
gilda_file_just_means['true'] = np.log10(1+gilda_file['true'].copy())
gilda_file_just_means['pred_mean'] = np.log10(1+gilda_file['pred_mean'].copy())
gilda_file_just_means['flag']='mirkwood'
lower_file_just_means = pd.DataFrame()
lower_file_just_means['true'] = np.log10(1+lower_file['true_dust_mass'].copy())
lower_file_just_means['pred_mean'] = np.log10(1+lower_file['est_dustmass'].copy())
lower_file_just_means['flag']='Non-Parametric'

titletrue='M$_{\mathrm{dust, true}}$'
titlemodel='M$_{\mathrm{dust, model}}$'
title_den='M$_{\odot}$'

plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
ax.grid(True)
ax.set_facecolor('white')
ax.plot(gilda_file_just_means.true.values, gilda_file_just_means.true.values, lw=2, color='k', ls='--')
sns.kdeplot(x=lower_file_just_means.true, y=lower_file_just_means.pred_mean, cmap="Blues", shade=False, ax=ax, linestyles="-", label='Traditional', linewidths=4, thresh=0.1, levels=10)
ax.scatter(x=gilda_file_just_means.true, y=gilda_file_just_means.pred_mean, color='orange', alpha=0.5)
#
ax.set_ylim(bottom=ax.get_xlim()[0], top=ax.get_xlim()[1])
ax.set_xlabel(r'$\log$ ('+titletrue+' / '+title_den+')', ha='center', size=25)
ax.set_ylabel(r'$\log$ ('+titlemodel+' / '+title_den+')', va='center', size=25, labelpad=20)
#
label_nrmse_patch1 = nrmse(10**gilda_file_just_means['true'].values-1, 10**gilda_file_just_means['pred_mean'].values-1)
label_nmae_patch1 = nmae(10**gilda_file_just_means['true'].values-1, 10**gilda_file_just_means['pred_mean'].values-1)
label_mape_patch1 = mape(10**gilda_file_just_means['true'].values-1, 10**gilda_file_just_means['pred_mean'].values-1)
label_bias_patch1 = nbe(10**gilda_file_just_means['true'].values-1, 10**gilda_file_just_means['pred_mean'].values-1)
textstr_patch1 = '\n'.join((
r'$\bf{Mirkwood}$',
r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse_patch1, ),
r'$\mathrm{NMAE}=%.2f$' % (label_nmae_patch1, ),
r'$\mathrm{NBE}=%.2f$' % (label_bias_patch1, ),
))
label_nrmse_patch2 = nrmse(10**lower_file_just_means['true'].values-1, 10**lower_file_just_means['pred_mean'].values-1)
label_nmae_patch2 = nmae(10**lower_file_just_means['true'].values-1, 10**lower_file_just_means['pred_mean'].values-1)
label_mape_patch2 = mape(10**lower_file_just_means['true'].values-1, 10**lower_file_just_means['pred_mean'].values-1)
label_bias_patch2 = nbe(10**lower_file_just_means['true'].values-1, 10**lower_file_just_means['pred_mean'].values-1)
textstr_patch2 = '\n'.join((
r'$\bf{Traditional}$',
r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse_patch2, ),
r'$\mathrm{NMAE}=%.2f$' % (label_nmae_patch2, ),
r'$\mathrm{NBE}=%.2f$' % (label_bias_patch2, ),
))
#
label_patch1 = Line2D([0], [0], marker=None, color='orange', label=textstr_patch1, lw=4, linestyle='dotted')
label_patch2 = Line2D([0], [0], marker=None, color='blue', label=textstr_patch2, lw=4, linestyle='solid')
legend_elements = [label_patch1, label_patch2]
L1 = ax.legend(handles=legend_elements[0:1], loc=[0, 0.68], fancybox=True, framealpha=0.7, fontsize=18)
ax.add_artist(L1)
ax.legend(handles=legend_elements[1:], loc=[.58, 0.05], framealpha=0.7, fontsize=18)
#
plt.tight_layout()
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()


###### SHAP values #######################################
df_test = pd.read_csv(xtst_filename, index_col=0)
df_test.columns = central_wav_list
df_shap_test = pd.read_csv(shapmean_filename, index_col=0)
#this is temporary
df_shap_test = df_shap_test.loc[:,list(df_shap_test)[:-2]].copy()

plt.close('all')
fig, ax = plt.subplots(1,1, figsize=(16,5))
figname = 'SHAP_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow
plot_shapsubplot(df_shap_test, df_test, ax, fig)
plt.show()

####### Calibration Plot ################################

u_gilda = cdf_normdist(y=gilda_file.true.values, loc=gilda_file.pred_mean.values, scale=.5*(gilda_file.pred_upper-gilda_file.pred_lower).values)
label_ace_mirk = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))
label_is_mirk = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))
#

plt.close('all')
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.set_facecolor='white'
ax.grid(True)
ax.plot(np.linspace(0, 1, len(u_gilda)), np.sort(u_gilda), ls='-', color='orange', lw=4, label='Mirkwood')
ax.plot(np.linspace(0, 1, len(u_lower)), np.linspace(0, 1, len(u_lower)), lw=2, color='k', ls='--')
ax.set_xlabel(r'Expected Confidence Level', ha='center', size=25)
ax.set_ylabel(r'Observed Confidence Level', ha='center', size=25)
#
textstr_patch1 = '\n'.join((
r'$\bf{Mirkwood}$',
r'$\mathrm{ACE}=%.2f$' % (label_ace_mirk, ),
r'$\mathrm{IS}=%.2f$' % (label_is_mirk, ),
))
#
label_patch1 = Line2D([0], [0], marker=None, color='orange', label=textstr_patch1, lw=4, linestyle='solid')
legend_elements = [label_patch1]
L1 = ax.legend(handles=legend_elements[0:1], loc=[0, 0.68], fancybox=True, framealpha=0.7, fontsize=18)
ax.add_artist(L1)
#
plt.tight_layout()
plt.savefig('Calibration_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig('Calibration_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()


#####################################################################
### epistemic errors ################################################
'''
variables to change for different properties:
col_name, xtrn_col, x_tst_col
'''
q=get_data(train_data)
col_name = list(q[0])[-3]
fluxcutoff = np.max(q[0].loc[np.where(q[1][0]==2.)[0], col_name].copy().values/mulfac)
xtrn_col = pd.read_csv(xtrn_filename, index_col=0).loc[:,col_name].copy().values/mulfac
xtst_col = pd.read_csv(xtst_filename, index_col=0).loc[:,col_name].copy().values/mulfac
xtrn_colz0 = xtrn_col[xtrn_col>=fluxcutoff]
xtrn_colz2 = xtrn_col[xtrn_col<fluxcutoff]
xtst_colz0 = xtst_col[xtst_col>=fluxcutoff]
xtst_colz2 = xtst_col[xtst_col<fluxcutoff]

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

#epis_err_rel = gilda_file.pred_std_epis.values.reshape(-1,)
epis_err_rel = gilda_file.pred_std_epis.values.reshape(-1,)/gilda_file.pred_mean.values

epis_err_relz0 = epis_err_rel[xtst_col>=fluxcutoff]
epis_err_relz2 = epis_err_rel[xtst_col<fluxcutoff]

epis_err_relz0 = epis_err_relz0[idxz0]
epis_err_relz2 = epis_err_relz2[idxz2]

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
factor_trn = 0.4*np.max(epis_err_relz0)/np.mean(counts_trn)
factor_tst = 0.4*np.max(epis_err_relz0)/np.mean(counts_tst)
_ = ax.hist(bins_tst[:-1], bins_tst, weights=factor_tst*counts_tst, color='red', alpha=0.3, label='Test Set')
_ = ax.hist(bins_trn[:-1], bins_trn, weights=factor_trn*counts_trn, color='blue', alpha=0.3, label='Training Set')
_ = ax.scatter(xtst_colz0, epis_err_relz0, color='black', alpha=0.5)
#ax.set_xlim(left=min(np.min(xtrn_col),np.min(xtst_col)), right=min(np.min(xtrn_col),np.min(xtst_col)))
ax.set_xlim(left=0.99*np.min(xtst_colz0), right=1.01*np.max(xtst_colz0))
ax.set_ylim(bottom=0.99*np.min(epis_err_relz0), top=1.1*np.max(epis_err_relz0))
ax.set_xlabel(r'$\log(1+$'+col_name+')', ha='center', size=25)
ax.set_ylabel(r'$\log(1+\frac{\sigma_{\rm e}}{\mu})$', va='center', size=25, labelpad=20)
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
factor_trn = 0.5*np.max(epis_err_relz2)/np.mean(counts_trn)
factor_tst = 0.5*np.max(epis_err_relz2)/np.mean(counts_tst)
_ = ax.hist(bins_tst[:-1], bins_tst, weights=factor_tst*counts_tst, color='red', alpha=0.3, label='Test Set')
_ = ax.hist(bins_trn[:-1], bins_trn, weights=factor_trn*counts_trn, color='blue', alpha=0.3, label='Training Set')
_ = ax.scatter(xtst_colz2, epis_err_relz2, color='black', alpha=0.5)
#ax.set_xlim(left=min(np.min(xtrn_col),np.min(xtst_col)), right=min(np.min(xtrn_col),np.min(xtst_col)))
ax.set_xlim(left=0.99*np.min(xtst_colz2), right=1.01*np.max(xtst_colz2))
ax.set_ylim(bottom=0.99*np.min(epis_err_relz2), top=1.1*np.max(epis_err_relz2))
ax.set_xlabel(r'$\log(1+$'+col_name+')', ha='center', size=25)
ax.set_ylabel(r'$\log(1+\frac{\sigma_{\rm e}}{\mu})$', va='center', size=25, labelpad=20)
plt.legend()
plt.tight_layout()
plt.savefig('Epis_z2_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig('Epis_z2_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()


#####################################################################
#######################3# for metallicity ###########################
label_str = 3
uncal_filename = 'uncal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
shapmean_filename = 'shapmea_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
xtrn_filename = 'xtrain_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
xtst_filename = 'xtest_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'

gilda_file = pd.read_csv(uncal_filename, index_col=0)
lower_file = 10**(combined_prosp[['true_log(z/zsol)', 'est_log(z/zsol)_50', 'est_log(z/zsol)_16', 'est_log(z/zsol)_84']].copy())
idx0 = np.where(lower_file.loc[:,list(lower_file)[0]].copy().values>=gilda_file['true'].values.min())[0]
idx1 = np.where(lower_file.loc[:,list(lower_file)[0]].copy().values<=gilda_file['true'].values.max())[0]
idx_common = np.intersect1d(idx0, idx1)
lower_file = lower_file.loc[idx_common].copy()
lower_file.reset_index(inplace=True, drop=True)
gilda_file = gilda_file.apply(pd.to_numeric)
lower_file = lower_file.apply(pd.to_numeric)

gilda_file_just_means = pd.DataFrame()
gilda_file_just_means['true'] = np.log10(gilda_file['true'].copy())
gilda_file_just_means['pred_mean'] = np.log10(gilda_file['pred_mean'].copy())
gilda_file_just_means['flag']='mirkwood'
#gilda_file_just_means['redshift'] = np.nan
lower_file_just_means = pd.DataFrame()
lower_file_just_means['true'] = np.log10(lower_file.loc[:, list(lower_file)[0]].copy())
lower_file_just_means['pred_mean'] = np.log10(lower_file.loc[:, list(lower_file)[1]].copy())
lower_file_just_means['flag']='Non-Parametric'
#lower_file_just_means['redshift'] = lower_file['redshift'].copy()

titletrue='Z$^{\star}_{\mathrm{true}}$'
titlemodel='Z$^{\star}_{\mathrm{model}}$'
title_den='Z$_{\odot}$'

plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
x, y, k = joint_file["true"], joint_file["pred_mean"], joint_file["flag"]
ax.plot(gilda_file_just_means.true.values, gilda_file_just_means.true.values, lw=2, color='k', ls='--')
sns.kdeplot(x=lower_file_just_means.true, y=lower_file_just_means.pred_mean, cmap="Blues", shade=False, ax=ax, linestyles="-", label='Traditional', linewidths=4)#, thresh=0.1, levels=5
ax.scatter(x=gilda_file_just_means.true, y=gilda_file_just_means.pred_mean, color='orange', alpha=0.5)
#
ax.set_facecolor('white')
ax.grid(True)
ax.set_ylim(bottom=ax.get_xlim()[0], top=ax.get_xlim()[1])
ax.set_xlabel(r'$\log$ ('+titletrue+' / '+title_den+')', ha='center', size=25)
ax.set_ylabel(r'$\log$ ('+titlemodel+' / '+title_den+')', va='center', size=25, labelpad=20)
#
label_nrmse_patch1 = nrmse(10**gilda_file_just_means['true'].values, 10**gilda_file_just_means['pred_mean'].values)
label_nmae_patch1 = nmae(10**gilda_file_just_means['true'].values, 10**gilda_file_just_means['pred_mean'].values)
label_mape_patch1 = mape(10**gilda_file_just_means['true'].values, 10**gilda_file_just_means['pred_mean'].values)
label_bias_patch1 = nbe(10**gilda_file_just_means['true'].values, 10**gilda_file_just_means['pred_mean'].values)
textstr_patch1 = '\n'.join((
r'$\bf{Mirkwood}$',
r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse_patch1, ),
r'$\mathrm{NMAE}=%.2f$' % (label_nmae_patch1, ),
r'$\mathrm{NBE}=%.2f$' % (label_bias_patch1, ),
))
label_nrmse_patch2 = nrmse(10**lower_file_just_means['true'].values, 10**lower_file_just_means['pred_mean'].values)
label_nmae_patch2 = nmae(10**lower_file_just_means['true'].values, 10**lower_file_just_means['pred_mean'].values)
label_mape_patch2 = mape(10**lower_file_just_means['true'].values, 10**lower_file_just_means['pred_mean'].values)
label_bias_patch2 = nbe(10**lower_file_just_means['true'].values, 10**lower_file_just_means['pred_mean'].values)
textstr_patch2 = '\n'.join((
r'$\bf{Traditional}$',
r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse_patch2, ),
r'$\mathrm{NMAE}=%.2f$' % (label_nmae_patch2, ),
r'$\mathrm{NBE}=%.2f$' % (label_bias_patch2, ),
))
#
label_patch1 = Line2D([0], [0], marker=None, color='orange', label=textstr_patch1, lw=4, linestyle='dotted')
label_patch2 = Line2D([0], [0], marker=None, color='blue', label=textstr_patch2, lw=4, linestyle='solid')
legend_elements = [label_patch1, label_patch2]
L1 = ax.legend(handles=legend_elements[0:1], loc=[0, 0.68], fancybox=True, framealpha=0.7, fontsize=18)
ax.add_artist(L1)
ax.legend(handles=legend_elements[1:], loc=[.58, 0.05], framealpha=0.7, fontsize=18)
#
plt.tight_layout()
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()

####### Calibration Plot ################################

u_gilda = cdf_normdist(y=gilda_file.true.values, loc=gilda_file.pred_mean.values, scale=.5*(gilda_file.pred_upper-gilda_file.pred_lower).values)
u_lower = cdf_normdist(y=lower_file.iloc[:,0].values, loc=lower_file.iloc[:,1].values, scale=.5*(lower_file.iloc[:,3].values-lower_file.iloc[:,2].values))
#
label_ace_mirk = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))
label_is_mirk = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))
label_ace_lower = ace(lower_file.iloc[:,0].values, (lower_file.iloc[:,1].values, lower_file.iloc[:,2].values, lower_file.iloc[:,3].values))
label_is_lower = interval_sharpness(lower_file.iloc[:,0].values, (lower_file.iloc[:,1].values, lower_file.iloc[:,2].values, lower_file.iloc[:,3].values))
#
plt.close('all')
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.set_facecolor='white'
ax.grid(True)
ax.plot(np.linspace(0, 1, len(u_gilda)), np.sort(u_gilda), ls='-', color='orange', lw=5, label='Mirkwood')
ax.plot(np.linspace(0, 1, len(u_lower)), np.sort(u_lower), ls='-', color='blue', lw=5, label='Traditional')
ax.plot(np.linspace(0, 1, len(u_lower)), np.linspace(0, 1, len(u_lower)), lw=4, color='k', ls='--')
ax.set_xlabel(r'Expected Confidence Level', ha='center', size=25)
ax.set_ylabel(r'Observed Confidence Level', ha='center', size=25)
#
textstr_patch1 = '\n'.join((
r'$\bf{Mirkwood}$',
r'$\mathrm{ACE}=%.2f$' % (label_ace_mirk, ),
r'$\mathrm{IS}=%.2f$' % (label_is_mirk, ),
))
textstr_patch2 = '\n'.join((
r'$\bf{Traditional}$',
r'$\mathrm{ACE}=%.2f$' % (label_ace_lower, ),
r'$\mathrm{IS}=%.2f$' % (label_is_lower, ),
))
#
label_patch1 = Line2D([0], [0], marker=None, color='orange', label=textstr_patch1, lw=4, linestyle='solid')
label_patch2 = Line2D([0], [0], marker=None, color='blue', label=textstr_patch2, lw=4, linestyle='solid')
legend_elements = [label_patch1, label_patch2]
L1 = ax.legend(handles=legend_elements[0:1], loc=[0, 0.68], fancybox=True, framealpha=0.7, fontsize=18)
ax.add_artist(L1)
ax.legend(handles=legend_elements[1:], loc=[.58, 0.05], framealpha=0.7, fontsize=18)
#
plt.tight_layout()
plt.savefig('Calibration_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig('Calibration_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()

###### SHAP values #######################################
df_test = pd.read_csv(xtst_filename, index_col=0)
df_test.columns = central_wav_list
df_shap_test = pd.read_csv(shapmean_filename, index_col=0)
# this is temporary
df_shap_test = df_shap_test.loc[:6201,list(df_shap_test)[:35]].copy()

col_name = 'jwst_f200w'
q=get_data(train_data)
fluxcutoff = np.max(q[0].loc[np.where(q[1][0]==2.)[0], col_name].copy().values/mulfac)

df_testz0 = df_test.loc[df_test.iloc[:,-1]/mulfac>=fluxcutoff,:].copy()
df_testz2 = df_test.loc[df_test.iloc[:,-1]/mulfac<fluxcutoff,:].copy()
df_shap_testz0 = df_shap_test.loc[df_test.iloc[:,-1]/mulfac>=fluxcutoff,:].copy()
df_shap_testz2 = df_shap_test.loc[df_test.iloc[:,-1]/mulfac<fluxcutoff,:].copy()

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

#####################################################################
### epistemic errors ################################################
'''
variables to change for different properties:
col_name, xtrn_col, x_tst_col
'''

q=get_data(train_data)

col_name = col_namez2 = list(q[0])[-1]
col_namez0 = list(q[0])[16]

fluxcutoffz0 = np.max(q[0].loc[np.where(q[1][0]==2.)[0], col_namez0].copy().values/mulfac)
fluxcutoffz2 = np.max(q[0].loc[np.where(q[1][0]==2.)[0], col_namez2].copy().values/mulfac)

xtrn_colz0 = pd.read_csv(xtrn_filename, index_col=0).loc[:,col_namez0].copy().values/mulfac
xtst_colz0 = pd.read_csv(xtst_filename, index_col=0).loc[:,col_namez0].copy().values/mulfac

xtrn_colz2 = pd.read_csv(xtrn_filename, index_col=0).loc[:,col_namez2].copy().values/mulfac
xtst_colz2 = pd.read_csv(xtst_filename, index_col=0).loc[:,col_namez2].copy().values/mulfac

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

epis_err_rel = gilda_file.pred_std_epis.values.reshape(-1,)
#epis_err_rel = gilda_file.pred_std_epis.values.reshape(-1,)/gilda_file.pred_mean.values
# this is temporary
epis_err_rel = epis_err_rel[:6202]

epis_err_relz0 = epis_err_rel[bool_tst_z0]
epis_err_relz2 = epis_err_rel[bool_tst_z2]

epis_err_relz0 = epis_err_relz0[idxz0]
epis_err_relz2 = epis_err_relz2[idxz2]

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
ax.set_ylabel(r'$\log(1+\sigma_{\rm e})$', va='center', size=25, labelpad=20)
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
ax.set_ylabel(r'$\log(1+\sigma_{\rm e})$', va='center', size=25, labelpad=20)
plt.legend()
plt.tight_layout()
plt.savefig('Epis_z2_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.savefig('Epis_z2_'+label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
plt.show()




















snr=5
timestr = '20200914'

label_str = 2
uncal_filename = 'uncal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_NATGRADFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(NATGRAD_FLAG))+timestr+'.csv'
cal_filename = 'cal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_NATGRADFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(NATGRAD_FLAG))+timestr+'.csv'
shapmean_filename = 'shapmea_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_NATGRADFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(NATGRAD_FLAG))+timestr+'.csv'

gilda_file = pd.concat((pd.read_csv(uncal_filename, index_col=0).drop(columns=['pred_std_epis']), pd.read_csv(cal_filename, index_col=0).drop(columns=['true', 'pred_std_epis'])), axis=1)
lower_file = 10**(combined_prosp[['true_log(z/zsol)', 'est_log(z/zsol)_50', 'est_log(z/zsol)_16', 'est_log(z/zsol)_84']].copy())
idx0 = np.where(lower_file['true_log(z/zsol)'].values>=gilda_file['true'].values.min())[0]
idx1 = np.where(lower_file['true_log(z/zsol)'].values<=gilda_file['true'].values.max())[0]
idx_common = np.intersect1d(idx0, idx1)
lower_file = lower_file.loc[idx_common].copy()
lower_file.reset_index(inplace=True, drop=True)

### bar plot ########

label_ace_prosp = ace(lower_file.iloc[:,0].values, (lower_file.iloc[:,1].values, lower_file.iloc[:,2].values, lower_file.iloc[:,3].values)) + 0.6827

label_is_prosp = interval_sharpness(lower_file.iloc[:,0].values, (lower_file.iloc[:,1].values, lower_file.iloc[:,2].values, lower_file.iloc[:,3].values))

label_ace_mirk = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,4].values, gilda_file.iloc[:,5].values, gilda_file.iloc[:,6].values)) + 0.6827

label_is_mirk = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,4].values, gilda_file.iloc[:,5].values, gilda_file.iloc[:,6].values))

label_ace_mirk_uncal = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values)) + 0.6827

label_is_mirk_uncal = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))

model_names = ['Mirkwood', 'Mirkwood,\nuncalibrated', 'Non-\nparametric']
clrs = ['blue', 'green', 'orange']
plt.close('all')
fig, ax = plt.subplots(figsize=(15,5)) # Create matplotlib figure
#ax = fig.add_subplot(111) # Create matplotlib axes
width = 0.6
dimw = width/2
x = np.arange(len(model_names)) + dimw/2
ax.set_yticks(x)#, size=25)
ax.set_ylim(bottom=x.min()-dimw, top=x.max()+dimw)
ax.set_yticklabels(model_names)

ax2 = ax.twiny()
ax2.set_ylim(ax.get_ylim())

ax.barh(np.arange(len(model_names)), (label_ace_mirk, label_ace_mirk_uncal, label_ace_prosp), height=dimw, color=clrs, alpha=1, left=0.001)
ax2.barh(np.arange(len(model_names)) + dimw, (label_is_mirk, label_is_mirk_uncal, label_is_prosp), height=dimw, color=clrs, alpha=0.4, left=0.001)

ax.set_xlabel(r'Fraction of Z$^{\star}_{\rm{true}}$ within 1$\sigma$ of Z$^{\star}_{\rm{model}}$ (ACE + 0.6827)', size=35)
ax2.set_xlabel('Interval Sharpness', labelpad=12, size=35)

ax.grid(False)
ax2.grid(False)

plt.savefig('barplot_'+label_list[label_str]+'_snr=%d'%snr+'.png', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor="#111111")
plt.savefig('barplot_'+label_list[label_str]+'_snr=%d'%snr+'.eps', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor="#111111")
plt.show()
################

gilda_file_just_means = pd.DataFrame()
gilda_file_just_means['true'] = np.log10(gilda_file['true'].copy())
gilda_file_just_means['pred_mean'] = np.log10(gilda_file['pred_mean_cal'].copy())
gilda_file_just_means['flag']='mirkwood'
lower_file_just_means = pd.DataFrame()
lower_file_just_means['true'] = np.log10(lower_file['true_log(z/zsol)'].copy())
lower_file_just_means['pred_mean'] = np.log10(lower_file['est_log(z/zsol)_50'].copy())
lower_file_just_means['flag']='Non-Parametric'
joint_file=pd.concat([gilda_file_just_means, lower_file_just_means]).reset_index(drop=True)

plt.close('all')
x, y, k = joint_file["true"], joint_file["pred_mean"], joint_file["flag"]
g = sns.JointGrid(height=15, space=0)
lp = sns.lineplot(x=gilda_file_just_means.true, y=gilda_file_just_means.true, lw=4, ax=g.ax_joint, color='white')
lp.lines[0].set_linestyle("--")
sns.kdeplot(x=gilda_file_just_means.true, y=gilda_file_just_means.pred_mean, cmap="Oranges", shade=False, thresh=0.1, levels=4, ax=g.ax_joint, linestyles="--", label='mirkwood', linewidths=5)
sns.kdeplot(x=lower_file_just_means.true, y=lower_file_just_means.pred_mean, cmap="Blues", shade=False, thresh=0.1, levels=4, ax=g.ax_joint, linestyles=":", label='Non-Parametric', linewidths=5)
g.ax_joint.set_ylim(bottom=g.ax_joint.get_xlim()[0], top=g.ax_joint.get_xlim()[1])

sns.kdeplot(x=gilda_file_just_means.true, color="orange", legend=False, lw=4, 
             ax=g.ax_marg_x, linestyle="--", fill=False)
sns.kdeplot(x=lower_file_just_means.true, color="blue", legend=False, lw=4, 
             ax=g.ax_marg_x, linestyle=":", fill=False)
sns.kdeplot(y=gilda_file_just_means.pred_mean, color="orange", legend=False, lw=4, 
             ax=g.ax_marg_y, linestyle="--", fill=False)
sns.kdeplot(y=lower_file_just_means.pred_mean, color="blue", legend=False, lw=4, 
             ax=g.ax_marg_y, linestyle=":", fill=False)

g.ax_joint.set_xlabel(r'$\log$ (Z$^{\star}_{\rm{true}}$ / Z$_{\odot}$)', ha='center', size=35)
g.ax_joint.set_ylabel(r'$\log$ (Z$^{\star}_{\rm{model}}$ / Z$_{\odot}$)', va='center', rotation='vertical', size=35, labelpad=30)
g.ax_joint.grid(False)
g.ax_marg_x.grid(False)
g.ax_marg_y.grid(False)

label_nrmse_patch1 = nrmse(10**gilda_file_just_means['true'].values, 10**gilda_file_just_means['pred_mean'].values)
label_nmae_patch1 = nmae(10**gilda_file_just_means['true'].values, 10**gilda_file_just_means['pred_mean'].values)
label_mape_patch1 = mape(10**gilda_file_just_means['true'].values, 10**gilda_file_just_means['pred_mean'].values)
label_bias_patch1 = nbe(10**gilda_file_just_means['true'].values, 10**gilda_file_just_means['pred_mean'].values)
textstr_patch1 = '\n'.join((
r'$\bf{Mirkwood}$',
r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse_patch1, ),
r'$\mathrm{NMAE}=%.2f$' % (label_nmae_patch1, ),
r'$\mathrm{NBE}=%.2f$' % (label_bias_patch1, ),
))
label_nrmse_patch2 = nrmse(10**lower_file_just_means['true'].values, 10**lower_file_just_means['pred_mean'].values)
label_nmae_patch2 = nmae(10**lower_file_just_means['true'].values, 10**lower_file_just_means['pred_mean'].values)
label_mape_patch2 = mape(10**lower_file_just_means['true'].values, 10**lower_file_just_means['pred_mean'].values)
label_bias_patch2 = nbe(10**lower_file_just_means['true'].values, 10**lower_file_just_means['pred_mean'].values)
textstr_patch2 = '\n'.join((
r'$\bf{Non-Parametric}$',
r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse_patch2, ),
r'$\mathrm{NMAE}=%.2f$' % (label_nmae_patch2, ),
r'$\mathrm{NBE}=%.2f$' % (label_bias_patch2, ),
))

label_patch1 = Line2D([0], [0], marker=None, color='orange', label=textstr_patch1, lw=4, linestyle='--')
label_patch2 = Line2D([0], [0], marker=None, color='blue', label=textstr_patch2, lw=4, linestyle=':')
legend_elements = [label_patch1, label_patch2]

L1 = g.ax_joint.legend(handles=legend_elements[0:1], loc=[0, 0.75], fontsize=30, fancybox=True, framealpha=0.7)
g.ax_joint.add_artist(L1)
g.ax_joint.legend(handles=legend_elements[1:], loc=[0.38, 0.75], fontsize=30)

plt.savefig(label_list[label_str]+'_snr=%d'%snr+'.png', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor="#111111")
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'.eps', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor="#111111")
plt.show()



#####################################################################
######################### for SFR ###################################
snr=5
timestr = '20200914'

label_str = 3
uncal_filename = 'uncal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_NATGRADFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(NATGRAD_FLAG))+timestr+'.csv'
cal_filename = 'cal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_NATGRADFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(NATGRAD_FLAG))+timestr+'.csv'
shapmean_filename = 'shapmea_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_NATGRADFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(NATGRAD_FLAG))+timestr+'.csv'

gilda_file = pd.concat((pd.read_csv(uncal_filename, index_col=0).drop(columns=['pred_std_epis']), pd.read_csv(cal_filename, index_col=0).drop(columns=['true', 'pred_std_epis'])), axis=1)
lower_file = combined_prosp[['true_sfr', 'est_sfr_50', 'est_sfr_16', 'est_sfr_84']].copy()
idx0 = np.where(lower_file['true_sfr'].values>=gilda_file['true'].values.min())[0]
idx1 = np.where(lower_file['true_sfr'].values<=gilda_file['true'].values.max())[0]
idx_common = np.intersect1d(idx0, idx1)
lower_file = lower_file.loc[idx_common].copy()
lower_file.reset_index(inplace=True, drop=True)

### bar plot ########
label_ace_prosp = ace(lower_file.iloc[:,0].values, (lower_file.iloc[:,1].values, lower_file.iloc[:,2].values, lower_file.iloc[:,3].values)) + 0.6827

label_is_prosp = interval_sharpness(lower_file.iloc[:,0].values, (lower_file.iloc[:,1].values, lower_file.iloc[:,2].values, lower_file.iloc[:,3].values))

label_ace_mirk = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,4].values, gilda_file.iloc[:,5].values, gilda_file.iloc[:,6].values)) + 0.6827

label_is_mirk = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,4].values, gilda_file.iloc[:,5].values, gilda_file.iloc[:,6].values))

label_ace_mirk_uncal = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values)) + 0.6827

label_is_mirk_uncal = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))

model_names = ['Mirkwood', 'Mirkwood,\nuncalibrated', 'Non-\nparametric']
clrs = ['blue', 'green', 'orange']
plt.close('all')
fig, ax = plt.subplots(figsize=(15,5)) # Create matplotlib figure
#ax = fig.add_subplot(111) # Create matplotlib axes
width = 0.6
dimw = width/2
x = np.arange(len(model_names)) + dimw/2
ax.set_yticks(x)#, size=25)
ax.set_ylim(bottom=x.min()-dimw, top=x.max()+dimw)
ax.set_yticklabels(model_names)

ax2 = ax.twiny()
ax2.set_ylim(ax.get_ylim())

ax.barh(np.arange(len(model_names)), (label_ace_mirk, label_ace_mirk_uncal, label_ace_prosp), height=dimw, color=clrs, alpha=1, left=0.001)
ax2.barh(np.arange(len(model_names)) + dimw, (label_is_mirk, label_is_mirk_uncal, label_is_prosp), height=dimw, color=clrs, alpha=0.4, left=0.001)

ax.set_xlabel(r'Fraction of SFR$_{\rm{100, true}}$ within 1$\sigma$ of SFR$_{\rm{100, model}}$(ACE + 0.6827)', size=35)
ax2.set_xlabel('Interval Sharpness', labelpad=12, size=35)

ax.grid(False)
ax2.grid(False)

plt.savefig('barplot_'+label_list[label_str]+'_snr=%d'%snr+'.png', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor="#111111")
plt.savefig('barplot_'+label_list[label_str]+'_snr=%d'%snr+'.eps', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor="#111111")
plt.show()

################

gilda_file_just_means = pd.DataFrame()
gilda_file_just_means['true'] = gilda_file['true'].copy()
gilda_file_just_means['pred_mean'] = gilda_file['pred_mean_cal'].copy()
gilda_file_just_means = np.log10(gilda_file_just_means+1)
gilda_file_just_means['flag']='mirkwood'

idx = gilda_file_just_means.true < 0.75
gilda_file_just_means = gilda_file_just_means[idx]
gilda_file_just_means[idx].reset_index(inplace=True)

lower_file_just_means = pd.DataFrame()
lower_file_just_means['true'] = lower_file['true_sfr'].copy()
lower_file_just_means['pred_mean'] = lower_file['est_sfr_50'].copy()
lower_file_just_means = np.log10(lower_file_just_means+1)
lower_file_just_means['flag']='Non-Parametric'

idx = lower_file_just_means.true < 0.75
lower_file_just_means = lower_file_just_means[idx]
lower_file_just_means[idx].reset_index(inplace=True)


joint_file=pd.concat([gilda_file_just_means, lower_file_just_means]).reset_index(drop=True)

idx = joint_file.true < 0.75
joint_file = joint_file[idx]
joint_file[idx].reset_index(inplace=True)

plt.close('all')
x, y, k = joint_file["true"], joint_file["pred_mean"], joint_file["flag"]
g = sns.JointGrid(height=15, space=0)
lp = sns.lineplot(x=gilda_file_just_means.true, y=gilda_file_just_means.true, lw=4, ax=g.ax_joint, color='white')
lp.lines[0].set_linestyle("--")
sns.kdeplot(x=gilda_file_just_means.true, y=gilda_file_just_means.pred_mean, cmap="Oranges", shade=False, thresh=0.1, levels=5, ax=g.ax_joint, linestyles="--", label='mirkwood', linewidths=5)
sns.kdeplot(x=lower_file_just_means.true, y=lower_file_just_means.pred_mean, cmap="Blues", shade=False, thresh=0.1, levels=5, ax=g.ax_joint, linestyles=":", label='Non-Parametric', linewidths=5)
g.ax_joint.set_ylim(bottom=g.ax_joint.get_xlim()[0], top=g.ax_joint.get_xlim()[1])

sns.kdeplot(x=gilda_file_just_means.true, color="orange", legend=False, lw=4, 
             ax=g.ax_marg_x, linestyle="--", fill=False)
sns.kdeplot(x=lower_file_just_means.true, color="blue", legend=False, lw=4, 
             ax=g.ax_marg_x, linestyle=":", fill=False)
sns.kdeplot(y=gilda_file_just_means.pred_mean, color="orange", legend=False, lw=4, 
             ax=g.ax_marg_y, linestyle="--", fill=False)
sns.kdeplot(y=lower_file_just_means.pred_mean, color="blue", legend=False, lw=4, 
             ax=g.ax_marg_y, linestyle=":", fill=False)

g.ax_joint.set_xlabel(r'$\log$ (SFR$_{\rm{100, true}}$)', ha='center', size=35)
g.ax_joint.set_ylabel(r'$\log$ (SFR$_{\rm{100, model}}$)', va='center', rotation='vertical', size=35, labelpad=30)
g.ax_joint.grid(False)
g.ax_marg_x.grid(False)
g.ax_marg_y.grid(False)

label_nrmse_patch1 = nrmse(gilda_file_just_means['true'].values, gilda_file_just_means['pred_mean'].values)
label_nmae_patch1 = nmae(gilda_file_just_means['true'].values, gilda_file_just_means['pred_mean'].values)
label_mape_patch1 = mape(gilda_file_just_means['true'].values, gilda_file_just_means['pred_mean'].values)
label_bias_patch1 = nbe(gilda_file_just_means['true'].values, gilda_file_just_means['pred_mean'].values)
textstr_patch1 = '\n'.join((
r'$\bf{Mirkwood}$',
r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse_patch1, ),
r'$\mathrm{NMAE}=%.2f$' % (label_nmae_patch1, ),
r'$\mathrm{NBE}=%.2f$' % (label_bias_patch1, ),
))
label_nrmse_patch2 = nrmse(lower_file_just_means['true'].values, lower_file_just_means['pred_mean'].values)
label_nmae_patch2 = nmae(lower_file_just_means['true'].values, lower_file_just_means['pred_mean'].values)
label_mape_patch2 = mape(lower_file_just_means['true'].values, lower_file_just_means['pred_mean'].values)
label_bias_patch2 = nbe(lower_file_just_means['true'].values, lower_file_just_means['pred_mean'].values)
textstr_patch2 = '\n'.join((
r'$\bf{Non-Parametric}$',
r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse_patch2, ),
r'$\mathrm{NMAE}=%.2f$' % (label_nmae_patch2, ),
r'$\mathrm{NBE}=%.2f$' % (label_bias_patch2, ),
))

label_patch1 = Line2D([0], [0], marker=None, color='orange', label=textstr_patch1, lw=4, linestyle='--')
label_patch2 = Line2D([0], [0], marker=None, color='blue', label=textstr_patch2, lw=4, linestyle=':')
legend_elements = [label_patch1, label_patch2]

L1 = g.ax_joint.legend(handles=legend_elements[0:1], loc=[0, 0.75], fontsize=30, fancybox=True, framealpha=0.7)
g.ax_joint.add_artist(L1)
g.ax_joint.legend(handles=legend_elements[1:], loc=[0.38, 0.75], fontsize=30)

plt.savefig(label_list[label_str]+'_snr=%d'%snr+'.png', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor="#111111")
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'.eps', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor="#111111")
plt.show()






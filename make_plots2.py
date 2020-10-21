
import seaborn as sns, numpy as np, matplotlib.pyplot as plt, pandas as pd
import time
import shap

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
plt.style.use('seaborn-bright')
# also including the facecolor argument in plt.savefig. remove for regular images.
timenow = time.strftime("%Y%m%d")


def stackedbarplots()



#'VALUE': "SHAP value\n(Impact on model output)"
labels = {
    'MAIN_EFFECT': "SHAP main effect value for\n%s",
    'INTERACTION_VALUE': "SHAP interaction value",
    'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
    'VALUE': "SHAP value\n(Impact on model output)",
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


def plot_shapsubplot(df_shap_test, df_features_test, axshap, figshap, tickstoplot='even'):
    sort=False
    CHAIN_FLAG = False
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
    #q = shap.summary_plot(df_shap_test.values, df_shap_test, max_display=max_display, sort=sort, show=False, plot_size=None)#, plot_type="bar")
    #plt.cla()
    ### horizontal plotting as opposed the default vertical plotting
    #plt.close('all')
    #fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
    #for i,j,k,l,m in zip(*q):
    #    _ = axshap.scatter(j, i, cmap=colors.red_blue, vmin=k, vmax=l, s=16, c=m, alpha=1, zorder=3, rasterized=len(i) > 500)#,linewidth=2
    #
    color_bar_label=labels["FEATURE_VALUE"]
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
    #starting_idx = 0 if ticksotplot=='even' else 1
    for tick in axshap.xaxis.get_major_ticks()[0::2]:
        tick.set_visible(False)#set_pad(100)
    ## Stagger x-axis labels [1::2] means start from the second element in the list and get every other element. 
    #axshap.minorticks_on()
    """
    axshap2 = axshap.twiny()
    axshap2.set_ylim(axshap.get_ylim())
    axshap2.xaxis.set_ticks_position('top')
    axshap2.yaxis.set_ticks_position('none')
    axshap2.spines['right'].set_visible(False)
    axshap2.spines['top'].set_visible(False)
    axshap2.spines['left'].set_visible(False)
    axshap.tick_params(color=axis_color, labelcolor=axis_color)
    _ = axshap2.set_xticks(range(len(feature_order)))
    _ = axshap2.set_xticklabels([feature_names[i] for i in feature_order], fontsize=25, rotation=90)
    axshap2.grid(False)
    for tick in axshap2.xaxis.get_major_ticks()[1::2]:
        tick.set_visible(False)#set_pad(100)
    """
    axshap.tick_params('x', length=20, width=0.5, which='major')
    axshap.tick_params('y', labelsize=25)
    _ = axshap.set_xlim(left=-1, right=len(feature_order))
    _ = axshap.set_ylabel(labels['VALUE'], fontsize=25)
    _ = axshap.set_xlabel(r'Wavelength ($\mu$m)',fontsize=25)
    _ = axshap.invert_xaxis()
    return


plt.close('all')
fig, ax = plt.subplots(1,1)
ax2 = ax.twiny()
ax2.set_ylim(ax.get_ylim())
plot_shapsubplot(df_shap_test, df_test, ax, fig)
plt.show()

###########################################################################
################################# for mass ################################
snr=5
timestr = '20200909'
label_str = 0
uncal_filename = 'uncal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_NATGRADFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(NATGRAD_FLAG))+timestr+'.csv'
cal_filename = 'cal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_NATGRADFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(NATGRAD_FLAG))+timestr+'.csv'
shapmean_filename = 'shapmea_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_NATGRADFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(NATGRAD_FLAG))+timestr+'.csv'

gilda_file = pd.concat((pd.read_csv(uncal_filename, index_col=0).drop(columns=['pred_std_epis']), pd.read_csv(cal_filename, index_col=0).drop(columns=['true', 'pred_std_epis']).rename(columns=lambda x: x+'_cal')), axis=1)
lower_file = combined_prosp[['true_stellar_mass', 'est_stellar_mass_50', 'est_stellar_mass_16', 'est_stellar_mass_84']].copy()
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
gilda_file_just_means['pred_mean'] = np.log10(gilda_file['pred_mean_cal'].copy())
gilda_file_just_means['flag']='mirkwood'
lower_file_just_means = pd.DataFrame()
lower_file_just_means['true'] = np.log10(lower_file['true_stellar_mass'].copy())
lower_file_just_means['pred_mean'] = np.log10(lower_file['est_stellar_mass_50'].copy())
lower_file_just_means['flag']='Non-Parametric'
joint_file=pd.concat([gilda_file_just_means, lower_file_just_means]).reset_index(drop=True)

label_ace_prosp = ace(lower_file.iloc[:,0].values, (lower_file.iloc[:,1].values, lower_file.iloc[:,2].values, lower_file.iloc[:,3].values)) + 0.6827

label_is_prosp = interval_sharpness(lower_file.iloc[:,0].values, (lower_file.iloc[:,1].values, lower_file.iloc[:,2].values, lower_file.iloc[:,3].values))

label_ace_mirk = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,4].values, gilda_file.iloc[:,5].values, gilda_file.iloc[:,6].values)) + 0.6827

label_is_mirk = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,4].values, gilda_file.iloc[:,5].values, gilda_file.iloc[:,6].values))

label_ace_mirk_uncal = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values)) + 0.6827

label_is_mirk_uncal = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))

labels_ace = [label_ace_mirk, label_ace_mirk_uncal, label_ace_prosp]
labels_is = [label_is_mirk, label_is_mirk_uncal, label_is_prosp]
model_names = ['Mirkwood', 'Mirkwood,\nuncalibrated', 'Traditional\nSED Fitting']
clrs = ['blue', 'green', 'orange']

#################################################################
#################################################################

ytestattrib_filename = 'shapmea_label=Mass_TRAININGDATA=[\'simba\', \'eagle\', \'tng\']_TESTINGDATA=None_SNR=5_NUMBS=24_WEIGHTFLAG=False_nfoldsval=5_nfoldstest=5_xtrans=False_ytrans=True_MEDIANFLAG=True_NATGRADFLAG=True_20201013.csv'
ytestattrib_std_filename = 'shapstd_label=Mass_TRAININGDATA=[\'simba\', \'eagle\', \'tng\']_TESTINGDATA=None_SNR=5_NUMBS=24_WEIGHTFLAG=False_nfoldsval=5_nfoldstest=5_xtrans=False_ytrans=True_MEDIANFLAG=True_NATGRADFLAG=True_20201013.csv'
df_test_filename = 'uncal_label=Mass_TRAININGDATA=[\'simba\',\'eagle\',\'tng\']_TESTINGDATA=None_SNR=5_NUMBS=24_WEIGHTFLAG=False_nfoldsval=5_nfoldstest=5_xtrans=False_ytrans=True_MEDIANFLAG=True_NATGRADFLAG=True_20201013.csv'
df_test, _ = get_data(train_data)
df_test.columns = central_wav_list
df_shap_test = pd.read_csv(ytestattrib_filename, index_col=0)
df_shap_test_std = pd.read_csv(ytestattrib_std_filename, index_col=0)


def make_plot(snr=snr, label_str=label_str, model_names=model_names, crls=clrs, labels_ace=labels_ace, labels_is=labels_is, titletrue='M$^{\star}_{\mathrm{true}}$', titlemodel='M$^{\star}_{\mathrm{model}}$', title_den='M$_{\odot}$', isthisdust=False, df_shap_test=df_shap_test, df_test=df_test, figsize=(10,18), height_ratios=[4,1,4], tickstoplot='even'):
    plt.close('all')
    fig, axtot = plt.subplots(nrows=len(height_ratios), ncols=1, figsize=figsize, gridspec_kw={'height_ratios': height_ratios}) # Create matplotlib figure
    if len(height_ratios)==2:
        ax3, ax = axtot
    else:
        ax3, ax, ax4 = axtot
    width = 0.6
    dimw = width/2
    x = np.arange(len(model_names)) + dimw/2
    ax.set_yticks(x)#, size=25)
    ax.set_ylim(bottom=x.min()-dimw, top=x.max()+dimw)
    ax.set_yticklabels(model_names)
    ax2 = ax.twiny()
    ax2.set_ylim(ax.get_ylim())
    ax.barh(np.arange(len(model_names)), (*labels_ace,), height=dimw, color=clrs, alpha=1, left=0.001)
    ax2.barh(np.arange(len(model_names)) + dimw, (*labels_is,), height=dimw, color=clrs, alpha=0.4, left=0.001)
    if isthisdust:
        yticks = ax.yaxis.get_major_ticks()
        yticks[-1].set_visible(False)
    ax.set_xlabel(r'Fraction of '+titletrue+' within 1$\sigma$ of '+titlemodel+'\n(ECE + 0.6827)', size=25)
    ax2.set_xlabel('Interval Sharpness', labelpad=12, size=25)
    ax.grid(False)
    ax2.grid(False)
    #
    x, y, k = joint_file["true"], joint_file["pred_mean"], joint_file["flag"]
    ax3.plot(gilda_file_just_means.true.values, gilda_file_just_means.true.values, lw=4, color='k', ls='--')
    sns.kdeplot(x=gilda_file_just_means.true, y=gilda_file_just_means.pred_mean, cmap="Oranges", shade=False, thresh=0.1, levels=5, ax=ax3, linestyles="--", label='mirkwood', linewidths=5)
    sns.kdeplot(x=lower_file_just_means.true, y=lower_file_just_means.pred_mean, cmap="Blues", shade=False, thresh=0.1, levels=5, ax=ax3, linestyles=":", label='Traditional', linewidths=5)
    ax3.set_ylim(bottom=ax3.get_xlim()[0], top=ax3.get_xlim()[1])
    ax3.set_xlabel(r'$\log$ ('+titletrue+' / '+title_den+')', ha='center', size=25)
    ax3.set_ylabel(r'$\log$ ('+titlemodel+' / '+title_den+')', va='center', size=25, labelpad=20)
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
    label_patch1 = Line2D([0], [0], marker=None, color='orange', label=textstr_patch1, lw=4, linestyle='--')
    label_patch2 = Line2D([0], [0], marker=None, color='blue', label=textstr_patch2, lw=4, linestyle=':')
    legend_elements = [label_patch1, label_patch2]
    L1 = ax3.legend(handles=legend_elements[0:1], loc=[0, 0.78], fancybox=True, framealpha=0.7)
    ax3.add_artist(L1)
    ax3.legend(handles=legend_elements[1:], loc=[0.38, 0.78])
    if len(height_ratios)==3:
        plot_shapsubplot(df_shap_test, df_test, ax4, fig, tickstoplot)
    plt.tight_layout()
    #
    plt.savefig(label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
    plt.savefig(label_list[label_str]+'_snr=%d'%snr+'_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)


'''
#ax.set_title(label_list[label_str], fontsize=25)
savename_add = '_shap_chain_ngb' if CHAIN_FLAG else '_shap_ngb'
savename_add2 = savename_add + '_SNR_%d.png'%int(1/x_noise)
savename_add += '_SNR_%d.pdf'%int(1/x_noise)
'''


labels_ace = [label_ace_mirk, label_ace_mirk_uncal, label_ace_prosp]
labels_is = [label_is_mirk, label_is_mirk_uncal, label_is_prosp]
model_names = ['Mirkwood', 'Mirkwood,\nuncalibrated', 'Traditional\nSED Fitting']
clrs = ['blue', 'green', 'orange']

make_plot(snr=snr, label_str=label_str, model_names=model_names, crls=clrs, labels_ace=labels_ace, labels_is=labels_is, titletrue='M$^{\star}_{\mathrm{true}}$', titlemodel='M$^{\star}_{\mathrm{model}}$', title_den='M$_{\odot}$', isthisdust=False, df_shap_test=df_shap_test, df_test=df_test, figsize=(10.5,21), height_ratios=[4,1,2], tickstoplot='even')

##########################################################
#################### for dust mass #######################
snr=5
timestr = '20200909'

label_str = 1
uncal_filename = 'uncal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_NATGRADFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(NATGRAD_FLAG))+timestr+'.csv'
cal_filename = 'cal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_NATGRADFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(NATGRAD_FLAG))+timestr+'.csv'
shapmean_filename = 'shapmea_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_NATGRADFLAG=%s_'%(label_list[label_str], train_data, test_data, int(snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(NATGRAD_FLAG))+timestr+'.csv'

gilda_file = pd.concat((pd.read_csv(uncal_filename, index_col=0).drop(columns=['pred_std_epis']), pd.read_csv(cal_filename, index_col=0).drop(columns=['true', 'pred_std_epis'])), axis=1)
lower_file = combined_prosp[['true_dust_mass', 'est_dustmass']].copy()
idx0 = np.where(lower_file['true_dust_mass'].values>=gilda_file['true'].values.min())[0]
idx1 = np.where(lower_file['true_dust_mass'].values<=gilda_file['true'].values.max())[0]
idx_common = np.intersect1d(idx0, idx1)
lower_file = lower_file.loc[idx_common].copy()
lower_file.reset_index(inplace=True, drop=True)

gilda_file_just_means = pd.DataFrame()
gilda_file_just_means['true'] = np.log10(1+gilda_file['true'].copy())
gilda_file_just_means['pred_mean'] = np.log10(1+gilda_file['pred_mean_cal'].copy())
gilda_file_just_means['flag']='mirkwood'
lower_file_just_means = pd.DataFrame()
lower_file_just_means['true'] = np.log10(1+lower_file['true_dust_mass'].copy())
lower_file_just_means['pred_mean'] = np.log10(1+lower_file['est_dustmass'].copy())
lower_file_just_means['flag']='Non-Parametric'
joint_file=pd.concat([gilda_file_just_means, lower_file_just_means]).reset_index(drop=True)

label_ace_mirk = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,4].values, gilda_file.iloc[:,5].values, gilda_file.iloc[:,6].values)) + 0.6827

label_is_mirk = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,4].values, gilda_file.iloc[:,5].values, gilda_file.iloc[:,6].values))

label_ace_mirk_uncal = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values)) + 0.6827

label_is_mirk_uncal = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))

labels_ace = [label_ace_mirk, label_ace_mirk_uncal, label_ace_mirk_uncal]
labels_is = [label_is_mirk, label_is_mirk_uncal, label_is_mirk_uncal]
model_names = ['Mirkwood', 'Mirkwood,\nuncalibrated', None]
clrs = ['blue', 'green', 'white']

ytestattrib_filename = 'shapmea_label=Mass_TRAININGDATA=[\'simba\', \'eagle\', \'tng\']_TESTINGDATA=None_SNR=5_NUMBS=24_WEIGHTFLAG=False_nfoldsval=5_nfoldstest=5_xtrans=False_ytrans=True_MEDIANFLAG=True_NATGRADFLAG=True_20201013.csv'
ytestattrib_std_filename = 'shapstd_label=Mass_TRAININGDATA=[\'simba\', \'eagle\', \'tng\']_TESTINGDATA=None_SNR=5_NUMBS=24_WEIGHTFLAG=False_nfoldsval=5_nfoldstest=5_xtrans=False_ytrans=True_MEDIANFLAG=True_NATGRADFLAG=True_20201013.csv'
df_test_filename = 'uncal_label=Mass_TRAININGDATA=[\'simba\',\'eagle\',\'tng\']_TESTINGDATA=None_SNR=5_NUMBS=24_WEIGHTFLAG=False_nfoldsval=5_nfoldstest=5_xtrans=False_ytrans=True_MEDIANFLAG=True_NATGRADFLAG=True_20201013.csv'
df_test, _ = get_data(train_data)
df_test.columns = central_wav_list
df_shap_test = pd.read_csv(ytestattrib_filename, index_col=0)
df_shap_test_std = pd.read_csv(ytestattrib_std_filename, index_col=0)

make_plot(snr=snr, label_str=label_str, model_names=model_names, crls=clrs, labels_ace=labels_ace, labels_is=labels_is, titletrue='M$^{\mathrm{dust}}_{\mathrm{true}}$', titlemodel='M$^{\mathrm{dust}}_{\mathrm{model}}$', title_den='M$_{\odot}$', isthisdust=True, df_shap_test=df_shap_test, df_test=df_test, figsize=(10.5,21), height_ratios=[4,1,2], tickstoplot='even')

#####################################################################
#######################3# for metallicity ###########################
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

gilda_file_just_means = pd.DataFrame()
gilda_file_just_means['true'] = np.log10(gilda_file['true'].copy())
gilda_file_just_means['pred_mean'] = np.log10(gilda_file['pred_mean_cal'].copy())
gilda_file_just_means['flag']='mirkwood'
lower_file_just_means = pd.DataFrame()
lower_file_just_means['true'] = np.log10(lower_file['true_log(z/zsol)'].copy())
lower_file_just_means['pred_mean'] = np.log10(lower_file['est_log(z/zsol)_50'].copy())
lower_file_just_means['flag']='Non-Parametric'
joint_file=pd.concat([gilda_file_just_means, lower_file_just_means]).reset_index(drop=True)

labels_ace = [label_ace_mirk, label_ace_mirk_uncal, label_ace_prosp]
labels_is = [label_is_mirk, label_is_mirk_uncal, label_is_prosp]
model_names = ['Mirkwood', 'Mirkwood,\nuncalibrated',  'Non-\nparametric']
clrs = ['blue', 'green', 'orange']

make_plot(snr=snr, label_str=label_str, model_names=model_names, crls=clrs, labels_ace=labels_ace, labels_is=labels_is, titletrue='Z$^{\star}_{\mathrm{true}}$', titlemodel='Z$^{\star}_{\mathrm{model}}$', title_den='Z$_{\odot}$', isthisdust=False)


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

label_ace_prosp = ace(lower_file.iloc[:,0].values, (lower_file.iloc[:,1].values, lower_file.iloc[:,2].values, lower_file.iloc[:,3].values)) + 0.6827

label_is_prosp = interval_sharpness(lower_file.iloc[:,0].values, (lower_file.iloc[:,1].values, lower_file.iloc[:,2].values, lower_file.iloc[:,3].values))

label_ace_mirk = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,4].values, gilda_file.iloc[:,5].values, gilda_file.iloc[:,6].values)) + 0.6827

label_is_mirk = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,4].values, gilda_file.iloc[:,5].values, gilda_file.iloc[:,6].values))

label_ace_mirk_uncal = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values)) + 0.6827

label_is_mirk_uncal = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))

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

labels_ace = [label_ace_mirk, label_ace_mirk_uncal, label_ace_prosp]
labels_is = [label_is_mirk, label_is_mirk_uncal, label_is_prosp]
model_names = ['Mirkwood', 'Mirkwood,\nuncalibrated',  'Non-\nparametric']
clrs = ['blue', 'green', 'orange']

make_plot(snr=snr, label_str=label_str, model_names=model_names, crls=clrs, labels_ace=labels_ace, labels_is=labels_is, titletrue='SFR$_{\mathrm{100, true}}$', titlemodel='SFR$_{\mathrm{100, model}}$', title_den='SFR$_{\mathrm{100, \odot}}$', isthisdust=False)


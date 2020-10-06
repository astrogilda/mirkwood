

import seaborn as sns, numpy as np, matplotlib.pyplot as plt, pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams.update({
    "savefig.facecolor": "w",
    "figure.facecolor" : 'w',
    "figure.figsize" : (10,8),
    "text.color": "k",
    "legend.fontsize" : 30,
    "font.size" : 35,
    "axes.edgecolor": "k",
    "axes.labelcolor": "k",
    "axes.linewidth": 4,
    "xtick.color": "k",
    "ytick.color": "k",
    "xtick.labelsize" : 35,
    "ytick.labelsize" : 35,
    "ytick.major.size" : 12,
    "xtick.major.size" : 12,
    "ytick.major.width" : 3,
    "xtick.major.width" : 3,
    "font.family": "STIXGeneral",
    "mathtext.fontset" : "cm"
})
from matplotlib.lines import Line2D # for legend purposes

# black background and white text
plt.style.use('dark_background')
black_facecolor = "#111111"
# also including the facecolor argument in plt.savefig. remove for regular images.


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

ax.set_xlabel(r'Fraction of M$^{\star}_{\mathrm{true}}$ within 1$\sigma$ of M$^{\star}_{\mathrm{model}}$ (ACE + 0.6827)', size=35)
ax2.set_xlabel('Interval Sharpness', labelpad=12, size=35)

ax.grid(False)
ax2.grid(False)

ax.set_facecolor(black_facecolor)
ax2.set_facecolor(black_facecolor)

plt.savefig('barplot_'+label_list[label_str]+'_snr=%d'%snr+'.png', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor=black_facecolor, edgecolor=black_facecolor)
plt.savefig('barplot_'+label_list[label_str]+'_snr=%d'%snr+'.eps', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor=black_facecolor)
plt.savefig('barplot_'+label_list[label_str]+'_snr=%d'%snr+'.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor=black_facecolor)
plt.show()

################

gilda_file_just_means = pd.DataFrame()
gilda_file_just_means['true'] = np.log10(gilda_file['true'].copy())
gilda_file_just_means['pred_mean'] = np.log10(gilda_file['pred_mean_cal'].copy())
gilda_file_just_means['flag']='mirkwood'
lower_file_just_means = pd.DataFrame()
lower_file_just_means['true'] = np.log10(lower_file['true_stellar_mass'].copy())
lower_file_just_means['pred_mean'] = np.log10(lower_file['est_stellar_mass_50'].copy())
lower_file_just_means['flag']='Non-Parametric'
joint_file=pd.concat([gilda_file_just_means, lower_file_just_means]).reset_index(drop=True)

plt.close('all')
x, y, k = joint_file["true"], joint_file["pred_mean"], joint_file["flag"]
g = sns.JointGrid(height=15, space=0)
lp = sns.lineplot(x=gilda_file_just_means.true, y=gilda_file_just_means.true, lw=4, ax=g.ax_joint, color='white')
lp.lines[0].set_linestyle("--")
sns.kdeplot(x=gilda_file_just_means.true, y=gilda_file_just_means.pred_mean, cmap="Oranges", shade=False, thresh=0.1, levels=5, ax=g.ax_joint, linestyles="--", label='mirkwood', linewidths=3)
sns.kdeplot(x=lower_file_just_means.true, y=lower_file_just_means.pred_mean, cmap="Blues", shade=False, thresh=0.1, levels=5, ax=g.ax_joint, linestyles=":", label='Non-Parametric', linewidths=3)
g.ax_joint.set_ylim(bottom=g.ax_joint.get_xlim()[0], top=g.ax_joint.get_xlim()[1])

sns.kdeplot(x=gilda_file_just_means.true, color="orange", legend=False, lw=4, 
             ax=g.ax_marg_x, linestyle="--", fill=False)
sns.kdeplot(x=lower_file_just_means.true, color="blue", legend=False, lw=4, 
             ax=g.ax_marg_x, linestyle=":", fill=False)
sns.kdeplot(y=gilda_file_just_means.pred_mean, color="orange", legend=False, lw=4, ax=g.ax_marg_y, linestyle="--", fill=False)
sns.kdeplot(y=lower_file_just_means.pred_mean, color="blue", legend=False, lw=4, 
             ax=g.ax_marg_y, linestyle=":", fill=False)

g.ax_joint.set_xlabel(r'$\log$ (M$^{\star}_{\rm true}$ / M$_{\odot}$)', ha='center', size=35)
g.ax_joint.set_ylabel(r'$\log$ (M$^{\star}_{\rm model}$ / M$_{\odot}$)', va='center', rotation='vertical', size=35, labelpad=30)
g.ax_marg_x.grid(False)
g.ax_marg_y.grid(False)
g.ax_joint.grid(False)

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

plt.savefig(label_list[label_str]+'_snr=%d'%snr+'.png', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor=black_facecolor, edgecolor=black_facecolor)
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'.eps', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor=black_facecolor, edgecolor=black_facecolor)
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor=black_facecolor, edgecolor=black_facecolor)
plt.show()


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

### bar plot ########

label_ace_mirk = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,4].values, gilda_file.iloc[:,5].values, gilda_file.iloc[:,6].values)) + 0.6827

label_is_mirk = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,4].values, gilda_file.iloc[:,5].values, gilda_file.iloc[:,6].values))

label_ace_mirk_uncal = ace(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values)) + 0.6827

label_is_mirk_uncal = interval_sharpness(gilda_file.iloc[:,0].values, (gilda_file.iloc[:,1].values, gilda_file.iloc[:,2].values, gilda_file.iloc[:,3].values))

model_names = ['Mirkwood', 'Mirkwood,\nuncalibrated']
clrs = ['blue', 'green']
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

ax.barh(np.arange(len(model_names)), (label_ace_mirk, label_ace_mirk_uncal), height=dimw, color=clrs, alpha=1, left=0.001)
ax2.barh(np.arange(len(model_names)) + dimw, (label_is_mirk, label_is_mirk_uncal), height=dimw, color=clrs, alpha=0.4, left=0.001)

ax.set_xlabel(r'Fraction of M$_{\rm{dust, true}}$ within 1$\sigma$ of M$_{\rm{dust, model}}$ (ACE + 0.6827)', size=35)
ax2.set_xlabel('Interval Sharpness', labelpad=12, size=35)

ax.grid(False)
ax2.grid(False)

ax.set_facecolor(black_facecolor)
ax2.set_facecolor(black_facecolor)

plt.savefig('barplot_'+label_list[label_str]+'_snr=%d'%snr+'.png', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor=black_facecolor, edgecolor=black_facecolor)
plt.savefig('barplot_'+label_list[label_str]+'_snr=%d'%snr+'.eps', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor=black_facecolor, edgecolor=black_facecolor)
plt.savefig('barplot_'+label_list[label_str]+'_snr=%d'%snr+'.eps', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor=black_facecolor, edgecolor=black_facecolor)
plt.show()


################

gilda_file_just_means = pd.DataFrame()
gilda_file_just_means['true'] = np.log10(1+gilda_file['true'].copy())
gilda_file_just_means['pred_mean'] = np.log10(1+gilda_file['pred_mean_cal'].copy())
gilda_file_just_means['flag']='mirkwood'
lower_file_just_means = pd.DataFrame()
lower_file_just_means['true'] = np.log10(1+lower_file['true_dust_mass'].copy())
lower_file_just_means['pred_mean'] = np.log10(1+lower_file['est_dustmass'].copy())
lower_file_just_means['flag']='Non-Parametric'
joint_file=pd.concat([gilda_file_just_means, lower_file_just_means]).reset_index(drop=True)

plt.close('all')
x, y, k = joint_file["true"], joint_file["pred_mean"], joint_file["flag"]
g = sns.JointGrid(height=15, space=0)
lp = sns.lineplot(x=gilda_file_just_means.true, y=gilda_file_just_means.true, lw=4, ax=g.ax_joint, color='white')
lp.lines[0].set_linestyle("--")
sns.kdeplot(x=gilda_file_just_means.true, y=gilda_file_just_means.pred_mean, cmap="Oranges", shade=False, thresh=0.1, levels=5, ax=g.ax_joint, linestyles="--", label='mirkwood', linewidths=3)
sns.kdeplot(x=lower_file_just_means.true, y=lower_file_just_means.pred_mean, cmap="Blues", shade=False, thresh=0.1, levels=5, ax=g.ax_joint, linestyles=":", label='Non-Parametric', linewidths=3)
g.ax_joint.set_ylim(bottom=g.ax_joint.get_xlim()[0], top=g.ax_joint.get_xlim()[1])

sns.kdeplot(x=gilda_file_just_means.true, color="orange", legend=False, lw=4, 
             ax=g.ax_marg_x, linestyle="--", fill=False)
sns.kdeplot(x=lower_file_just_means.true, color="blue", legend=False, lw=4, 
             ax=g.ax_marg_x, linestyle=":", fill=False)
sns.kdeplot(y=gilda_file_just_means.pred_mean, color="orange", legend=False, lw=4, 
             ax=g.ax_marg_y, linestyle="--", fill=False)
sns.kdeplot(y=lower_file_just_means.pred_mean, color="blue", legend=False, lw=4, 
             ax=g.ax_marg_y, linestyle=":", fill=False)

g.ax_joint.set_xlabel(r'$\log$ (M$_{\rm{dust, true}}$ / M$_{\odot}$)', ha='center', size=35)
g.ax_joint.set_ylabel(r'$\log$ (M$_{\rm{dust, model}}$ / M$_{\odot}$)', va='center', rotation='vertical', size=35, labelpad=30)
g.ax_joint.grid(False)
g.ax_marg_x.grid(False)
g.ax_marg_y.grid(False)

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

g.ax_joint.set_facecolor(black_facecolor)

plt.savefig(label_list[label_str]+'_snr=%d'%snr+'.png', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor=black_facecolor, edgecolor=black_facecolor)
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'.eps', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor=black_facecolor, edgecolor=black_facecolor)
plt.savefig(label_list[label_str]+'_snr=%d'%snr+'.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor=black_facecolor, edgecolor=black_facecolor)
plt.show()



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
sns.kdeplot(x=gilda_file_just_means.true, y=gilda_file_just_means.pred_mean, cmap="Oranges", shade=False, thresh=0.1, levels=4, ax=g.ax_joint, linestyles="--", label='mirkwood', linewidths=3)
sns.kdeplot(x=lower_file_just_means.true, y=lower_file_just_means.pred_mean, cmap="Blues", shade=False, thresh=0.1, levels=4, ax=g.ax_joint, linestyles=":", label='Non-Parametric', linewidths=3)
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
lower_file_just_means = pd.DataFrame()
lower_file_just_means['true'] = lower_file['true_sfr'].copy()
lower_file_just_means['pred_mean'] = lower_file['est_sfr_50'].copy()
lower_file_just_means = np.log10(lower_file_just_means+1)

lower_file_just_means['flag']='Non-Parametric'
joint_file=pd.concat([gilda_file_just_means, lower_file_just_means]).reset_index(drop=True)

plt.close('all')
x, y, k = joint_file["true"], joint_file["pred_mean"], joint_file["flag"]
g = sns.JointGrid(height=15, space=0)
lp = sns.lineplot(x=gilda_file_just_means.true, y=gilda_file_just_means.true, lw=4, ax=g.ax_joint, color='white')
lp.lines[0].set_linestyle("--")
sns.kdeplot(x=gilda_file_just_means.true, y=gilda_file_just_means.pred_mean, cmap="Oranges", shade=False, thresh=0.1, levels=5, ax=g.ax_joint, linestyles="--", label='mirkwood', linewidths=3)
sns.kdeplot(x=lower_file_just_means.true, y=lower_file_just_means.pred_mean, cmap="Blues", shade=False, thresh=0.1, levels=5, ax=g.ax_joint, linestyles=":", label='Non-Parametric', linewidths=3)
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






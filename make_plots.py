

from scipy.stats import norm
import scipy.stats as st
from six import string_types
from seaborn.palettes import color_palette, blend_palette
import seaborn.distributions as sd
from matplotlib.lines import Line2D  # for legend purposes
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time

import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams.update({
    "savefig.facecolor": "w",
    "figure.facecolor": 'w',
    "figure.figsize": (10, 8),
    "text.color": "k",
    "legend.fontsize": 23,
    "font.size": 25,
    "axes.edgecolor": "k",
    "axes.labelcolor": "k",
    "axes.linewidth": 4,
    "xtick.color": "k",
    "ytick.color": "k",
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "ytick.major.size": 12,
    "xtick.major.size": 12,
    "ytick.major.width": 3,
    "xtick.major.width": 3,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "cm"
})

plt.style.use('seaborn-bright')
timenow = time.strftime("%Y%m%d")

figsize = (8, 8)


# https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python

def contour_plot(x, y, ax, cmap='Blues', colors='blue'):
    # Peform the kernel density estimate
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    #
    # set zi to 0-1 scale
    f = (f-f.min())/(f.max() - f.min())
    f = f.reshape(xx.shape)
    t_contours = [0.05, 0.16, 0.50, 0.84, 0.95]
    # Contour plot
    cset = ax.contour(xx, yy, f, levels=t_contours,
                      colors=colors, origin='lower')
    # Label plot
    ax.clabel(cset, inline=1, fontsize=10)


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
    uncal_filename = 'uncal_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_' % (label_list[label_str], train_data, test_data, int(
        snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
    shapmean_filename = 'shapmea_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_' % (label_list[label_str], train_data, test_data, int(
        snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
    xtrn_filename = 'xtrain_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_' % (label_list[label_str], train_data, test_data, int(
        snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
    xtst_filename = 'xtest_label=%s_TRAININGDATA=%s_TESTINGDATA=%s_SNR=%d_NUMBS=%d_WEIGHTFLAG=%s_nfoldsval=%d_nfoldstest=%d_xtrans=%s_ytrans=%s_MEDIANFLAG=%s_CHAINFLAG=%s_' % (label_list[label_str], train_data, test_data, int(
        snr), NUM_BS, str(WEIGHT_FLAG), int(n_folds_val), int(n_folds_test), str(x_transformer is not None), str(y_transformer is not None), str(MEDIANFLAG), str(CHAIN_FLAG))+timestr+'.csv'
    mean_plot_func = label_func[label_list[label_str]]
    return uncal_filename, shapmean_filename, xtrn_filename, xtst_filename, mean_plot_func


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


def plot_shapsubplot(df_shap_test, df_features_test, axshap, figshap, figname, ticksevenorodd='even'):
    axshap.set_facecolor('white')
    sort = False
    max_display = 40
    feature_names = df_features_test.columns
    features = df_features_test.values
    shap_values = df_shap_test.values
    num_features = len(feature_names)
    if sort:
        feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-max_display:]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)
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
        # make sure this can be numeric
        values = np.array(values, dtype=np.float64)
        N = len(shaps)
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) /
                         (np.max(shaps) - np.min(shaps) + 1e-8))
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
        if vmin > vmax:  # fixes rare numerical precision issues
            vmin = vmax
        # plot the nan values in the interaction feature as grey
        nan_mask = np.isnan(values)
        axshap.scatter(pos + ys[nan_mask], shaps[nan_mask], color="#777777", vmin=vmin,
                       vmax=vmax, s=16, alpha=alpha, linewidth=0, zorder=3, rasterized=len(shaps) > 500)
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
    color_bar_label = ''  # labels["FEATURE_VALUE"]
    m = cm.ScalarMappable(cmap=colors.red_blue)
    m.set_array([0, 1])
    cb = plt.colorbar(m, ticks=[0, 1], ax=axshap)  # , aspect=1000)
    cb.set_ticklabels([labels['FEATURE_VALUE_LOW'],
                      labels['FEATURE_VALUE_HIGH']])
    cb.set_label(color_bar_label, size=25, labelpad=-25)
    cb.ax.tick_params(labelsize=25, length=0)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(figshap.dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.9) * 20)
    axis_color = "#333333"
    max_display = min(len(feature_names), max_display)
    axshap.xaxis.set_ticks_position('bottom')
    axshap.yaxis.set_ticks_position('none')
    axshap.spines['right'].set_visible(False)
    axshap.spines['top'].set_visible(False)
    axshap.spines['left'].set_visible(False)
    axshap.tick_params(color=axis_color, labelcolor=axis_color)
    _ = axshap.set_xticks(range(len(feature_order)))
    _ = axshap.set_xticklabels([feature_names[i]
                               for i in feature_order], fontsize=25, rotation=90)
    axshap.grid(False)
    if ticksevenorodd != 'both':
        for tick in axshap.xaxis.get_major_ticks()[int(ticksevenorodd == 'even')::2]:
            tick.set_visible(False)  # set_pad(100)
    axshap.tick_params('x', length=20, width=0.5, which='major', labelsize=15)
    axshap.tick_params('y', labelsize=25)
    _ = axshap.set_xlim(left=-1, right=len(feature_order))
    _ = axshap.set_ylabel(labels['VALUE'], fontsize=25)
    _ = axshap.set_xlabel(r'Wavelength ($\mu$m)', fontsize=25)
    _ = axshap.invert_xaxis()
    plt.tight_layout()
    plt.savefig(figname+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
    plt.savefig(figname+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
    return

######### COMMON CODE #########


def run_the_damn_code():
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    _ = ax.plot(mean_plot_func(gilda_file.true.values), mean_plot_func(
        gilda_file.true.values), lw=2, color='k', ls='--')
    _ = ax.scatter(x=mean_plot_func(gilda_file.true), y=mean_plot_func(
        gilda_file.pred_mean), color='orange', alpha=0.5)
    if label_str != 3:
        contour_plot(x=mean_plot_func(lower_file.true.values), y=mean_plot_func(
            lower_file.pred_mean.values), ax=ax, cmap='Blues', colors='blue')
    else:
        _ = ax.scatter(x=mean_plot_func(lower_file.true), y=mean_plot_func(
            lower_file.pred_mean), color='blue', alpha=0.2, marker='x')
    #
    ax.set_facecolor('white')
    ax.grid(True)
    _ = ax.set_ylim(bottom=ax.get_xlim()[0], top=ax.get_xlim()[1])
    if len(titletrue) == 1:
        _ = ax.set_xlabel(r'$\log\left(\frac{%s}{%s}\right)$' % (
            str(titletrue[0]), str(title_den)), ha='center', size=25)
        _ = ax.set_ylabel(r'$\log\left(\frac{%s}{%s}\right)$' % (
            str(titlemodel[0]), str(title_den)), ha='center', size=25)
    else:
        _ = ax.set_xlabel(r'$\log\left(%s\frac{%s}{%s}\right)$' % (
            str(titletrue[0]), str(titletrue[1]), str(title_den)), ha='center', size=25)
        _ = ax.set_ylabel(r'$\log\left(%s\frac{%s}{%s}\right)$' % (
            str(titlemodel[0]), str(titlemodel[1]), str(title_den)), ha='center', size=25)
    label_nrmse_patch1 = nrmse(
        gilda_file.true.values, gilda_file.pred_mean.values)
    label_nmae_patch1 = nmae(gilda_file.true.values,
                             gilda_file.pred_mean.values)
    label_bias_patch1 = nbe(gilda_file.true.values,
                            gilda_file.pred_mean.values)
    textstr_patch1 = '\n'.join((
        r'$\bf{Mirkwood}$',
        r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse_patch1, ),
        r'$\mathrm{NMAE}=%.2f$' % (label_nmae_patch1, ),
        r'$\mathrm{NBE}=%.2f$' % (label_bias_patch1, ),
    ))
    label_patch1 = Line2D([0], [0], marker=None, color='orange',
                          label=textstr_patch1, lw=4, linestyle='dotted')
    label_nrmse_patch2 = nrmse(
        lower_file.true.values, lower_file.pred_mean.values)
    label_nmae_patch2 = nmae(lower_file.true.values,
                             lower_file.pred_mean.values)
    label_bias_patch2 = nbe(lower_file.true.values,
                            lower_file.pred_mean.values)
    textstr_patch2 = '\n'.join((
        r'$\bf{Traditional}$'+'\n'+r'$\bf{SED\;Fitting}$',
        r'$\mathrm{NRMSE}=%.2f$' % (label_nrmse_patch2, ),
        r'$\mathrm{NMAE}=%.2f$' % (label_nmae_patch2, ),
        r'$\mathrm{NBE}=%.2f$' % (label_bias_patch2, ),
    ))
    label_patch2 = Line2D([0], [0], marker=None, color='blue',
                          label=textstr_patch2, lw=4, linestyle='solid')
    legend_elements = [label_patch1, label_patch2]
    #
    L1 = ax.legend(handles=legend_elements[0:1], loc=[
                   0, 0.68], fancybox=True, framealpha=0.7, fontsize=18)
    ax.add_artist(L1)
    ax.legend(handles=legend_elements[1:],
              loc=[.55, 0.05], framealpha=0.7, fontsize=18)
    #
    plt.tight_layout()
    plt.savefig(label_list[label_str]+'_snr=%d' % snr+'_'+timenow +
                '.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
    plt.savefig(label_list[label_str]+'_snr=%d' % snr+'_'+timenow +
                '.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
    ####### Calibration Plot ################################
    u_gilda = cdf_normdist(y=gilda_file.true.values, loc=gilda_file.pred_mean.values,
                           scale=.5*(gilda_file.pred_upper-gilda_file.pred_lower).values)
    label_ace_mirk = ace(gilda_file.iloc[:, 0].values, (
        gilda_file.iloc[:, 1].values, gilda_file.iloc[:, 2].values, gilda_file.iloc[:, 3].values))
    label_is_mirk = interval_sharpness(gilda_file.iloc[:, 0].values, (
        gilda_file.iloc[:, 1].values, gilda_file.iloc[:, 2].values, gilda_file.iloc[:, 3].values))
    #
    if label_str not in [1]:  # [0,2]:
        u_lower = cdf_normdist(y=lower_file.iloc[:, 0].values, loc=lower_file.iloc[:, 1].values, scale=.5*(
            lower_file.iloc[:, 3].values-lower_file.iloc[:, 2].values))
        label_ace_lower = ace(lower_file.iloc[:, 0].values, (
            lower_file.iloc[:, 1].values, lower_file.iloc[:, 2].values, lower_file.iloc[:, 3].values))
        label_is_lower = interval_sharpness(lower_file.iloc[:, 0].values, (
            lower_file.iloc[:, 1].values, lower_file.iloc[:, 2].values, lower_file.iloc[:, 3].values))
    #
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_facecolor = 'white'
    ax.grid(True)
    ax.plot(np.linspace(0, 1, len(u_gilda)), np.sort(u_gilda),
            ls='-', color='orange', lw=5, label='Mirkwood')
    if label_str not in [1]:  # 0,2]:
        ax.plot(np.linspace(0, 1, len(u_lower)), np.sort(u_lower),
                ls='-', color='blue', lw=5, label='Traditional')
    #
    ax.plot(np.linspace(0, 1, len(u_gilda)), np.linspace(
        0, 1, len(u_gilda)), lw=4, color='k', ls='--')
    ax.set_xlabel(r'Expected Confidence Level', ha='center', size=25)
    ax.set_ylabel(r'Observed Confidence Level', ha='center', size=25)
    #
    textstr_patch1 = '\n'.join((
        r'$\bf{Mirkwood}$',
        r'$\mathrm{ACE}=%.2f$' % (label_ace_mirk, ),
        r'$\mathrm{IS}=%.2f$' % (label_is_mirk, ),
    ))
    label_patch1 = Line2D([0], [0], marker=None, color='orange',
                          label=textstr_patch1, lw=4, linestyle='solid')
    if label_str not in [1]:  # [0,2]:
        textstr_patch2 = '\n'.join((
            r'$\bf{Traditional}$'+'\n'+r'$\bf{SED\;Fitting}$',
            r'$\mathrm{ACE}=%.2f$' % (label_ace_lower, ),
            r'$\mathrm{IS}=%.2f$' % (label_is_lower, ),
        ))
        label_patch2 = Line2D([0], [0], marker=None, color='blue',
                              label=textstr_patch2, lw=4, linestyle='solid')
    #
    if label_str not in [1]:  # [0,2]:
        legend_elements = [label_patch1, label_patch2]
    else:
        legend_elements = [label_patch1]
    #
    L1 = ax.legend(handles=legend_elements[0:1], loc=[
                   0, 0.68], fancybox=True, framealpha=0.7, fontsize=18)
    ax.add_artist(L1)
    if label_str not in [1]:  # [0,2]:
        ax.legend(
            handles=legend_elements[1:], loc=[.58, 0.05], framealpha=0.7, fontsize=18)
    #
    plt.tight_layout()
    plt.savefig('Calibration_'+label_list[label_str]+'_snr=%d' % snr +
                '_'+timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
    plt.savefig('Calibration_'+label_list[label_str]+'_snr=%d' % snr +
                '_'+timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
    # plt.show()
    ###### Histogram #########################################
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    _ = ax.hist(mean_plot_func(gilda_file.pred_mean.values) - mean_plot_func(gilda_file.true.values),
                color='orange', label=r'$\bf{Mirkwood}$', alpha=1, density=False, lw=2, bins=40)
    _ = ax.hist(mean_plot_func(lower_file.pred_mean.values) - mean_plot_func(lower_file.true.values), color='blue',
                label=r'$\bf{Traditional}$'+'\n'+r'$\bf{SED\;Fitting}$', alpha=1, density=False, lw=2, histtype='step', bins=40)
    _ = ax.axvline(x=0, lw=2, ls='--', color='k')
    if len(titletrue) == 1:
        num = titlemodel[0]
        den = titletrue[0]
        _ = ax.set_xlabel(r'$\log\left(\frac{%s}{%s}\right)$' % (
            str(num), str(den)), ha='center', size=25)
    else:
        num = titletrue[0] + titlemodel[1]
        den = titletrue[0] + titletrue[1]
        _ = ax.set_xlabel(r'$\log\left(\frac{%s}{%s}\right)$' % (
            str(num), str(den)), ha='center', size=25)
    _ = ax.set_ylabel('# Galaxies', va='center', size=25, labelpad=20)
    # fixing the legend
    handles, labels = ax.get_legend_handles_labels()
    colors = ['orange', 'blue']
    new_handles = [Line2D([], [], c=colors[i]) for i, h in enumerate(handles)]
    _ = ax.legend(handles=new_handles, labels=labels, loc='upper right')
    plt.tight_layout()
    plt.savefig('Hist_'+label_list[label_str]+'_snr=%d' % snr+'_' +
                timenow+'.png', bbox_inches='tight', pad_inches=0.001, dpi=300)
    plt.savefig('Hist_'+label_list[label_str]+'_snr=%d' % snr+'_' +
                timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=300)
    ###### SHAP values #######################################
    df_test = pd.read_csv(xtst_filename, index_col=0)
    df_train = pd.read_csv(xtrn_filename, index_col=0)
    for i, j in zip(list(df_test), central_wav_list):
        df_test.rename(columns={i: j}, inplace=True)
        df_train.rename(columns={i: j}, inplace=True)
    #
    df_shap_test = pd.read_csv(shapmean_filename, index_col=0)
    if 'z' in list(df_shap_test):
        df_shap_test.drop(columns='z', inplace=True)
        df_test.drop(columns='z', inplace=True)
    #
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    figname = 'SHAP_'+label_list[label_str]+'_snr=%d' % snr+'_'+timenow
    plot_shapsubplot(df_shap_test, df_test, ax, fig,
                     figname, ticksevenorodd='even')
    #####################################################################
    # metrics #####
    metrics_mkwd = [label_nrmse_patch1, label_nmae_patch1,
                    label_bias_patch1, label_ace_mirk, label_is_mirk]
    if label_str != 1:
        metrics_trad = [label_nrmse_patch2, label_nmae_patch2,
                        label_bias_patch2, label_ace_lower, label_is_lower]
    else:
        metrics_trad = [label_nrmse_patch2, label_nmae_patch2,
                        label_bias_patch2, np.nan, np.nan]
    # extended metrics ######
    label_ace_mirk_epis = ace(gilda_file.iloc[:, 0].values, (gilda_file.iloc[:, 1].values, gilda_file.iloc[:,
                              1].values - gilda_file.iloc[:, 4].values, gilda_file.iloc[:, 1].values + gilda_file.iloc[:, 4].values))
    label_is_mirk_epis = interval_sharpness(gilda_file.iloc[:, 0].values, (gilda_file.iloc[:, 1].values, gilda_file.iloc[:,
                                            1].values - gilda_file.iloc[:, 4].values, gilda_file.iloc[:, 1].values + gilda_file.iloc[:, 4].values))
    total_lower = gilda_file.iloc[:, 1].values - np.sqrt(
        (gilda_file.iloc[:, 2].values - gilda_file.iloc[:, 2].values)**2 + gilda_file.iloc[:, 4].values**2)
    total_upper = gilda_file.iloc[:, 1].values + np.sqrt(
        (gilda_file.iloc[:, 3].values - gilda_file.iloc[:, 1].values)**2 + gilda_file.iloc[:, 4].values**2)
    label_ace_mirk_total = ace(
        gilda_file.iloc[:, 0].values, (gilda_file.iloc[:, 1].values, total_lower, total_upper))
    label_is_mirk_total = interval_sharpness(
        gilda_file.iloc[:, 0].values, (gilda_file.iloc[:, 1].values, total_lower, total_upper))
    metrics_mkwd_extended = [label_nrmse_patch1, label_nmae_patch1, label_bias_patch1, label_ace_mirk,
                             label_is_mirk, label_ace_mirk_epis, label_is_mirk_epis, label_ace_mirk_total, label_is_mirk_total]
    if label_str != 1:
        metrics_trad_extended = [label_nrmse_patch2, label_nmae_patch2, label_bias_patch2,
                                 np.nan, np.nan, np.nan, np.nan, label_ace_lower, label_is_lower]
    else:
        metrics_trad_extended = [label_nrmse_patch2, label_nmae_patch2,
                                 label_bias_patch2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    return metrics_trad, metrics_mkwd, metrics_trad_extended, metrics_mkwd_extended


# create fake dataframes
list_of_metrics = ["RMSE", "MAE", "BE", "ACE", "IS"]

df1 = pd.DataFrame(np.zeros((5, 2))*np.nan,
                   index=list_of_metrics,
                   columns=["Mirkwood", "Traditional"])

df_Mass_snr5 = df1.copy()
df_Dust_snr5 = df1.copy()
df_Z_snr5 = df1.copy()
df_SFR_snr5 = df1.copy()

df_Mass_snr10 = df1.copy()
df_Dust_snr10 = df1.copy()
df_Z_snr10 = df1.copy()
df_SFR_snr10 = df1.copy()

df_Mass_snr20 = df1.copy()
df_Dust_snr20 = df1.copy()
df_Z_snr20 = df1.copy()
df_SFR_snr20 = df1.copy()

df_dict = {'df_Mass_snr5': df_Mass_snr5, 'df_Dust_snr5': df_Dust_snr5, 'df_Z_snr5': df_Z_snr5, 'df_SFR_snr5': df_SFR_snr5, 'df_Mass_snr10': df_Mass_snr10, 'df_Dust_snr10': df_Dust_snr10,
           'df_Z_snr10': df_Z_snr10, 'df_SFR_snr10': df_SFR_snr10, 'df_Mass_snr20': df_Mass_snr20, 'df_Dust_snr20': df_Dust_snr20, 'df_Z_snr20': df_Z_snr20, 'df_SFR_snr20': df_SFR_snr20}

# *np.nan#np.array([[1,2,3,4,5], [0.1,0.2,0.3,0.4,0.5], [0.2,0.4,0.6,0.8,1.0], [0.3,0.6,0.9,1.2,1.5], [0.4,0.8,1.2,1.6,2.0], [0.5,1.,1.5,2.,2.5]])
q = np.zeros((6, 9))
df_mass_ood_snr5 = pd.DataFrame(q, columns=['nmae', 'nrmse', 'nbe', 'ace_a', 'ints_a',
                                'ace_e', 'ints_e', 'ace', 'ints'], index=['P', 'SET', 'ET', 'T', 'E', 'S'])
df_dustmass_ood_snr5 = df_mass_ood_snr5.copy()
df_Z_ood_snr5 = df_mass_ood_snr5.copy()
df_sfr_ood_snr5 = df_mass_ood_snr5.copy()

df_mass_ood_snr10 = df_mass_ood_snr5.copy()
df_dustmass_ood_snr10 = df_mass_ood_snr5.copy()
df_Z_ood_snr10 = df_mass_ood_snr5.copy()
df_sfr_ood_snr10 = df_mass_ood_snr5.copy()

df_mass_ood_snr20 = df_mass_ood_snr5.copy()
df_dustmass_ood_snr20 = df_mass_ood_snr5.copy()
df_Z_ood_snr20 = df_mass_ood_snr5.copy()
df_sfr_ood_snr20 = df_mass_ood_snr5.copy()

df_ood_dict = {'df_Mass_ood_snr5': df_mass_ood_snr5, 'df_Dust_ood_snr5': df_dustmass_ood_snr5, 'df_Z_ood_snr5': df_Z_ood_snr5, 'df_SFR_ood_snr5': df_sfr_ood_snr5, 'df_Mass_ood_snr10': df_mass_ood_snr10, 'df_Dust_ood_snr10': df_dustmass_ood_snr10,
               'df_Z_ood_snr10': df_Z_ood_snr10, 'df_SFR_ood_snr10': df_sfr_ood_snr10, 'df_Mass_ood_snr20': df_mass_ood_snr20, 'df_Dust_ood_snr20': df_dustmass_ood_snr20, 'df_Z_ood_snr20': df_Z_ood_snr20, 'df_SFR_ood_snr20': df_sfr_ood_snr20}
##########################################################################

snr_list = [5, 10, 20]
# 'S': training set is Simba
# 'E': ... Eagle
# 'T': ... Illustris-TNG
TRAIN_SET_ABV_list = ['S', 'E', 'T', 'ET', 'SET']

test_data = ['simba']
for TRAIN_SET_ABV in TRAIN_SET_ABV_list:
    if TRAIN_SET_ABV == 'ET':
        NUM_BS, timestr, train_data, test_data = 24, '20201204', [
            'eagle', 'tng'], ['simba']
    elif TRAIN_SET_ABV == 'E':
        NUM_BS, timestr, train_data, test_data = 24, '20201206', [
            'eagle'], ['simba']
    elif TRAIN_SET_ABV == 'T':
        NUM_BS, timestr, train_data, test_data = 24, '20201207', [
            'tng'], ['simba']
    elif TRAIN_SET_ABV == 'S':
        NUM_BS, timestr, train_data, test_data = 22, '20201218', [
            'simba'], None
    elif TRAIN_SET_ABV == 'SET':
        NUM_BS, timestr, train_data, test_data = 22, '20201217', [
            'simba', 'eagle', 'tng'], ['simba']
    #
    print('training_set = %s' % TRAIN_SET_ABV)
    for snr in snr_list:
        if TRAIN_SET_ABV == 'ET':
            snr = 5
        print('SNR = %d' % snr)
        # get Prospector results from Sidney's files #############\
        simba_prosp = pd.read_pickle('simba_snr%d.pkl' % snr)
        eagle_prosp = pd.read_pickle('eagle_snr%d.pkl' % snr)
        tng_prosp = pd.read_pickle('tng_snr%d.pkl' % snr)
        combined_prosp = simba_prosp.copy()
        ### for mass ##########
        # Generating Figure 3
        label_str = 0
        uncal_filename, shapmean_filename, xtrn_filename, xtst_filename, mean_plot_func = update_filenames(
            label_str)
        #
        # .drop(columns=['pred_std_epis'])
        gilda_file = pd.read_csv(uncal_filename, index_col=0)
        lower_file = combined_prosp[[
            'true_stellar_mass', 'est_stellar_mass_50', 'est_stellar_mass_16', 'est_stellar_mass_84']].copy()
        lower_file.rename(columns={"true_stellar_mass": "true", "est_stellar_mass_50": "pred_mean",
                          "est_stellar_mass_16": "pred_lower", "est_stellar_mass_84": "pred_upper"}, inplace=True)
        idx0 = np.where(lower_file.true.values >=
                        gilda_file.true.values.min())[0]
        idx1 = np.where(lower_file.true.values <=
                        gilda_file.true.values.max())[0]
        idx_common = np.intersect1d(idx0, idx1)
        lower_file = lower_file.loc[idx_common].copy()
        lower_file.reset_index(inplace=True, drop=True)
        lower_file.pred_mean = 10**lower_file.pred_mean
        lower_file.pred_lower = 10**lower_file.pred_lower
        lower_file.pred_upper = 10**lower_file.pred_upper
        gilda_file = gilda_file.apply(pd.to_numeric)
        lower_file = lower_file.apply(pd.to_numeric)
        #
        titletrue = ['M^{\star}_{\mathrm{true}}']
        titlemodel = ['M^{\star}_{\mathrm{model}}']
        title_den = 'M_{\odot}'
        #
        metrics_trad, metrics_mkwd, metrics_trad_extended, metrics_mkwd_extended = run_the_damn_code()
        ### after running the common code ###
        df_dict['df_%s_snr%d' % (label_list[label_str], snr)
                ].loc[:, 'Traditional'] = metrics_trad
        df_dict['df_%s_snr%d' %
                (label_list[label_str], snr)].loc[:, 'Mirkwood'] = metrics_mkwd
        #
        df_ood_dict['df_%s_ood_snr%d' %
                    (label_list[label_str], snr)].loc['P'] = metrics_trad_extended
        df_ood_dict['df_%s_ood_snr%d' % (
            label_list[label_str], snr)].loc[TRAIN_SET_ABV] = metrics_mkwd_extended
        print('mass done')
        ### for dust mass ##########
        # Generating Figure 4
        label_str = 1
        uncal_filename, shapmean_filename, xtrn_filename, xtst_filename, mean_plot_func = update_filenames(
            label_str)
        #
        # .drop(columns=['pred_std_epis'])
        gilda_file = pd.read_csv(uncal_filename, index_col=0)
        lower_file = combined_prosp[['true_dust_mass', 'est_dustmass']].copy()
        lower_file.rename(
            columns={"true_dust_mass": "true", "est_dustmass": "pred_mean"}, inplace=True)
        idx0 = np.where(lower_file.true.values >=
                        gilda_file.true.values.min())[0]
        idx1 = np.where(lower_file.true.values <=
                        gilda_file.true.values.max())[0]
        idx_common = np.intersect1d(idx0, idx1)
        lower_file = lower_file.loc[idx_common].copy()
        lower_file.reset_index(inplace=True, drop=True)
        #
        gilda_file = gilda_file.apply(pd.to_numeric)
        lower_file = lower_file.apply(pd.to_numeric)
        #
        titletrue = ['1 + ', 'M_{\mathrm{dust, true}}']
        titlemodel = ['1 + ', 'M_{\mathrm{dust, model}}']
        title_den = 'M_{\odot}'
        #
        metrics_trad, metrics_mkwd, metrics_trad_extended, metrics_mkwd_extended = run_the_damn_code()
        #
        ### after running the common code ###
        df_dict['df_%s_snr%d' % (label_list[label_str], snr)
                ].loc[:, 'Traditional'] = metrics_trad
        df_dict['df_%s_snr%d' %
                (label_list[label_str], snr)].loc[:, 'Mirkwood'] = metrics_mkwd
        #
        df_ood_dict['df_%s_ood_snr%d' %
                    (label_list[label_str], snr)].loc['P'] = metrics_trad_extended
        df_ood_dict['df_%s_ood_snr%d' % (
            label_list[label_str], snr)].loc[TRAIN_SET_ABV] = metrics_mkwd_extended
        print('dustmass done')
        #######################################
        ### for metallicity #####
        # Generating Figure 5
        label_str = 2
        uncal_filename, shapmean_filename, xtrn_filename, xtst_filename, mean_plot_func = update_filenames(
            label_str)
        gilda_file = pd.read_csv(uncal_filename, index_col=0)
        lower_file = 10**(combined_prosp[['true_log(z/zsol)', 'est_log(z/zsol)_50',
                          'est_log(z/zsol)_16', 'est_log(z/zsol)_84']].copy())
        lower_file.rename(columns={"true_log(z/zsol)": "true", "est_log(z/zsol)_50": "pred_mean",
                          "est_log(z/zsol)_16": "pred_lower", "est_log(z/zsol)_84": "pred_upper"}, inplace=True)
        idx0 = np.where(lower_file.loc[:, list(lower_file)[0]].copy(
        ).values >= gilda_file['true'].values.min())[0]
        idx1 = np.where(lower_file.loc[:, list(lower_file)[0]].copy(
        ).values <= gilda_file['true'].values.max())[0]
        idx_common = np.intersect1d(idx0, idx1)
        lower_file = lower_file.loc[idx_common].copy()
        lower_file.reset_index(inplace=True, drop=True)
        gilda_file = gilda_file.apply(pd.to_numeric)
        lower_file = lower_file.apply(pd.to_numeric)
        #
        titletrue = ['Z^{\star}_{\mathrm{true}}']
        titlemodel = ['Z^{\star}_{\mathrm{model}}']
        title_den = 'Z_{\odot}'
        #
        metrics_trad, metrics_mkwd, metrics_trad_extended, metrics_mkwd_extended = run_the_damn_code()
        ### after running the common code ###
        df_dict['df_%s_snr%d' % (label_list[label_str], snr)
                ].loc[:, 'Traditional'] = metrics_trad
        df_dict['df_%s_snr%d' %
                (label_list[label_str], snr)].loc[:, 'Mirkwood'] = metrics_mkwd
        #
        df_ood_dict['df_%s_ood_snr%d' %
                    (label_list[label_str], snr)].loc['P'] = metrics_trad_extended
        df_ood_dict['df_%s_ood_snr%d' % (
            label_list[label_str], snr)].loc[TRAIN_SET_ABV] = metrics_mkwd_extended
        print('metallicity done')
        ############# 3### for SFR #############
        # Generating Figure 6
        label_str = 3
        uncal_filename, shapmean_filename, xtrn_filename, xtst_filename, mean_plot_func = update_filenames(
            label_str)
        gilda_file = pd.read_csv(uncal_filename, index_col=0)
        lower_file = combined_prosp[[
            'true_sfr', 'est_sfr_50', 'est_sfr_16', 'est_sfr_84']].copy()
        idx0 = np.where(lower_file['true_sfr'].values >=
                        gilda_file['true'].values.min())[0]
        idx1 = np.where(lower_file['true_sfr'].values <=
                        gilda_file['true'].values.max())[0]
        idx_common = np.intersect1d(idx0, idx1)
        lower_file = lower_file.loc[idx_common].copy()
        lower_file.reset_index(inplace=True, drop=True)
        lower_file.rename(columns={"true_sfr": "true", "est_sfr_50": "pred_mean",
                          "est_sfr_16": "pred_lower", "est_sfr_84": "pred_upper"}, inplace=True)
        gilda_file = gilda_file.apply(pd.to_numeric)
        lower_file = lower_file.apply(pd.to_numeric)
        #
        titletrue = ['1 + ', r'SFR_{\mathrm{100, true}}']
        titlemodel = ['1 + ', r'SFR_{\mathrm{100, model}}']
        title_den = 'M_{\odot} \mathrm{yr}^{-1}'
        #
        metrics_trad, metrics_mkwd, metrics_trad_extended, metrics_mkwd_extended = run_the_damn_code()
        ### after running the common code ###
        df_dict['df_%s_snr%d' % (label_list[label_str], snr)
                ].loc[:, 'Traditional'] = metrics_trad
        df_dict['df_%s_snr%d' %
                (label_list[label_str], snr)].loc[:, 'Mirkwood'] = metrics_mkwd
        #
        df_ood_dict['df_%s_ood_snr%d' %
                    (label_list[label_str], snr)].loc['P'] = metrics_trad_extended
        df_ood_dict['df_%s_ood_snr%d' % (
            label_list[label_str], snr)].loc[TRAIN_SET_ABV] = metrics_mkwd_extended
        print('SFR done')


################################################
# Generating Table 4
label_str = 2
df1 = df_dict['df_%s_snr%s' % (label_list[label_str], 5)]
df2 = df_dict['df_%s_snr%s' % (label_list[label_str], 10)]
df3 = df_dict['df_%s_snr%s' % (label_list[label_str], 20)]

dfall = [df1, df2, df3]


def pandas_to_latex(dfall):
    df_new = pd.DataFrame(
        np.ones((3, len(list(dfall[0].T)))), columns=list(dfall[0].T)) .astype(object)
    for i in range(len(dfall)):
        df = dfall[i].T
        for feature in list(df):
            df_new.loc[i, feature] = tuple(np.round(df[feature].values, 3))
    df_new.index = ['5', '10', '20']
    return df_new


df_new = pandas_to_latex(dfall)
print(df_new.to_latex(index=True, multicolumn_format='c'))

######################################################
####### dataset summary ##############################
# Generating data for Table 1 ########################

q, q2 = get_data(['tng'])[1], get_data(['tng2'])[1]
for i, j in zip(q, q2):
    round(np.min(i), 2), round(np.min(j), 2)
    round(np.max(i), 2), round(np.max(j), 2)
    round(np.mean(i), 2), round(np.mean(j), 2)
    round(np.median(i), 2), round(np.median(j), 2)
    round(np.std(i), 2), round(np.std(j), 2)
    print('###################')

####################################################
### 12-06-2020. trying to plot MAE and RMSE on two y axes, with colors denoted by a colorbar############################################
# for MASS. 3x2. each row is a single SNR. left column is MAE, RMSE on the two y-axes, colorbar is bias. right colum is ace and is on the two y-axes.
# this is what the pandas df looks like. property, snr.
#################### NMAE, NRMSE, NBE, ACE, IS##############################
# p(for prospector)
# train:e,t,s; test:s
# train:e,t; test:s
# train:t, test:s
# train:e, test:s
# train:s, test:s
# Plotting figures 8, 9, 10, and 11.


def ood_plot(df):
    import matplotlib
    matplotlib.use('Qt5Agg')
    matplotlib.rcParams.update({
        "savefig.facecolor": "w",
        "figure.facecolor": 'w',
        "figure.figsize": (10, 8),
        "text.color": "k",
        "legend.fontsize": 13,
        "font.size": 15,
        "axes.edgecolor": "k",
        "axes.labelcolor": "k",
        "axes.linewidth": 4,
        "xtick.color": "k",
        "ytick.color": "k",
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "ytick.major.size": 12,
        "xtick.major.size": 12,
        "ytick.major.width": 3,
        "xtick.major.width": 3,
        "font.family": "STIXGeneral",
        "mathtext.fontset": "cm"
    })
    #
    # left plot
    cmap = matplotlib.cm.jet
    plt.close('all')
    fig, ax_all = plt.subplots(2, 4, figsize=(18, 6), gridspec_kw={
                               'width_ratios': [1., 1, 1, 1]}, constrained_layout=True)
    #
    ax, ax2 = ax_all[0, 0], ax_all[1, 0]
    ax_prob = ax_all[:, 1:]
    #
    cb = ax.scatter(df.index.values, df.nmae.values,
                    c=df.nbe.values, cmap=cmap, s=100)  # , s=12)
    cb2 = ax2.scatter(df.index.values, df.nrmse.values,
                      c=df.nbe.values, cmap=cmap, s=100)  # , s=12)#, s=4)
    _ = ax.axhline(y=0, lw=2, ls='--', color='darkgreen')
    _ = ax2.axhline(y=0, lw=2, ls='--', color='darkgreen')
    _ = ax.annotate("Better", xy=(np.median(ax.get_xlim()), 0.02), xytext=(-25, 50), arrowprops=dict(arrowstyle='->',
                    lw=4, color='darkgreen'), va='center', textcoords='offset points', color='darkgreen', fontsize=20)
    _ = ax2.annotate("Better", xy=(np.median(ax.get_xlim()), 0.02), xytext=(-25, 50), arrowprops=dict(arrowstyle='->',
                     lw=4, color='darkgreen'), va='center', textcoords='offset points', color='darkgreen', fontsize=20)
    cbar = plt.colorbar(mappable=cb, pad=0.1,
                        orientation='vertical', aspect=50)
    cbar.ax.plot(np.median(df.nbe.values), df.nbe.values[0], 'w.')
    cbar.set_label('NBE', size=20)
    ax.set_ylabel('NMAE', size=20)
    ax2.set_ylabel('NRMSE', size=20)
    ax.grid(False)
    ax2.grid(False)
    # right plot
    axr, axr2 = ax_prob[0, 0], ax_prob[1, 0]
    cb = axr.scatter(df.index.values, df.ace_a.values, s=100,
                     color='red', alpha=0.5)  # , s=12)
    cb2 = axr2.scatter(df.index.values, df.ints_a.values,
                       s=100, color='red', alpha=0.5)  # , s=12)
    _ = axr.axhline(y=0, lw=2, ls='--', color='darkgreen')
    _ = axr2.axhline(y=0, lw=2, ls='--', color='darkgreen')
    _ = axr.annotate("Better", xy=(np.quantile(ax.get_xlim(), 0.67), -0.02), xytext=(-25, -50), arrowprops=dict(
        arrowstyle='->', lw=4, color='darkgreen'), va='center', textcoords='offset points', color='darkgreen', fontsize=20)
    _ = axr2.annotate("Better", xy=(np.median(ax.get_xlim()), -0.02), xytext=(-25, -50), arrowprops=dict(arrowstyle='->',
                      lw=4, color='darkgreen'), va='center', textcoords='offset points', color='darkgreen', fontsize=20)
    axr.set_ylabel(r'ACE$_{\mathrm{al}}$', size=20)
    axr2.set_ylabel(r'IS$_{\mathrm{al}}$', size=20)
    axr.grid(False)
    axr2.grid(False)
    # right plot
    axr, axr2 = ax_prob[0, 1], ax_prob[1, 1]
    cb = axr.scatter(df.index.values, df.ace_e.values, s=100,
                     color='red', alpha=0.5)  # , s=12)
    cb2 = axr2.scatter(df.index.values, df.ints_e.values,
                       s=100, color='red', alpha=0.5)  # , s=12)
    _ = axr.axhline(y=0, lw=2, ls='--', color='darkgreen')
    _ = axr2.axhline(y=0, lw=2, ls='--', color='darkgreen')
    _ = axr2.annotate("Better", xy=(np.median(ax.get_xlim()), -0.02), xytext=(-25, -50), arrowprops=dict(arrowstyle='->',
                      lw=4, color='darkgreen'), va='center', textcoords='offset points', color='darkgreen', fontsize=20)
    _ = axr.annotate("Better", xy=(np.median(ax.get_xlim()), -0.02), xytext=(-25, -50), arrowprops=dict(arrowstyle='->',
                     lw=4, color='darkgreen'), va='center', textcoords='offset points', color='darkgreen', fontsize=20)
    axr.set_ylabel(r'ACE$_{\mathrm{epis}}$', size=20)
    axr2.set_ylabel(r'IS$_{\mathrm{epis}}$', size=20)
    axr.grid(False)
    axr2.grid(False)
    # right plot
    axr, axr2 = ax_prob[0, 2], ax_prob[1, 2]
    cb = axr.scatter(df.index.values, df.ace.values, s=100,
                     color='red', alpha=0.5)  # , s=12)
    cb2 = axr2.scatter(df.index.values, df.ints.values,
                       s=100, color='red', alpha=0.5)  # , s=12)
    _ = axr.axhline(y=0, lw=2, ls='--', color='darkgreen')
    _ = axr2.axhline(y=0, lw=2, ls='--', color='darkgreen')
    _ = axr2.annotate("Better", xy=(np.median(ax.get_xlim()), -0.02), xytext=(-25, -50), arrowprops=dict(arrowstyle='->',
                      lw=4, color='darkgreen'), va='center', textcoords='offset points', color='darkgreen', fontsize=20)
    _ = axr.annotate("Better", xy=(np.median(ax.get_xlim()), -0.02), xytext=(-25, -50), arrowprops=dict(arrowstyle='->',
                     lw=4, color='darkgreen'), va='center', textcoords='offset points', color='darkgreen', fontsize=20)
    axr.set_ylabel(r'ACE$_{\mathrm{total}}$', size=20)
    axr2.set_ylabel(r'IS$_{\mathrm{total}}$', size=20)
    axr.grid(False)
    axr2.grid(False)
    plt.savefig('ood_'+label_list[label_str]+'_snr=%d' % snr+'_' +
                timenow+'.pdf', bbox_inches='tight', pad_inches=0.001, dpi=600)


for label_str in [0, 1, 2, 3]:
    for snr in snr_list:
        if TRAIN_SET_ABV == 'ET':
            snr = 5
        df = df_ood_dict['df_%s_ood_snr%d' % (label_list[label_str], snr)]
        ood_plot(df)
        print('%s, %d done' % (label_list[label_str], snr))

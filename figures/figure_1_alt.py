"""
Turner, Mann, Clandinin: Figure generation script: Fig. 1.

https://github.com/mhturner/SC-FC
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm, zscore, kstest, pearsonr
import pandas as pd
from skimage import io
import seaborn as sns
from neuprint import (fetch_neurons, NeuronCriteria, Client)

from scfc import bridge, anatomical_connectivity
from matplotlib import rcParams
rcParams.update({'font.size': 10})
rcParams.update({'figure.autolayout': True})
rcParams.update({'axes.spines.right': False})
rcParams.update({'axes.spines.top': False})
rcParams['svg.fonttype'] = 'none' # let illustrator handle the font type

data_dir = bridge.getUserConfiguration()['data_dir']
analysis_dir = bridge.getUserConfiguration()['analysis_dir']

plot_colors = plt.get_cmap('tab10')(np.arange(8)/8)
save_dpi = 400

# %% ~Lognormal distribtution of connection strengths
include_inds_ito, name_list_ito = bridge.getItoNames()
ConnectivityCount = anatomical_connectivity.getAtlasConnectivity(include_inds_ito, name_list_ito, 'ito', metric='cellcount')
ConnectivityTBars = anatomical_connectivity.getAtlasConnectivity(include_inds_ito, name_list_ito, 'ito', metric='tbar')


pull_region = 'AL_R'

fig1_0, ax = plt.subplots(2, 1, figsize=(4.5, 3.5))
ax = ax.ravel()
fig1_0.tight_layout(w_pad=2, h_pad=8)

figS1_0, axS1 = plt.subplots(10, 4, figsize=(8, 9))
axS1 = axS1.ravel()
[x.set_axis_off() for x in axS1]

z_scored_cell = []
z_scored_tbar = []
for p_ind, pr in enumerate(ConnectivityCount.index):
    # # # # CELL COUNT:
    outbound = ConnectivityCount.loc[pr, :]
    outbound = outbound.sort_values(ascending=False)
    ki = np.where(outbound > 0)
    ct = outbound.iloc[ki]
    z_scored_cell.append(zscore(np.log10(ct)))

    lognorm_model = norm(loc=np.mean(np.log10(ct)), scale=np.std(np.log10(ct)))
    iterations = 1000
    samples = []
    for it in range(iterations):
        new_samples = np.sort(lognorm_model.rvs(size=len(ct)))[::-1]
        samples.append(new_samples)
    samples = np.vstack(samples)
    # Note order of operations matters here, get mean and std before going back out of log
    mod_mean = 10**np.mean(samples, axis=0)
    err_down = 10**(np.mean(samples, axis=0) - 2*np.std(samples, axis=0))
    err_up = 10**(np.mean(samples, axis=0) + 2*np.std(samples, axis=0))
    axS1[p_ind].set_axis_on()

    axS1[p_ind].fill_between(list(range(len(mod_mean))), err_up, err_down, color='k', alpha=0.4, rasterized=False)
    axS1[p_ind].plot(mod_mean, 'k--', rasterized=False)
    axS1[p_ind].plot(ct, marker='.', linestyle='none', rasterized=False)

    axS1[p_ind].set_xticks([])
    axS1[p_ind].annotate('{}'.format(bridge.displayName(pr)), (12, 6e3), fontsize=8)
    axS1[p_ind].set_yscale('log')
    axS1[p_ind].set_ylim([0.5, 5e4])
    axS1[p_ind].set_yticks([1e0, 1e2, 1e4])

    if pr == pull_region:
        ax[0].fill_between(list(range(len(mod_mean))), err_up, err_down, color='k', alpha=0.4)
        ax[0].plot(mod_mean, 'k--')
        ax[0].plot(ct, marker='o', linestyle='none')

        ax[0].set_xticks(list(range(len(ct))))
        ax[0].tick_params(axis='both', which='major', labelsize=10)
        ax[0].set_xticklabels([bridge.displayName(x) for x in ct.index])
        ax[0].set_yscale('log')
        ax[0].set_ylim([1, 8e3])
        for tick in ax[0].get_xticklabels():
            tick.set_rotation(90)
            tick.set_fontsize(7)
        ax[0].set_ylabel('Cells')
        ax[0].annotate('Source: {}'.format(bridge.displayName(pr)), (12, 1e4), fontsize=12)

    # # # TBAR COUNT:
    outbound = ConnectivityTBars.loc[pr, :]
    outbound = outbound.sort_values(ascending=False)
    ki = np.where(outbound > 0)
    ct = outbound.iloc[ki]
    z_scored_tbar.append(zscore(np.log10(ct)))

    if pr == pull_region:
        lognorm_model = norm(loc=np.mean(np.log10(ct)), scale=np.std(np.log10(ct)))
        iterations = 1000
        samples = []
        for it in range(iterations):
            new_samples = np.sort(lognorm_model.rvs(size=len(ct)))[::-1]
            samples.append(new_samples)
        samples = np.vstack(samples)

        # Note order of operations matters here, get mean and std before going back out of log
        mod_mean = 10**np.mean(samples, axis=0)
        err_down = 10**(np.mean(samples, axis=0) - 2*np.std(samples, axis=0))
        err_up = 10**(np.mean(samples, axis=0) + 2*np.std(samples, axis=0))
        ax[1].fill_between(list(range(len(mod_mean))), err_up, err_down, color='k', alpha=0.4)
        ax[1].plot(mod_mean, 'k--')
        ax[1].plot(ct, marker='o', linestyle='none')

        ax[1].set_xticks(list(range(len(ct))))
        ax[1].tick_params(axis='both', which='major', labelsize=10)
        ax[1].set_xticklabels([bridge.displayName(x) for x in ct.index])
        ax[1].set_yscale('log')
        ax[1].set_ylim([10, 2e7])
        ax[1].tick_params(axis='y', which='minor')
        for tick in ax[1].get_xticklabels():
            tick.set_rotation(90)
            tick.set_fontsize(7)
        ax[1].set_ylabel('T-Bars')

figS1_0.text(-0.01, 0.5, 'Connections from source region (cells)', va='center', rotation='vertical', fontsize=14)

fig1_0.savefig(os.path.join(analysis_dir, 'figpanels', 'fig1_0.svg'), format='svg', transparent=True, dpi=save_dpi)
figS1_0.savefig(os.path.join(analysis_dir, 'figpanels', 'figS1_0.svg'), format='svg', transparent=True, dpi=save_dpi)

# %%
token = bridge.getUserConfiguration()['token']

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.2', token=token)
AC = anatomical_connectivity.AnatomicalConnectivity(data_dir=data_dir, neuprint_client=neuprint_client, mapping=bridge.getRoiMapping())
old_t = AC.getConnectivityMatrix('TBars')
old_t.sum().sum()
ConnectivityTBars.sum().sum()

# %% Summary across all regions: zscore within each outgoing and compare to lognorm
# CELL COUNT:
p_vals_cellcount = []
for arr in z_scored_cell:
    _, p = kstest(arr, 'norm')
    p_vals_cellcount.append(p)

data = np.hstack(z_scored_cell)

# fit norm model on log transformed data
params = norm.fit(data)
norm_model = norm(loc=params[0], scale=params[1])
theory_distr = []
for iter in range(100):
    theory_distr.append(norm_model.rvs(size=len(data)))
theory_distr = np.vstack(theory_distr)

fig1_1, ax = plt.subplots(2, 2, figsize=(4.0, 3.5))
val, bin = np.histogram(data, 20, density=True)
bin_ctrs = bin[:-1]
xx = np.linspace(-3.5, 3.5)
ax[0, 0].plot(10**xx, norm_model.pdf(xx), linewidth=2, color='k', linestyle='--')
ax[0, 0].plot(10**bin_ctrs, val, linewidth=3)
ax[0, 0].set_xscale('log')
ax[0, 0].set_xlabel('Cells (z-score)')
ax[0, 0].set_ylabel('Prob.')
ax[0, 0].set_xticks([1e-2, 1, 1e2])


# Q-Q plot of log-transformed data vs. fit normal distribution
ax[0, 1].plot([10**-4, 10**4], [10**-4, 10**4], 'k-')
quants = np.linspace(0, 1, 20)
for q in quants:
    th_pts = np.quantile(theory_distr, q, axis=1)  # quantile value for each iteration
    ax[0, 1].plot([10**np.quantile(data, q), 10**np.quantile(data, q)], [10**(np.mean(th_pts) - 2*np.std(th_pts)), 10**(np.mean(th_pts) + 2*np.std(th_pts))], color=plot_colors[0], alpha=0.5)
    ax[0, 1].plot(10**np.quantile(data, q), 10**np.mean(th_pts), marker='o', color=plot_colors[0], alpha=0.5)
ax[0, 1].set_xlabel('Q. Measured')
ax[0, 1].set_ylabel('Q. Lognorm.')
ax[0, 1].set_xscale('log')
ax[0, 1].set_yscale('log')

ax[0, 1].set_xticks([1e-2, 1e2])
ax[0, 1].set_yticks([1e-2, 1e2])
ax[0, 1].set_xticklabels(['-2$\sigma$', '+2$\sigma$'])
ax[0, 1].set_yticklabels(['-2$\sigma$', '+2$\sigma$'])
ax[0, 1].axhline(y=1, color='k', zorder=0, alpha=0.5)
ax[0, 1].axvline(x=1, color='k', zorder=0, alpha=0.5)


# # # # TBARS # # # # # # # # # # # # # # # #:
p_vals_tbar = []
for arr in z_scored_tbar:
    _, p = kstest(arr, 'norm')
    p_vals_tbar.append(p)

data = np.hstack(z_scored_tbar)

# fit norm model on log transformed data
params = norm.fit(data)
norm_model = norm(loc=params[0], scale=params[1])
theory_distr = []
for iter in range(100):
    theory_distr.append(norm_model.rvs(size=len(data)))
theory_distr = np.vstack(theory_distr)

val, bin = np.histogram(data, 20, density=True)
bin_ctrs = bin[:-1]
xx = np.linspace(-3.5, 3.5)
ax[1, 0].plot(10**xx, norm_model.pdf(xx), linewidth=2, color='k', linestyle='--')
ax[1, 0].plot(10**bin_ctrs, val, linewidth=3)
ax[1, 0].set_xscale('log')
ax[1, 0].set_xlabel('T-Bars (z-score)')
ax[1, 0].set_ylabel('Prob.')
ax[1, 0].set_xticks([1e-2, 1, 1e2])


# Q-Q plot of log-transformed data vs. fit normal distribution
ax[1, 1].plot([10**-4, 10**4], [10**-4, 10**4], 'k-')
quants = np.linspace(0, 1, 20)
for q in quants:
    th_pts = np.quantile(theory_distr, q, axis=1)  # quantile value for each iteration
    ax[1, 1].plot([10**np.quantile(data, q), 10**np.quantile(data, q)], [10**(np.mean(th_pts) - 2*np.std(th_pts)), 10**(np.mean(th_pts) + 2*np.std(th_pts))], color=plot_colors[0], alpha=0.5)
    ax[1, 1].plot(10**np.quantile(data, q), 10**np.mean(th_pts), marker='o', color=plot_colors[0], alpha=0.5)
ax[1, 1].set_xlabel('Q. Measured')
ax[1, 1].set_ylabel('Q. Lognorm.')
ax[1, 1].set_xscale('log')
ax[1, 1].set_yscale('log')

ax[1, 1].set_xticks([1e-2, 1e2])
ax[1, 1].set_yticks([1e-2, 1e2])
ax[1, 1].set_xticklabels(['-2$\sigma$', '+2$\sigma$'])
ax[1, 1].set_yticklabels(['-2$\sigma$', '+2$\sigma$'])
ax[1, 1].axhline(y=1, color='k', zorder=0, alpha=0.5)
ax[1, 1].axvline(x=1, color='k', zorder=0, alpha=0.5)


fig1_1.savefig(os.path.join(analysis_dir, 'figpanels', 'fig1_1.svg'), format='svg', transparent=True, dpi=save_dpi)


# %%
# load synmask tifs and atlases
synmask_jrc2018 = io.imread(os.path.join(data_dir, 'hemi_2_atlas', 'JRC2018_synmask.tif'))
# scale synmask by max synapse per. voxel density from script (=64).
#       writeTIF in R script auto scales to 2**16 max for some reason
synmask_jrc2018 = (synmask_jrc2018 / np.max(synmask_jrc2018))
synmask_jrc2018 = 64 * synmask_jrc2018

branson_jrc2018 = io.imread(os.path.join(data_dir, 'template_brains', '2018_999_atlas.tif'))
ito_jrc2018 = io.imread(os.path.join(data_dir, 'template_brains', 'ito_2018.tif'))

include_inds_ito, name_list_ito = bridge.getItoNames()
include_inds_branson, name_list_branson = bridge.getBransonNames()

# %% atlas alignment images
# branson atlas
figS1_2, ax = plt.subplots(3, 1, figsize=(4, 8))
[x.set_axis_off() for x in ax.ravel()]

np.random.seed(1)
tmp = 0.75 * np.ones((1000, 3))
tmp[include_inds_branson, :] = np.random.rand(len(include_inds_branson), 3)
tmp[0, :] = [1, 1, 1]
cmap = matplotlib.colors.ListedColormap(tmp)
ax[1].imshow(branson_jrc2018[250, :, :], cmap=cmap, interpolation='None')

# Ito atlas
np.random.seed(1)
tmp = 0.75 * np.ones((86, 3))
tmp[include_inds_ito, :] = np.random.rand(len(include_inds_ito), 3)
tmp[0, :] = [1, 1, 1]
cmap = matplotlib.colors.ListedColormap(tmp)
ax[0].imshow(ito_jrc2018[250, :, :], cmap=cmap, interpolation='None')

# syn density mask
im = ax[2].imshow(synmask_jrc2018[240:260, :, :].mean(axis=0), interpolation='None')
cb = figS1_2.colorbar(im, ax=ax[2], shrink=1.0, orientation="horizontal", pad=0.2, label='Synapse density (Tbars/voxel)')
figS1_2.tight_layout()
figS1_2.savefig(os.path.join(analysis_dir, 'figpanels', 'figS1_2.svg'), format='svg', transparent=True, dpi=save_dpi)

# %% Alignment testing
token = bridge.getUserConfiguration()['token']
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.2', token=token)


def doAlignmentTest(cell_type, neuprint_search):

    # (1) Get TBar count based on atlas aligntment
    tbar = pd.read_csv(os.path.join(data_dir, 'hemi_2_atlas', '{}_ito_tbar.csv'.format(cell_type)), header=0).iloc[:, 1:]
    tbar.index = np.arange(1, 87)
    include_inds_ito, name_list_ito = bridge.getItoNames()
    tbar = tbar.loc[include_inds_ito, :]
    tbar.index = name_list_ito
    tbar = pd.DataFrame(tbar.sum(axis=1), columns=['sum'])

    fh, ax = plt.subplots(2, 1, figsize=(4, 2))

    vmin = 1
    vmax = np.nanmax(tbar.to_numpy())

    sns.heatmap(np.log10(tbar).replace(-np.inf, 0).T, ax=ax[1],
                cmap='cividis', cbar=False, yticklabels=False, xticklabels=[bridge.displayName(x) for x in tbar.index],
                vmin=0, vmax=np.log10(vmax), rasterized=True)

    # ax[1].set_title('Atlas\nregistration')
    ax[1].tick_params(axis='both', which='major', labelsize=7)

    # (2) Get TBar count according to Neuprint & Janelia region tags
    Neur, _ = fetch_neurons(NeuronCriteria(type=neuprint_search, status='Traced', regex=True))

    neuprint_rois = [bridge.ito_to_neuprint(x) for x in name_list_ito]

    tbar_count = np.zeros((Neur.shape[0], len(neuprint_rois)))
    for roi_ind, nr in enumerate(neuprint_rois):
        for n_ind, roiInfo in enumerate(Neur.roiInfo):
            if len(nr) > 1:
                new_tbars = np.sum([roiInfo.get(x, {'pre': 0}).get('pre', 0) for x in nr])
            else:
                new_tbars = roiInfo.get(nr[0], {'pre': 0}).get('pre', 0)

            tbar_count[n_ind, roi_ind] = new_tbars

    tbar_neuprint = pd.DataFrame(tbar_count.sum(axis=0).T, index=name_list_ito, columns=['ct'])

    sns.heatmap(np.log10(tbar_neuprint).replace(-np.inf, 0).T, ax=ax[0],
                cmap='cividis', cbar=False, yticklabels=False, xticklabels=False,
                vmin=0, vmax=np.log10(vmax), rasterized=True)
    # ax[0].set_title('Neuprint')
    ax[0].tick_params(axis='both', which='major', labelsize=7)
    # position = fh.add_axes([1.0, 0.1, 0.05, 0.75])
    cb = fh.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.SymLogNorm(vmin=vmin, vmax=vmax, base=10, linthresh=0.1, linscale=1), cmap="cividis"), ax=ax, shrink=1, label='T-Bars')

    r, p = pearsonr(tbar.to_numpy().ravel(), tbar_neuprint.to_numpy().ravel())
    print('r = {}'.format(r))

    print(tbar.sum().sum())
    print(tbar_neuprint.sum().sum())

    return fh


figS1_3 = doAlignmentTest(cell_type='ER', neuprint_search='ER.*')
figS1_3.savefig(os.path.join(analysis_dir, 'figpanels', 'figS1_3.svg'), format='svg', transparent=True, dpi=save_dpi)

figS1_4 = doAlignmentTest(cell_type='LNO', neuprint_search="LNO.*")
figS1_4.savefig(os.path.join(analysis_dir, 'figpanels', 'figS1_4.svg'), format='svg', transparent=True, dpi=save_dpi)



#

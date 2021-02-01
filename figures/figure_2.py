"""
Turner, Mann, Clandinin: Figure generation script: Fig. 2.

https://github.com/mhturner/SC-FC
"""

import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import os
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns
import glob
import nibabel as nib
from scipy.ndimage.measurements import center_of_mass

from scipy.spatial.distance import pdist
from seriate import seriate


from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
import matplotlib
from matplotlib import rcParams
rcParams.update({'font.size': 12})
rcParams.update({'axes.spines.right': False})
rcParams.update({'axes.spines.top': False})
rcParams['svg.fonttype'] = 'none'  # let illustrator handle the font type

data_dir = bridge.getUserConfiguration()['data_dir']
analysis_dir = bridge.getUserConfiguration()['analysis_dir']

plot_colors = plt.get_cmap('tab10')(np.arange(8)/8)
save_dpi = 400

# %% Eg region traces and cross corrs
pull_regions = ['AL_R', 'CAN_R', 'LH_R', 'SPS_R']

include_inds_ito, name_list_ito = bridge.getItoNames()

display_names = [x.replace('_R', '(R)').replace('_L', '(L)').replace('_', '') for x in name_list_ito]

pull_inds = [np.where(np.array(name_list_ito) == x)[0][0] for x in pull_regions]

resp_fp = os.path.join(data_dir, 'ito_responses', 'ito_2018-11-03_5.pkl')
voxel_size = [3, 3, 3]  # um, xyz

brain_str = '2018-11-03_5'
brain_fn = 'func_volreg_{}_meanbrain.nii'.format(brain_str)
atlas_fn = 'vfb_68_{}.nii.gz'.format(brain_str)
brain_fp = os.path.join(data_dir, 'ito_responses', brain_fn)
atlas_fp = os.path.join(data_dir, 'ito_responses', atlas_fn)

# load eg meanbrain and region masks
meanbrain = functional_connectivity.getMeanBrain(brain_fp)
all_masks = functional_connectivity.loadAtlasData(atlas_fp, include_inds_ito, name_list_ito)
masks = list(np.array(all_masks)[pull_inds])

cmap = plt.get_cmap('Set2')
colors = cmap(np.arange(len(pull_regions))/len(pull_regions))

zslices = [12, 45]
fig2_0 = plt.figure(figsize=(2.5, 2.5))
for z_ind, z in enumerate(zslices):
    ax = fig2_0.add_subplot(2, 2, z_ind+3)
    ax.annotate('z={} $ \mu m$'.format(z*voxel_size[2]), (1, 18), color='w', fontsize=10)

    overlay = plotting.overlayImage(meanbrain, masks, 0.5, colors=colors, z=z) + 60  # arbitrary brighten here for visualization

    img = ax.imshow(np.swapaxes(overlay, 0, 1), rasterized=True)
    ax.set_axis_off()
    ax.set_aspect('equal')

ax = fig2_0.add_subplot(2, 2, 2)
ax.imshow(np.mean(meanbrain, axis=2).T, cmap='inferno')
ax.annotate('Mean proj.', (12, 18), color='w', fontsize=10)
ax.set_axis_off()
ax.set_aspect('equal')

dx = 100  # um
dx_pix = int(dx / voxel_size[0])
ax.plot([5, dx_pix], [120, 120], 'w-')
fig2_0.subplots_adjust(hspace=0.02, wspace=0.02)

fs = 1.2  # Hz
cutoff = 0.01

x_start = 200
dt = 300  # datapts
timevec = np.arange(0, dt) / fs  # sec

file_id = resp_fp.split('/')[-1].replace('.pkl', '')
region_response = pd.read_pickle(resp_fp).loc[include_inds_ito]

# convert to dF/F
dff = (region_response.to_numpy() - np.mean(region_response.to_numpy(), axis=1)[:, None]) / np.mean(region_response.to_numpy(), axis=1)[:, None]

# trim and filter
resp = functional_connectivity.filterRegionResponse(dff, cutoff=cutoff, fs=fs)
resp = functional_connectivity.trimRegionResponse(file_id, resp)
region_dff = pd.DataFrame(data=resp, index=name_list_ito)

# fig2_1, ax = plt.subplots(4, 1, figsize=(3.5, 4))
fig2_1, ax = plt.subplots(4, 1, figsize=(8, 4))
ax = ax.ravel()
[x.set_axis_off() for x in ax]
[x.set_ylim([-0.2, 0.29]) for x in ax]
[x.set_xlim([-15, timevec[-1]]) for x in ax]
for p_ind, pr in enumerate(pull_regions):
    ax[p_ind].plot(timevec, region_dff.loc[pr, x_start:(x_start+dt-1)], color=colors[p_ind])
    ax[p_ind].annotate(bridge.displayName(pr), (-10, 0), rotation=90, fontsize=10)

plotting.addScaleBars(ax[0], dT=10, dF=0.10, T_value=-2.5, F_value=-0.10)
fig2_1.subplots_adjust(hspace=0.02, wspace=0.02)


fig2_2, ax = plt.subplots(3, 3, figsize=(2.5, 2.5))
[x.set_xticks([]) for x in ax.ravel()]
[x.set_yticks([]) for x in ax.ravel()]
for ind_1, eg1 in enumerate(pull_regions):
    for ind_2, eg2 in enumerate(pull_regions):
        if ind_1 > ind_2:

            r, p = pearsonr(region_dff.loc[eg1, :], region_dff.loc[eg2, :])

            # normed xcorr plot
            window_size = 180
            total_len = len(region_dff.loc[eg1, :])

            a = (region_dff.loc[eg1, :] - np.mean(region_dff.loc[eg1, :])) / (np.std(region_dff.loc[eg1, :]) * len(region_dff.loc[eg1, :]))
            b = (region_dff.loc[eg2, :] - np.mean(region_dff.loc[eg2, :])) / (np.std(region_dff.loc[eg2, :]))
            c = np.correlate(a, b, 'same')
            time = np.arange(-window_size/2, window_size/2) / fs # sec
            ax[ind_1-1, ind_2].plot(time, c[int(total_len/2-window_size/2): int(total_len/2+window_size/2)], 'k')
            ax[ind_1-1, ind_2].set_ylim([-0.2, 1])
            ax[ind_1-1, ind_2].axhline(0, color='k', alpha=0.5, linestyle='-')
            ax[ind_1-1, ind_2].axvline(0, color='k', alpha=0.5, linestyle='-')
            if ind_2==0:
                ax[ind_1-1, ind_2].set_ylabel(bridge.displayName(eg1), fontsize=10)
            if ind_1==3:
                ax[ind_1-1, ind_2].set_xlabel(bridge.displayName(eg2), fontsize=10)
fig2_2.subplots_adjust(hspace=0.02, wspace=0.02)

plotting.addScaleBars(ax[0, 0], dT=-30, dF=0.25, T_value=time[-1], F_value=-0.15)
sns.despine(top=True, right=True, left=True, bottom=True)
fig2_0.savefig(os.path.join(analysis_dir, 'figpanels', 'fig2_0.svg'), format='svg', transparent=True, dpi=save_dpi)
fig2_1.savefig(os.path.join(analysis_dir, 'figpanels', 'fig2_1.svg'), format='svg', transparent=True, dpi=save_dpi)
fig2_2.savefig(os.path.join(analysis_dir, 'figpanels', 'fig2_2.svg'), format='svg', transparent=True, dpi=save_dpi)

# %%
# Plot heatmaps, ordered by TSP seriation
Structural_Matrix = anatomical_connectivity.getAtlasConnectivity(include_inds_ito, name_list_ito, 'ito')
np.fill_diagonal(Structural_Matrix.to_numpy(), 0)

response_filepaths = glob.glob(os.path.join(data_dir, 'ito_responses') + '/' + '*.pkl')
Functional_Matrix, cmats_z = functional_connectivity.getCmat(response_filepaths, include_inds_ito, name_list_ito)
Fxn_tmp = Functional_Matrix.to_numpy().copy()
np.fill_diagonal(Fxn_tmp, 1)

sort_inds = seriate(pdist(Structural_Matrix))
sort_keys = np.array(name_list_ito)[sort_inds]
np.fill_diagonal(Structural_Matrix.to_numpy(), np.nan)

SC_ordered = pd.DataFrame(data=np.zeros_like(Structural_Matrix), columns=sort_keys, index=sort_keys)
FC_ordered = pd.DataFrame(data=np.zeros_like(Functional_Matrix), columns=sort_keys, index=sort_keys)
for r_ind, r_key in enumerate(sort_keys):
    for c_ind, c_key in enumerate(sort_keys):
        SC_ordered.iloc[r_ind, c_ind]=Structural_Matrix.loc[[r_key], [c_key]].to_numpy()
        FC_ordered.iloc[r_ind, c_ind]=Functional_Matrix.loc[[r_key], [c_key]].to_numpy()

fig2_3, ax = plt.subplots(1, 2, figsize=(9, 4))
# fxnal heatmap
sns.heatmap(FC_ordered, ax=ax[0],
            yticklabels=[bridge.displayName(x) for x in FC_ordered.index],
            xticklabels=[bridge.displayName(x) for x in FC_ordered.index],
            cbar_kws={'label': 'Functional Correlation (z)', 'shrink': .75}, cmap="cividis", rasterized=True)
ax[0].set_aspect('equal')
ax[0].tick_params(axis='both', which='major', labelsize=6)
# structural heatmap
sns.heatmap(np.log10(SC_ordered).replace([np.inf, -np.inf], 0), ax=ax[1],
            yticklabels=[bridge.displayName(x) for x in SC_ordered.index],
            xticklabels=[bridge.displayName(x) for x in SC_ordered.index],
            cmap="cividis", rasterized=True, cbar=False)
cb = fig2_3.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.SymLogNorm(vmin=1, vmax=np.nanmax(SC_ordered.to_numpy()), base=10, linthresh=0.1, linscale=1), cmap="cividis"), ax=ax[1], shrink=0.75, label='Connecting cells')
cb.outline.set_linewidth(0)
ax[1].set_aspect('equal')
ax[1].tick_params(axis='both', which='major', labelsize=6)
fig2_3.subplots_adjust(wspace=0.25)

# %%
# # # SC vs FC scatter plot and linear corr # # #
# Make adjacency matrices
# Log transform anatomical connectivity
Structural_Matrix = anatomical_connectivity.getAtlasConnectivity(include_inds_ito, name_list_ito, 'ito').to_numpy().copy()
Structural_Matrix = (Structural_Matrix + Structural_Matrix.T) / 2 # symmetrize

keep_inds = np.where(Structural_Matrix[np.triu_indices(len(name_list_ito), k=1)] > 0)
anatomical_adjacency = np.log10(Structural_Matrix[np.triu_indices(len(name_list_ito), k=1)][keep_inds])

functional_adjacency = Functional_Matrix.to_numpy()[np.triu_indices(len(name_list_ito), k=1)][keep_inds]

r, p = pearsonr(anatomical_adjacency, functional_adjacency)
coef = np.polyfit(anatomical_adjacency, functional_adjacency, 1)
linfit = np.poly1d(coef)

fig2_4, ax = plt.subplots(1, 1, figsize=(2.25, 2.25))
ax.plot(10**anatomical_adjacency, functional_adjacency, color='k', marker='.', linestyle='none', alpha=1.0)
xx = np.linspace(anatomical_adjacency.min(), anatomical_adjacency.max(), 100)
ax.plot(10**xx, linfit(xx), color='k', linewidth=2, marker=None)
ax.set_xscale('log')
ax.set_xlabel('Cell Count')
ax.set_ylabel('Functional corr. (z)')
ax.annotate('r = {:.2f}'.format(r), xy=(0.8, 1.05))
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)


# %% # # # volume-normalized SC vs FC scatter plot and linear corr # # #
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')
include_inds_ito, name_list_ito = bridge.getItoNames()
coms, roi_size, DistanceMatrix, SizeMatrix = functional_connectivity.getRegionGeometry(atlas_path, include_inds_ito, name_list_ito)

ct_per_size = Structural_Matrix / SizeMatrix
ct_per_size = ct_per_size.to_numpy()[np.triu_indices(len(name_list_ito), k=1)]

keep_inds = np.where(ct_per_size > 0)
x = np.log10(ct_per_size[keep_inds])
y = Functional_Matrix.to_numpy()[np.triu_indices(len(name_list_ito), k=1)][keep_inds]

r, p = pearsonr(x, y)
coef = np.polyfit(x, y, 1)
linfit = np.poly1d(coef)

fig2_5, ax = plt.subplots(1, 1, figsize=(2.25, 2.25))
ax.plot(10**x, y, color='k', marker='.', linestyle='none', alpha=1.0)
xx = np.linspace(x.min(), x.max(), 100)
ax.plot(10**xx, linfit(xx), color='k', linewidth=2, marker=None)
ax.set_xscale('log')
ax.set_xlabel('Norm. Cell Count\n(cells/voxel)')
ax.set_ylabel('Functional corr. (z)')
ax.annotate('r = {:.2f}'.format(r), xy=(1e-3, 1.0))
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

# %%


metrics = ['cellcount', 'weighted_tbar', 'tbar', 'Size', 'Nearness']
R_by_metric = pd.DataFrame(data=np.zeros((len(cmats_z), len(metrics))), columns=metrics)
pop_r = []
for metric in metrics:
    if metric in ['cellcount', 'weighted_tbar', 'tbar']:
        Structural_Matrix = anatomical_connectivity.getAtlasConnectivity(include_inds_ito, name_list_ito, 'ito', metric=metric).to_numpy().copy()
        Structural_Matrix = (Structural_Matrix + Structural_Matrix.T) / 2 # symmetrize
        keep_inds = np.where(Structural_Matrix[np.triu_indices(len(name_list_ito), k=1)] > 0)
        anatomical_adjacency = np.log10(Structural_Matrix[np.triu_indices(len(name_list_ito), k=1)][keep_inds])
    elif metric == 'Size':
        anatomical_adjacency = SizeMatrix.to_numpy()[np.triu_indices(len(name_list_ito), k=1)]
        keep_inds = np.arange(np.triu_indices(len(name_list_ito), k=1)[0].size)
    elif metric == 'Nearness':
        anatomical_adjacency = 1/DistanceMatrix.to_numpy()[np.triu_indices(len(name_list_ito), k=1)]
        keep_inds = np.arange(np.triu_indices(len(name_list_ito), k=1)[0].size)

    functional_adjacency_pop = Functional_Matrix.to_numpy()[np.triu_indices(len(name_list_ito), k=1)][keep_inds]
    r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_pop)
    pop_r.append(r_new)

    r_vals = []
    for c_ind in range(len(cmats_z)):
        cmat = cmats_z[c_ind]
        functional_adjacency_new = cmat[np.triu_indices(len(name_list_ito), k=1)][keep_inds]
        r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_new)
        r_vals.append(r_new)
    R_by_metric.loc[:, metric] = r_vals


# Individual fly sc/fc corr, for different metrics
fig2_6, ax = plt.subplots(1, 1, figsize=(2.75, 2.5))
ax.set_ylabel('Structure-function corr. (r)')
ax.set_ylim([-0.2, 1])
ax.axhline(0, color=[0.8, 0.8, 0.8], linestyle='-', zorder=0)
sns.violinplot(data=R_by_metric, color=[0.8, 0.8, 0.8], alpha=0.5, zorder=1)
sns.stripplot(data=R_by_metric, color=plot_colors[0], alpha=1.0, zorder=2)

ax.plot(np.arange(len(pop_r)), pop_r, color='k', marker='s', markersize=6, linestyle='None', alpha=1.0, zorder=3)
ax.set_xticklabels(['Cell\ncount',
                    'Weighted\nT-Bar\ncount',
                    'Raw\nT-Bar \ncount',
                    'Region\nsize',
                    'Region\nnearness'])
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=10)

fig2_3.savefig(os.path.join(analysis_dir, 'figpanels', 'fig2_3.svg'), format='svg', transparent=True, dpi=save_dpi)
fig2_4.savefig(os.path.join(analysis_dir, 'figpanels', 'fig2_4.svg'), format='svg', transparent=True, dpi=save_dpi)
fig2_5.savefig(os.path.join(analysis_dir, 'figpanels', 'fig2_5.svg'), format='svg', transparent=True, dpi=save_dpi)
fig2_6.savefig(os.path.join(analysis_dir, 'figpanels', 'fig2_6.svg'), format='svg', transparent=True, dpi=save_dpi)

# %% Supp: Branson atlas SC-FC

response_filepaths = glob.glob(os.path.join(data_dir, 'branson_responses') + '/' + '*.pkl')

include_inds_branson, name_list_branson = bridge.getBransonNames()

CorrelationMatrix_branson, cmats_branson = functional_connectivity.getCmat(response_filepaths, include_inds_branson, name_list_branson)
Branson_JRC2018 = anatomical_connectivity.getAtlasConnectivity(include_inds_branson, name_list_branson, 'branson')

names, inds_unique = np.unique(name_list_branson, return_index=True)
inds_unique = np.append(inds_unique, len(name_list_branson))
cmap = plt.get_cmap('tab20')(np.arange(len(names))/len(names))
np.random.seed(0)
np.random.shuffle(cmap)

inds = [np.where(names==name)[0][0] for name in name_list_branson]
atlas_colors = [cmap[i] for i in inds]

# Functional corr
g_fxn = sns.clustermap(CorrelationMatrix_branson, cmap='cividis',
                       cbar_kws={},
                       rasterized=True,
                       row_cluster=False, col_cluster=False,
                       row_colors=atlas_colors, col_colors=atlas_colors,
                       linewidths=0, xticklabels=False, yticklabels=False,
                       figsize=(4, 4),
                       cbar_pos=(0, 0, 0.0, 0.0))



g_fxn.cax.set_visible(False)
for l_ind, label in enumerate(names):
    loc = (inds_unique[l_ind] + inds_unique[l_ind+1]) / 2
    # g_fxn.ax_heatmap.annotate(bridge.displayName(label), xy=(loc, 0), rotation=90, fontsize=4, color=cmap[l_ind], fontweight='bold')
    # g_fxn.ax_heatmap.annotate(bridge.displayName(label), xy=(0, loc), rotation=0, fontsize=4, color=cmap[l_ind], fontweight='bold', ha='right')

position=g_fxn.fig.add_axes([1.0, 0.1, 0.025, 0.6])
cb = g_fxn.fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=np.nanmin(CorrelationMatrix_branson.to_numpy()), vmax=np.nanmax(CorrelationMatrix_branson.to_numpy())), cmap="cividis"),
                        ax=g_fxn.ax_row_dendrogram, label='Functional Correlation (z)',
                        cax=position)


# Structural conn
tmp = Branson_JRC2018.to_numpy()
np.fill_diagonal(tmp, np.nan)
conn_mat = pd.DataFrame(data=tmp, index=name_list_branson, columns=name_list_branson)
g_struct = sns.clustermap(np.log10(conn_mat).replace([np.inf, -np.inf], 0), cmap='cividis',
                          cbar_kws={},
                          rasterized=True,
                          row_cluster=False, col_cluster=False,
                          row_colors=atlas_colors, col_colors=atlas_colors,
                          linewidths=0, xticklabels=False, yticklabels=False,
                          figsize=(4, 4),
                          cbar_pos=(0, 0, 0.0, 0.0))
g_struct.cax.set_visible(False)
for l_ind, label in enumerate(names):
    loc = (inds_unique[l_ind] + inds_unique[l_ind+1]) / 2
    g_struct.ax_heatmap.annotate(bridge.displayName(label), xy=(loc, 0), rotation=90, fontsize=4, color=cmap[l_ind], fontweight='bold')
    g_struct.ax_heatmap.annotate(bridge.displayName(label), xy=(0, loc), rotation=0, fontsize=4, color=cmap[l_ind], fontweight='bold', ha='right')

position=g_struct.fig.add_axes([1.0, 0.1, 0.025, 0.6])
cb = g_struct.fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.SymLogNorm(vmin=1, vmax=np.nanmax(conn_mat.to_numpy()), base=10, linthresh=0.1, linscale=1), cmap="cividis"),
                           ax=g_struct.ax_row_dendrogram, label='Connecting cells',
                           cax=position)

# corr: branson
figS2_2, ax = plt.subplots(1, 1, figsize=(2.5, 3.5))
x = Branson_JRC2018.to_numpy()[np.triu_indices(len(name_list_branson), k=1)]
keep_inds = np.where(x > 0)
x = np.log10(x[keep_inds])
y = CorrelationMatrix_branson.to_numpy()[np.triu_indices(len(name_list_branson), k=1)]
y = y[keep_inds]

r, p = pearsonr(x, y)
coef = np.polyfit(x, y, 1)
linfit = np.poly1d(coef)

# hexbin plot
hb = ax.hexbin(x, y, bins='log', gridsize=40)
xx = np.linspace(x.min(), x.max(), 100)
ax.plot(xx, linfit(xx), color='w', linewidth=2, marker=None)
ax.set_xlabel('Cell Count')
ax.set_ylabel('Functional corr. (z)')
ax.annotate('r = {:.2f}'.format(r), xy=(0.05, 1.05), color='w')
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$'])
cb = figS2_2.colorbar(hb, ax=ax, shrink=0.75, label='Connections', orientation='horizontal')

g_fxn.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_0.svg'), format='svg', transparent=True, dpi=save_dpi)
g_struct.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_1.svg'), format='svg', transparent=True, dpi=save_dpi)
figS2_2.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_2.svg'), format='svg', transparent=True, dpi=save_dpi)

# %% Branson avg vs Ito - load

response_filepaths = glob.glob(os.path.join(data_dir, 'branson_responses') + '/' + '*.pkl')
include_inds_branson, name_list_branson = bridge.getBransonNames()
CorrelationMatrix_branson, cmats_branson = functional_connectivity.getCmat(response_filepaths, include_inds_branson, name_list_branson)
Branson_JRC2018 = anatomical_connectivity.getAtlasConnectivity(include_inds_branson, name_list_branson, 'branson')

response_filepaths = glob.glob(os.path.join(data_dir, 'ito_responses') + '/' + '*.pkl')
include_inds_ito, name_list_ito = bridge.getItoNames()
CorrelationMatrix_ito, cmats_ito = functional_connectivity.getCmat(response_filepaths, include_inds_ito, name_list_ito)
Ito_JRC2018 = anatomical_connectivity.getAtlasConnectivity(include_inds_ito, name_list_ito, 'ito')

# %% Branson avg vs Ito - figs
unique_regions = np.unique(name_list_branson)

branson_matched_sc = np.zeros((len(unique_regions), len(unique_regions)))
branson_matched_sc.fill(np.nan)
branson_matched_fc = np.zeros((len(unique_regions), len(unique_regions)))
branson_matched_fc.fill(np.nan)

ito_matched_sc = np.zeros((len(unique_regions), len(unique_regions)))
ito_matched_sc.fill(np.nan)
ito_matched_fc = np.zeros((len(unique_regions), len(unique_regions)))
ito_matched_fc.fill(np.nan)

for r_ind, r_region in enumerate(unique_regions):
    if r_region in name_list_ito:
        ito_row = [r_region]
    elif r_region == 'MB_L':
        ito_row = ['MB_ML_L']
    elif r_region == 'MB_R':
        ito_row = ['MB_CA_R', 'MB_ML_R', 'MB_PED_R', 'MB_VL_R']
    else:
        continue

    for c_ind, c_region in enumerate(unique_regions):
        if c_region in name_list_ito:
            ito_col = [c_region]
        elif c_region == 'MB_L':
            ito_col = ['MB_ML_L']
        elif c_region == 'MB_R':
            ito_col = ['MB_CA_R', 'MB_ML_R', 'MB_PED_R', 'MB_VL_R']
        else:
            continue

        branson_matched_sc[r_ind, c_ind] = np.mean(np.mean(Branson_JRC2018.loc[r_region, c_region]))
        branson_matched_fc[r_ind, c_ind] = np.mean(np.mean(CorrelationMatrix_branson.loc[r_region, c_region]))

        ito_matched_sc[r_ind, c_ind] = np.mean(np.mean(Ito_JRC2018.loc[ito_row, ito_col]))
        ito_matched_fc[r_ind, c_ind] = np.mean(np.mean(CorrelationMatrix_ito.loc[ito_row, ito_col]))

np.fill_diagonal(branson_matched_sc, np.nan)
np.fill_diagonal(branson_matched_fc, np.nan)
np.fill_diagonal(ito_matched_sc, np.nan)
np.fill_diagonal(ito_matched_fc, np.nan)

branson_matched_sc = pd.DataFrame(data=branson_matched_sc, index=unique_regions, columns=unique_regions).dropna(axis=0, how='all').dropna(axis=1, how='all')
branson_matched_fc = pd.DataFrame(data=branson_matched_fc, index=unique_regions, columns=unique_regions).dropna(axis=0, how='all').dropna(axis=1, how='all')

ito_matched_sc = pd.DataFrame(data=ito_matched_sc, index=unique_regions, columns=unique_regions).dropna(axis=0, how='all').dropna(axis=1, how='all')
ito_matched_fc = pd.DataFrame(data=ito_matched_fc, index=unique_regions, columns=unique_regions).dropna(axis=0, how='all').dropna(axis=1, how='all')

figS2_3, ax = plt.subplots(1, 2, figsize=(6.5, 2.75))
sns.heatmap(np.log10(branson_matched_sc).replace([np.inf, -np.inf], 0), ax=ax[0],
            yticklabels=[bridge.displayName(x) for x in branson_matched_sc.index],
            xticklabels=[bridge.displayName(x) for x in branson_matched_sc.index],
            cmap="cividis", rasterized=True, cbar=False, vmin=0)
cb = figS2_3.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.SymLogNorm(vmin=1, vmax=np.nanmax(branson_matched_sc.to_numpy()), base=10, linthresh=0.1, linscale=1), cmap="cividis"),
                      ax=ax[0], shrink=0.75)
cb.outline.set_linewidth(0)
cb.ax.tick_params(labelsize=8)
ax[0].set_aspect('equal')
ax[0].tick_params(axis='both', which='major', labelsize=6)

sns.heatmap(np.log10(ito_matched_sc).replace([np.inf, -np.inf], 0), ax=ax[1],
            yticklabels=[bridge.displayName(x) for x in ito_matched_sc.index],
            xticklabels=[bridge.displayName(x) for x in ito_matched_sc.index],
            cmap="cividis", rasterized=True, cbar=False, vmin=0)
cb = figS2_3.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.SymLogNorm(vmin=1, vmax=np.nanmax(ito_matched_sc.to_numpy()), base=10, linthresh=0.1, linscale=1), cmap="cividis"),
                      ax=ax[1], shrink=0.75, label='Connecting cells')
cb.outline.set_linewidth(0)
cb.ax.tick_params(labelsize=6)
ax[1].set_aspect('equal')
ax[1].tick_params(axis='both', which='major', labelsize=6)

figS2_4, ax = plt.subplots(1, 2, figsize=(6.5, 2.75))
sns.heatmap(branson_matched_fc, ax=ax[0], cmap='cividis',
            xticklabels=[bridge.displayName(x) for x in branson_matched_fc.index], yticklabels=[bridge.displayName(x) for x in branson_matched_fc.index],
            vmin=0, cbar=False, rasterized=True)
cb = figS2_4.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=np.nanmin(branson_matched_fc.to_numpy()), vmax=np.nanmax(branson_matched_fc.to_numpy())), cmap="cividis"),
                      ax=ax[0], shrink=0.75)
cb.ax.tick_params(labelsize=8)
ax[0].set_aspect('equal')
ax[0].tick_params(axis='both', which='major', labelsize=5)

sns.heatmap(ito_matched_fc, ax=ax[1], cmap='cividis',
            xticklabels=[bridge.displayName(x) for x in ito_matched_fc.index], yticklabels=[bridge.displayName(x) for x in ito_matched_fc.index],
            vmin=0, cbar=False, rasterized=True)
cb = figS2_4.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=np.nanmin(ito_matched_fc.to_numpy()), vmax=np.nanmax(ito_matched_fc.to_numpy())), cmap="cividis"),
                      ax=ax[1], label='Functional Corr. (z)', shrink=0.75)
cb.ax.tick_params(labelsize=6)
ax[1].set_aspect('equal')
ax[1].tick_params(axis='both', which='major', labelsize=5)

figS2_5, ax = plt.subplots(1, 2, figsize=(5.5, 2.5))
x = ito_matched_sc.to_numpy().ravel()
y = branson_matched_sc.to_numpy().ravel()
keep_inds = np.where(np.logical_and(x>0, y>0))
x = np.log10(x[keep_inds])
y = np.log10(y[keep_inds])
r, p = pearsonr(x, y)

coef = np.polyfit(x, y, 1)
linfit = np.poly1d(coef)
xx = np.linspace(x.min(), x.max(), 100)
ax[0].annotate('r={:.2f}'.format(r), xy=(0.1, 8e3))
ax[0].plot(10**x, 10**y, color=[0.5, 0.5, 0.5], marker='.', linestyle='None')
ax[0].plot([0.1, 8000], [0.1, 8000], 'k--')
ax[0].plot(10**xx, 10**linfit(xx), color='k', linewidth=2, marker=None)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('Ito Conn. (cells)')
ax[0].set_ylabel('Branson Conn. (cells)')
ax[0].set_aspect('equal')

x = ito_matched_fc.to_numpy()[np.triu_indices(ito_matched_fc.shape[0], k=1)]
y = branson_matched_fc.to_numpy()[np.triu_indices(ito_matched_fc.shape[0], k=1)]
r, p = pearsonr(x, y)
coef = np.polyfit(x, y, 1)
linfit = np.poly1d(coef)
xx = np.linspace(x.min(), x.max(), 100)
ax[1].annotate('r={:.2f}'.format(r), xy=(0, 1.0))
ax[1].plot(x, y, color=[0.5, 0.5, 0.5], marker='.', linestyle='None')
ax[1].plot([0, 1], [0, 1], 'k--')
ax[1].plot(xx, linfit(xx), color='k', linewidth=2, marker=None)
ax[1].set_xlabel('Ito Corr. (z)')
ax[1].set_ylabel('Branson Corr. (z)')
ax[1].set_aspect('equal')

figS2_3.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_3.svg'), format='svg', transparent=True, dpi=save_dpi)
figS2_4.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_4.svg'), format='svg', transparent=True, dpi=save_dpi)
figS2_5.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_5.svg'), format='svg', transparent=True, dpi=save_dpi)
# %% Branson: intra regions
res = 0.68  # um / vox isotropic
atlas_path = os.path.join(data_dir, 'atlas_data', 'AnatomySubCompartments20150108_ms999centers.nii')

mask_brain = np.asarray(np.squeeze(nib.load(atlas_path).get_fdata()), 'uint16')

coms = np.vstack([center_of_mass(mask_brain==x) for x in include_inds_branson]) # x,y,z (voxel space) <---- LONG COMPUTE TIME

coms = coms * res
# %%
# calulcate euclidean distance matrix between roi centers of mass
dist_mat = np.zeros((len(include_inds_branson), len(include_inds_branson)))
dist_mat[np.triu_indices(len(include_inds_branson), k=1)] = pdist(coms)
dist_mat += dist_mat.T # symmetrize to fill in below diagonal
DistanceMatrix = pd.DataFrame(data=dist_mat, index=name_list_branson, columns=name_list_branson)

unique_regions = np.unique(name_list_branson)

figS2_6, ax = plt.subplots(4, 5, figsize=(6.5, 4))
ax = ax.ravel()
ct = 0
dists = []
for ind, ur in enumerate(unique_regions):
    pull_inds = np.where(ur == name_list_branson)[0]
    if len(pull_inds) > 3:
        dist = DistanceMatrix.iloc[pull_inds, pull_inds]

        intra_sc = Branson_JRC2018.loc[ur, ur]
        intra_sc = (intra_sc + intra_sc.T) / 2
        intra_fc = CorrelationMatrix_branson.loc[ur, ur]
        n_roi = intra_sc.shape[0]

        x = intra_sc.to_numpy()[np.triu_indices(n_roi, k=1)]
        y = intra_fc.to_numpy()[np.triu_indices(n_roi, k=1)]
        d = dist.to_numpy()[np.triu_indices(n_roi, k=1)]

        ax[ct].scatter(x, y, c=d, marker='.', vmin=0, vmax=123, rasterized=True)
        r, p = pearsonr(np.log10(x[x>0]), y[x>0])
        ax[ct].annotate('{}\nr={:.2f}'.format(bridge.displayName(ur), r), xy=(1.5, 0.75), fontsize=8)
        ax[ct].set_ylim([0, 1])
        ax[ct].set_xlim([1, 2000])
        ax[ct].set_xscale('log')
        if ct !=15:
            ax[ct].set_xticks([])
            ax[ct].set_yticks([])
        ct += 1
        dists.append(d)

dists = np.hstack(dists)
np.max(dists)
cb = figS2_6.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=123)), ax=ax, shrink=0.75, label='Distance (um)')
figS2_6.text(0.5, 0.08, 'Structural connectivity (cell count)', ha='center', fontsize=12)
figS2_6.text(0.08, 0.5, 'Functional correlation (z)', va='center', rotation='vertical', fontsize=12)

figS2_6.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_6.svg'), format='svg', transparent=True, dpi=save_dpi)


# %% Supp: subsampled region cmats and SC-FC corr

# Load SC mat:
include_inds_ito, name_list_ito = bridge.getItoNames()
Structural_Matrix = anatomical_connectivity.getAtlasConnectivity(include_inds_ito, name_list_ito, 'ito')

# Load Ito atlas geometry
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')
coms, roi_size, DistanceMatrix, SizeMatrix = functional_connectivity.getRegionGeometry(atlas_path, include_inds_ito, name_list_ito)

# Comp structural adjacency
Structural_Matrix = (Structural_Matrix + Structural_Matrix.T) / 2
keep_inds = np.where(Structural_Matrix.to_numpy()[np.triu_indices(len(name_list_ito), k=1)] > 0)
anatomical_adjacency = np.log10(Structural_Matrix.to_numpy()[np.triu_indices(len(name_list_ito), k=1)][keep_inds])

# Load precomputed subsampled Cmats for each brain
CorrelationMatrix_Full = pd.read_pickle(os.path.join(data_dir, 'subsample', 'subsample_CorrelationMatrix_Full.pkl'))
cmats_full = np.load(os.path.join(data_dir, 'subsample', 'subsample_cmats_full.npy'))
cmats_subsampled, subsampled_sizes = np.load(os.path.join(data_dir, 'subsample', 'subsample_cmats_sub.npy'), allow_pickle=True)

functional_adjacency_full = CorrelationMatrix_Full.to_numpy()[np.triu_indices(len(name_list_ito), k=1)][keep_inds]
r_full, _ = pearsonr(anatomical_adjacency, functional_adjacency_full)


bins = np.arange(np.floor(np.min(roi_size)), np.ceil(np.max(roi_size)))
values, base = np.histogram(roi_size, bins=bins, density=True)
cumulative = np.cumsum(values)

# mean cmat over brains for each subsampledsize and iteration
cmats_popmean = np.mean(cmats_subsampled, axis=4) # roi x roi x iterations x sizes
scfc_r = np.zeros(shape=(cmats_popmean.shape[2], cmats_popmean.shape[3])) # iterations x sizes
for s_ind, sz in enumerate(subsampled_sizes):
    for it in range(cmats_popmean.shape[2]):
        functional_adjacency_tmp = cmats_popmean[:, :, it, s_ind][np.triu_indices(len(name_list_ito), k=1)][keep_inds]
        new_r, _ = pearsonr(anatomical_adjacency, functional_adjacency_tmp)
        scfc_r[it, s_ind] = new_r

# plot mean+/-SEM results on top of region size cumulative histogram
err_y = np.std(scfc_r, axis=0)
mean_y = np.mean(scfc_r, axis=0)

figS2_7, ax1 = plt.subplots(1, 1, figsize=(3, 3))
ax1.plot(subsampled_sizes, mean_y, 'ko')
ax1.errorbar(subsampled_sizes, mean_y, yerr=err_y, color='k')
ax1.hlines(r_full, subsampled_sizes.min(), subsampled_sizes.max(), color='k', linestyle='--')
ax1.set_xlabel('Region size (voxels)')
ax1.set_ylabel('Correlation with anatomical connectivity')
ax1.set_xscale('log')
ax2 = ax1.twinx()
ax2.plot(bins[:-1], cumulative)
ax2.set_ylabel('Cumulative fraction')
ax2.set_ylim([0, 1.05])
ax1.set_ylim([0, 0.8])
ax2.set_xscale('log')
ax2.spines['right'].set_visible(True)

figS2_7.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_7.svg'), format='svg', transparent=True, dpi=save_dpi)

# %% Supp: AC+FC vs. completeness, distance

include_inds_ito, name_list_ito = bridge.getItoNames()
response_filepaths = glob.glob(os.path.join(data_dir, 'ito_responses') + '/' + '*.pkl')
Functional_Matrix, cmats_z = functional_connectivity.getCmat(response_filepaths, include_inds_ito, name_list_ito)
Structural_Matrix = anatomical_connectivity.getAtlasConnectivity(include_inds_ito, name_list_ito, 'ito')
Structural_Matrix = (Structural_Matrix + Structural_Matrix.T) / 2 # symmetrize

# start client
token = bridge.getUserConfiguration()['token']
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.2', token=token)

# Atlas roi completeness measures
roi_completeness = anatomical_connectivity.getRoiCompleteness(neuprint_client, name_list_ito)
CompletenessMatrix = pd.DataFrame(data=np.outer(roi_completeness['frac_post'], roi_completeness['frac_pre']), index=roi_completeness.index, columns=roi_completeness.index)
CompletenessMatrix = (CompletenessMatrix + CompletenessMatrix.T) / 2 # symmetrize to compare with symmetrized SC

# upper triangle
fc = Functional_Matrix.to_numpy()[np.triu_indices(len(name_list_ito), k=1)]
sc = Structural_Matrix.to_numpy()[np.triu_indices(len(name_list_ito), k=1)]
compl = CompletenessMatrix.to_numpy()[np.triu_indices(len(name_list_ito), k=1)]

figS2_8, ax = plt.subplots(1, 2, figsize=(6.5, 3))
ax[0].plot(compl, sc, 'k.', alpha=1.0, rasterized=True)
r, p = plotting.addLinearFit(ax[0], compl, sc, alpha=1.0)
ax[0].set_xlabel('Completeness')
ax[0].set_ylabel('Connecting cells')
ax[0].set_xlim([0, 1])
ax[0].annotate('r={:.2f}'.format(r), (0.72, 3000))

ax[1].plot(compl, fc, 'k.', alpha=1.0, rasterized=True)
r, p = plotting.addLinearFit(ax[1], compl, fc, alpha=1.0)
ax[1].set_xlabel('Completeness')
ax[1].set_ylabel('Functional Corr. (z)')
ax[1].set_xlim([0, 1])
ax[1].annotate('r={:.2f}'.format(r), (0.05, 1.02))
figS2_8.subplots_adjust(wspace=0.5)
figS2_8.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_8.svg'), format='svg', transparent=True, dpi=save_dpi)

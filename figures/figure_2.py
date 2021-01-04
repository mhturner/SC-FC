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
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score, contingency_matrix
from munkres import Munkres


from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
import matplotlib
from matplotlib import rcParams
rcParams.update({'font.size': 12})
rcParams.update({'axes.spines.right': False})
rcParams.update({'axes.spines.top': False})
rcParams['svg.fonttype'] = 'none' # let illustrator handle the font type

data_dir = bridge.getUserConfiguration()['data_dir']
analysis_dir = bridge.getUserConfiguration()['analysis_dir']
token = bridge.getUserConfiguration()['token']

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token=token)

# Get FunctionalConnectivity object
FC = functional_connectivity.FunctionalConnectivity(data_dir=data_dir, fs=1.2, cutoff=0.01, mapping=bridge.getRoiMapping())

# Get AnatomicalConnectivity object
AC = anatomical_connectivity.AnatomicalConnectivity(data_dir=data_dir, neuprint_client=neuprint_client, mapping=bridge.getRoiMapping())

plot_colors = plt.get_cmap('tab10')(np.arange(8)/8)
save_dpi = 400

# %% Eg region traces and cross corrs
pull_regions = ['AL(R)', 'CAN(R)', 'LH(R)', 'SPS(R)']
pull_inds = [np.where(np.array(FC.rois) == x)[0][0] for x in pull_regions]

resp_fp = os.path.join(data_dir, 'region_responses', '2018-11-03_5.pkl')
voxel_size = [3, 3, 3]  # um, xyz

brain_str = '2018-11-03_5'
brain_fn = 'func_volreg_{}_meanbrain.nii'.format(brain_str)
atlas_fn = 'vfb_68_{}.nii.gz'.format(brain_str)
brain_fp = os.path.join(data_dir, 'region_responses', brain_fn)
atlas_fp = os.path.join(data_dir, 'region_responses', atlas_fn)

# load eg meanbrain and region masks
meanbrain = FC.getMeanBrain(brain_fp)
all_masks, _ = FC.loadAtlasData(atlas_fp)
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
region_response = pd.read_pickle(resp_fp)
# convert to dF/F
dff = (region_response.to_numpy() - np.mean(region_response.to_numpy(), axis=1)[:, None]) / np.mean(region_response.to_numpy(), axis=1)[:, None]

# trim and filter
resp = functional_connectivity.filterRegionResponse(dff, cutoff=cutoff, fs=fs)
resp = functional_connectivity.trimRegionResponse(file_id, resp)
region_dff = pd.DataFrame(data=resp, index=region_response.index)

# fig2_1, ax = plt.subplots(4, 1, figsize=(3.5, 4))
fig2_1, ax = plt.subplots(4, 1, figsize=(8, 4))
ax = ax.ravel()
[x.set_axis_off() for x in ax]
[x.set_ylim([-0.2, 0.29]) for x in ax]
[x.set_xlim([-15, timevec[-1]]) for x in ax]
for p_ind, pr in enumerate(pull_regions):
    ax[p_ind].plot(timevec, region_dff.loc[pr, x_start:(x_start+dt-1)], color=colors[p_ind])
    ax[p_ind].annotate(pr, (-10, 0), rotation=90, fontsize=10)

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
                ax[ind_1-1, ind_2].set_ylabel(eg1, fontsize=10)
            if ind_1==3:
                ax[ind_1-1, ind_2].set_xlabel(eg2, fontsize=10)
fig2_2.subplots_adjust(hspace=0.02, wspace=0.02)

plotting.addScaleBars(ax[0, 0], dT=-30, dF=0.25, T_value=time[-1], F_value=-0.15)
sns.despine(top=True, right=True, left=True, bottom=True)
fig2_0.savefig(os.path.join(analysis_dir, 'figpanels', 'fig2_0.svg'), format='svg', transparent=True, dpi=save_dpi)
fig2_1.savefig(os.path.join(analysis_dir, 'figpanels', 'fig2_1.svg'), format='svg', transparent=True, dpi=save_dpi)
fig2_2.savefig(os.path.join(analysis_dir, 'figpanels', 'fig2_2.svg'), format='svg', transparent=True, dpi=save_dpi)

# %%

fig2_3, ax = plt.subplots(1, 2, figsize=(9, 4))
# fxnal heatmap
sns.heatmap(FC.CorrelationMatrix, ax=ax[0], yticklabels=True, xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)', 'shrink': .75}, cmap="cividis", rasterized=True)
ax[0].set_aspect('equal')
ax[0].tick_params(axis='both', which='major', labelsize=6)
# structural heatmap
df = AC.getConnectivityMatrix('CellCount', diag=np.nan)
sns.heatmap(np.log10(AC.getConnectivityMatrix('CellCount', diag=np.nan)).replace([np.inf, -np.inf], 0), ax=ax[1], yticklabels=True, xticklabels=True, cmap="cividis", rasterized=True, cbar=False)
cb = fig2_3.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.SymLogNorm(vmin=1, vmax=np.nanmax(df.to_numpy()), base=10, linthresh=0.1, linscale=1), cmap="cividis"), ax=ax[1], shrink=0.75, label='Connecting cells')
cb.outline.set_linewidth(0)
# ax[1].set_xlabel('Target', fontsize=10)
# ax[1].set_ylabel('Source', fontsize=10)
ax[1].set_aspect('equal')
ax[1].tick_params(axis='both', which='major', labelsize=6)
fig2_3.subplots_adjust(wspace=0.25)
# Make adjacency matrices
# Log transform anatomical connectivity
anatomical_adjacency, keep_inds = AC.getAdjacency('CellCount', do_log=True)
functional_adjacency = FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]

r, p = pearsonr(anatomical_adjacency, functional_adjacency)
coef = np.polyfit(anatomical_adjacency, functional_adjacency, 1)
linfit = np.poly1d(coef)

fig2_4, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.plot(10**anatomical_adjacency, functional_adjacency, color='k', marker='.', linestyle='none', alpha=1.0)
xx = np.linspace(anatomical_adjacency.min(), anatomical_adjacency.max(), 100)
ax.plot(10**xx, linfit(xx), color='k', linewidth=2, marker=None)
ax.set_xscale('log')
ax.set_xlabel('Cell Count')
ax.set_ylabel('Functional correlation (z)')
ax.annotate('r = {:.2f}'.format(r), xy=(0.8, 1.1))

metrics = ['CellCount', 'WeightedSynapseCount', 'TBars', 'Size', 'Nearness']
R_by_metric = pd.DataFrame(data=np.zeros((FC.cmats.shape[2], len(metrics))), columns=metrics)
pop_r = []
for metric in metrics:
    if metric in ['CellCount', 'WeightedSynapseCount', 'TBars']:
        anatomical_adjacency, keep_inds = AC.getAdjacency(metric, do_log=True)
    elif metric == 'Size':
        anatomical_adjacency = FC.SizeMatrix.to_numpy()[FC.upper_inds]
        keep_inds = np.arange(FC.upper_inds[0].size)
    elif metric == 'Nearness':
        anatomical_adjacency = 1/FC.DistanceMatrix.to_numpy()[FC.upper_inds]
        keep_inds = np.arange(FC.upper_inds[0].size)

    functional_adjacency_pop = FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]
    r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_pop)
    pop_r.append(r_new)

    r_vals = []
    for c_ind in range(FC.cmats.shape[2]):
        cmat = FC.cmats[:, :, c_ind]
        functional_adjacency_new = cmat[FC.upper_inds][keep_inds]
        r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_new)
        r_vals.append(r_new)
    R_by_metric.loc[:, metric] = r_vals

fig2_5, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.set_ylabel('Structure-function\n corr. (r)')
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

fig2_3.savefig(os.path.join(analysis_dir, 'figpanels', 'fig2_3.svg'), format='svg', transparent=True, dpi=save_dpi)
fig2_4.savefig(os.path.join(analysis_dir, 'figpanels', 'fig2_4.svg'), format='svg', transparent=True, dpi=save_dpi)
fig2_5.savefig(os.path.join(analysis_dir, 'figpanels', 'fig2_5.svg'), format='svg', transparent=True, dpi=save_dpi)

# %% Supp: subsampled region cmats and SC-FC corr

atlas_fns = glob.glob(os.path.join(data_dir, 'atlas_data', 'vfb_68_2*'))
sizes = []
for fn in atlas_fns:
    _, roi_size = FC.loadAtlasData(atlas_path=fn)
    sizes.append(roi_size)

sizes = np.vstack(sizes)
roi_size = np.mean(sizes, axis=0)

np.sort(roi_size)

anatomical_adjacency, keep_inds = AC.getAdjacency('CellCount', do_log=True)

bins = np.arange(np.floor(np.min(roi_size)), np.ceil(np.max(roi_size)))
values, base = np.histogram(roi_size, bins=bins, density=True)
cumulative = np.cumsum(values)


# Load precomputed subsampled Cmats for each brain
load_fn = os.path.join(data_dir, 'functional_connectivity', 'subsampled_cmats_20200626.npy')
(cmats_pop, CorrelationMatrix_Full, subsampled_sizes) = np.load(load_fn, allow_pickle=True)

# mean cmat over brains for each subsampledsize and iteration
cmats_popmean = np.mean(cmats_pop, axis=4) # roi x roi x iterations x sizes
scfc_r = np.zeros(shape=(cmats_popmean.shape[2], cmats_popmean.shape[3])) # iterations x sizes
for s_ind, sz in enumerate(subsampled_sizes):
    for it in range(cmats_popmean.shape[2]):
        functional_adjacency_tmp = cmats_popmean[:, :, it, s_ind][FC.upper_inds][keep_inds]
        new_r, _ = pearsonr(anatomical_adjacency, functional_adjacency_tmp)
        scfc_r[it, s_ind] = new_r

# plot mean+/-SEM results on top of region size cumulative histogram
err_y = np.std(scfc_r, axis=0)
mean_y = np.mean(scfc_r, axis=0)

figS2_1, ax1 = plt.subplots(1, 1, figsize=(4, 4))
ax1.plot(subsampled_sizes, mean_y, 'ko')
ax1.errorbar(subsampled_sizes, mean_y, yerr=err_y, color='k')
ax1.hlines(mean_y[-1], subsampled_sizes.min(), subsampled_sizes.max(), color='k', linestyle='--')
ax1.set_xlabel('Region size (voxels)')
ax1.set_ylabel('Correlation with anatomical connectivity')
ax1.set_xscale('log')
ax2 = ax1.twinx()
ax2.plot(bins[:-1], cumulative)
ax2.set_ylabel('Cumulative fraction')
ax2.set_ylim([0, 1.05])
ax2.set_xscale('log')
ax2.spines['right'].set_visible(True)

figS2_1.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_1.svg'), format='svg', transparent=True, dpi=save_dpi)

# %% Supp: AC+FC vs. completeness, distance
cell_ct, _ = AC.getAdjacency('CellCount', do_log=False)
completeness = (AC.CompletenessMatrix.to_numpy() + AC.CompletenessMatrix.to_numpy().T) / 2
fc = FC.CorrelationMatrix.to_numpy()[FC.upper_inds]
compl = completeness[FC.upper_inds]

figS2_2, ax = plt.subplots(1, 2, figsize=(6.5, 3))
ax[0].plot(compl, cell_ct, 'k.', alpha=1.0, rasterized=True)
r, p = plotting.addLinearFit(ax[0], compl, cell_ct, alpha=1.0)
ax[0].set_xlabel('Completeness')
ax[0].set_ylabel('Anat. conn. (cells)')
ax[0].set_xlim([0, 1])
ax[0].annotate('r={:.2f}'.format(r), (0.72, 3400))

ax[1].plot(compl, fc, 'k.', alpha=1.0, rasterized=True)
r, p = plotting.addLinearFit(ax[1], compl, fc, alpha=1.0)
ax[1].set_xlabel('Completeness')
ax[1].set_ylabel('Functional correlation (z)')
ax[1].set_xlim([0, 1])
ax[1].annotate('r={:.2f}'.format(r), (0.05, 1.02))
figS2_2.subplots_adjust(wspace=0.5)
figS2_2.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_2.svg'), format='svg', transparent=True, dpi=save_dpi)

# %% Supp: Predicting FC with cells per volume, to normalize for region size

ct_per_size = AC.getConnectivityMatrix('CellCount') / FC.SizeMatrix
ct_per_size = ct_per_size.to_numpy()[FC.upper_inds]

keep_inds = np.where(ct_per_size > 0)
x = np.log10(ct_per_size[keep_inds])
y = FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]

r, p = pearsonr(x, y)
coef = np.polyfit(x, y, 1)
linfit = np.poly1d(coef)

figS2_3, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.plot(10**x, y, color='k', marker='.', linestyle='none', alpha=1.0)
xx = np.linspace(x.min(), x.max(), 100)
ax.plot(10**xx, linfit(xx), color='k', linewidth=2, marker=None)
ax.set_xscale('log')
ax.set_xlabel('Cell Count / voxels')
ax.set_ylabel('Functional correlation (z)')
ax.annotate('r = {:.2f}'.format(r), xy=(4e-4, 0.95))

figS2_3.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_3.svg'), format='svg', transparent=True, dpi=save_dpi)

# %% heatmaps for non-connectome, anatomical data.

figS2_4, ax = plt.subplots(1, 4, figsize=(12, 3))
# fxnal heatmap
sns.heatmap(FC.CorrelationMatrix, ax=ax[0], yticklabels=False, xticklabels=False, cbar_kws={'shrink': .75}, cmap="cividis", rasterized=True)
ax[0].set_aspect('equal')
ax[0].set_title('Functional')

# structural heatmap
df = AC.getConnectivityMatrix('CellCount', diag=np.nan)
sns.heatmap(np.log10(AC.getConnectivityMatrix('CellCount', diag=np.nan)).replace([np.inf, -np.inf], 0), ax=ax[1], yticklabels=False, xticklabels=False, cmap="cividis", rasterized=True, cbar=False)
cb = fig2_3.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.SymLogNorm(vmin=1, vmax=np.nanmax(df.to_numpy()), base=10, linthresh=0.1, linscale=1), cmap="cividis"), ax=ax[1], shrink=0.75)
cb.outline.set_linewidth(0)
ax[1].set_aspect('equal')
ax[1].set_title('Cell count')

# size matrix
np.fill_diagonal(FC.SizeMatrix.to_numpy(), np.nan)
sns.heatmap(FC.SizeMatrix, ax=ax[2], yticklabels=False, xticklabels=False, cbar_kws={'shrink': .75}, cmap="cividis", rasterized=True, vmin=0)
ax[2].set_aspect('equal')
ax[2].set_title('Size')


# distance matrix
np.fill_diagonal(FC.DistanceMatrix.to_numpy(), np.nan)
sns.heatmap(1/FC.DistanceMatrix, ax=ax[3], yticklabels=False, xticklabels=False, cbar_kws={'shrink': .75}, cmap="cividis", rasterized=True)
ax[3].set_aspect('equal')
ax[3].set_title('Nearness')

figS2_4.subplots_adjust(wspace=0.25)

figS2_4.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_4.svg'), format='svg', transparent=True, dpi=save_dpi)

# %% Clustering of SC and FC networks, and comparison of clusters across datasets.

struct_mat = np.log10(AC.getConnectivityMatrix('CellCount', diag=0).to_numpy().copy())
struct_mat[np.where(np.isinf(struct_mat))] = 0

fxn_mat = FC.CorrelationMatrix.to_numpy().copy()
np.fill_diagonal(fxn_mat, 0)

figS2_5, f_ax = plt.subplots(1, 8, figsize=(16, 2))
for c_ind, num_clusters in enumerate(range(2, 10)):
    ax = f_ax[c_ind]

    clustering_struct = SpectralClustering(n_clusters=num_clusters, assign_labels="kmeans", random_state=0, affinity='precomputed').fit(struct_mat)

    clustering_fxn = SpectralClustering(n_clusters=num_clusters, assign_labels="kmeans", random_state=0, affinity='precomputed').fit(fxn_mat)

    # Contingency matrix := Cij is the number of samples in i that share the same label in j
    cont = contingency_matrix(clustering_struct.labels_, clustering_fxn.labels_)
    cont = cont / cont.sum(axis=1)[:, np.newaxis] # normalize by total number of regions in structural cluster, i.e. rows sum to 1.0

    ARI = adjusted_rand_score(clustering_struct.labels_, clustering_fxn.labels_)

    sns.heatmap(cont, ax=ax, yticklabels=False, xticklabels=False, cmap="cividis", rasterized=True, vmin=0, vmax=1, cbar=False)
    ax.set_title('ARI = {:.2f}'.format(ARI))
    ax.set_aspect('equal')

figS2_5.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_5.svg'), format='svg', transparent=True, dpi=save_dpi)


# %%

num_clusters = 5 # from peak ARI above

clustering_struct = SpectralClustering(n_clusters=num_clusters, assign_labels="kmeans", random_state=0, affinity='precomputed').fit(struct_mat)
clustering_fxn = SpectralClustering(n_clusters=num_clusters, assign_labels="kmeans", random_state=0, affinity='precomputed').fit(fxn_mat)

contmat = contingency_matrix(clustering_struct.labels_, clustering_fxn.labels_)
label_map = Munkres().compute(contmat.max() - contmat)

remapped_struct = np.zeros_like(clustering_struct.labels_)
remapped_struct[:] = np.nan
for c in range(num_clusters):
    remapped_struct[np.where(clustering_struct.labels_==label_map[c][0])] = label_map[c][1]

struct = np.log10(AC.getConnectivityMatrix('CellCount', diag=np.nan)).replace([np.inf, -np.inf], 0)

sort_inds = np.argsort(remapped_struct)
cluster_boundaries_struct = np.where(np.diff(np.sort(remapped_struct)) == 1)[0] + 1
sort_keys = struct.index[sort_inds]
clustered_struct = pd.DataFrame(data=np.zeros_like(struct), columns=sort_keys, index=sort_keys)
for r_ind, r_key in enumerate(sort_keys):
    for c_ind, c_key in enumerate(sort_keys):
        clustered_struct.iloc[r_ind, c_ind]=struct.loc[[r_key], [c_key]].to_numpy()


funct = FC.CorrelationMatrix.copy()
sort_inds = np.argsort(clustering_fxn.labels_)
cluster_boundaries_fxn = np.where(np.diff(np.sort(clustering_fxn.labels_)) == 1)[0] + 1
sort_keys = funct.index[sort_inds]
clustered_fxn = pd.DataFrame(data=np.zeros_like(struct), columns=sort_keys, index=sort_keys)
for r_ind, r_key in enumerate(sort_keys):
    for c_ind, c_key in enumerate(sort_keys):
        clustered_fxn.iloc[r_ind, c_ind]=funct.loc[[r_key], [c_key]].to_numpy()


# %%
figS2_6, ax = plt.subplots(1, 2, figsize=(9, 4))
# fxnal heatmap
sns.heatmap(clustered_fxn, ax=ax[0], yticklabels=True, xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)', 'shrink': .75}, cmap="cividis", rasterized=True)
ax[0].set_aspect('equal')
ax[0].tick_params(axis='both', which='major', labelsize=6)
[ax[0].axhline(x, color='w') for x in cluster_boundaries_fxn]
[ax[0].axvline(x, color='w') for x in cluster_boundaries_fxn]

# structural heatmap
df = AC.getConnectivityMatrix('CellCount', diag=np.nan)
sns.heatmap(clustered_struct, ax=ax[1], yticklabels=True, xticklabels=True, cmap="cividis", rasterized=True, cbar=False)
cb = fig2_3.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.SymLogNorm(vmin=1, vmax=np.nanmax(df.to_numpy()), base=10, linthresh=0.1, linscale=1), cmap="cividis"), ax=ax[1], shrink=0.75, label='Connecting cells')
cb.outline.set_linewidth(0)
ax[1].set_aspect('equal')
ax[1].tick_params(axis='both', which='major', labelsize=6)
[ax[1].axhline(x, color='w') for x in cluster_boundaries_struct]
[ax[1].axvline(x, color='w') for x in cluster_boundaries_struct]

figS2_6.subplots_adjust(wspace=0.25)

figS2_6.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_6.svg'), format='svg', transparent=True, dpi=save_dpi)
# %%
clust_colors = plt.get_cmap('Accent')(np.arange(num_clusters)/num_clusters)

frac_match = np.sum(remapped_struct == clustering_fxn.labels_) / len(remapped_struct)
sort_inds = np.argsort(remapped_struct)

print('{:.2f} regions assigned to matching cluster'.format(frac_match))

figS2_7, ax = plt.subplots(2, 1, figsize=(9, 3))
pal = clust_colors[remapped_struct[sort_inds]]
ax[0].imshow(np.arange(len(pal)).reshape(1, len(pal)), cmap=matplotlib.colors.ListedColormap(list(pal)), interpolation="nearest", aspect="auto")
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].spines['left'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].set_ylabel('Structural')
lab_x = np.hstack(([0], cluster_boundaries_struct, [36]))
for clust in range(5):
    y = 0.0
    x = lab_x[clust] + (lab_x[clust+1] - lab_x[clust])/2 - 1
    ax[0].annotate(clust+1, (x, y))

pal = clust_colors[clustering_fxn.labels_[sort_inds]]
ax[1].imshow(np.arange(len(pal)).reshape(1, len(pal)), cmap=matplotlib.colors.ListedColormap(list(pal)), interpolation="nearest", aspect="auto")
ax[1].set_xticklabels(struct.index[sort_inds], rotation=90)
ax[1].set_xticks(range(36))
ax[1].set_yticks([])
ax[1].spines['left'].set_visible(False)
ax[1].set_ylabel('Functional')

figS2_7.savefig(os.path.join(analysis_dir, 'figpanels', 'figS2_7.svg'), format='svg', transparent=True, dpi=save_dpi)

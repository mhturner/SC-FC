
import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import os
from scipy.stats import zscore, pearsonr, spearmanr
import pandas as pd
import seaborn as sns
import socket

from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
import matplotlib
from matplotlib import rcParams
rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})
rcParams.update({'axes.spines.right': False})
rcParams.update({'axes.spines.top': False})
rcParams['svg.fonttype'] = 'none'  # let illustrator handle the font type

if socket.gethostname() == 'MHT-laptop':  # windows
    data_dir = r'C:\Users\mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
    analysis_dir = r'C:\Users\mhturner/Dropbox/ClandininLab/Analysis/SC-FC'
elif socket.gethostname() == 'max-laptop':  # linux
    data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
    analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC'

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token=bridge.getNeuprintToken())

# Get FunctionalConnectivity object
FC = functional_connectivity.FunctionalConnectivity(data_dir=data_dir, fs=1.2, cutoff=0.01, mapping=bridge.getRoiMapping())

# Get AnatomicalConnectivity object
AC = anatomical_connectivity.AnatomicalConnectivity(data_dir=data_dir, neuprint_client=neuprint_client, mapping=bridge.getRoiMapping())

plot_colors = plt.get_cmap('tab10')(np.arange(8)/8)

# %% Difference matrix

# # compute difference matrix using original, asymmetric anatomical connectivity matrix
anatomical_mat = AC.getConnectivityMatrix('CellCount', diag=0).to_numpy().copy()
functional_mat = FC.CorrelationMatrix.to_numpy().copy()
np.fill_diagonal(functional_mat, 0)

# log transform anatomical connectivity values
keep_inds_diff = np.where(anatomical_mat > 0)
functional_adjacency_diff = functional_mat[keep_inds_diff]
anatomical_adjacency_diff = np.log10(anatomical_mat[keep_inds_diff])

F_zscore = zscore(functional_adjacency_diff)
A_zscore = zscore(anatomical_adjacency_diff)
diff = F_zscore - A_zscore

diff_m = np.zeros_like(anatomical_mat)
diff_m[keep_inds_diff] = diff
DifferenceMatrix = pd.DataFrame(data=diff_m, index=FC.rois, columns=FC.rois)


# %% sort difference matrix by most to least different rois
diff_by_region = DifferenceMatrix.mean()
sort_inds = np.argsort(diff_by_region)[::-1]
sort_keys = DifferenceMatrix.index[sort_inds]
sorted_diff = pd.DataFrame(data=np.zeros_like(DifferenceMatrix),columns=sort_keys, index=sort_keys)
for r_ind, r_key in enumerate(sort_keys):
    for c_ind, c_key in enumerate(sort_keys):
        sorted_diff.iloc[r_ind, c_ind]=DifferenceMatrix.loc[[r_key], [c_key]].to_numpy()

fig3_0, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
lim = np.nanmax(np.abs(DifferenceMatrix.to_numpy().ravel()))
ax.scatter(A_zscore, F_zscore, alpha=1, c=diff, cmap="RdBu",  vmin=-lim, vmax=lim, edgecolors='k', linewidths=0.5)
ax.plot([-3, 4], [-3, 4], 'k-')
ax.set_xlabel('Structural Conn. (z-score)')
ax.set_ylabel('Functional Conn. (z-score)');

fig3_1, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
sns.heatmap(sorted_diff, ax=ax, yticklabels=True, xticklabels=True, cbar_kws={'label': 'Difference (FC - SC)','shrink': .65}, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
ax.set_aspect('equal')
ax.tick_params(axis='both', which='major', labelsize=7)

# fig3_0.savefig(os.path.join(analysis_dir, 'figpanels', 'fig3_0.svg'), format='svg', transparent=True)
# fig3_1.savefig(os.path.join(analysis_dir, 'figpanels', 'fig3_1.svg'), format='svg', transparent=True)

# # Diff by region map
# diff_by_region = DifferenceMatrix.mean()
# diff_brain = np.zeros(shape=FC.roi_mask[0].shape)
# diff_brain[:] = np.nan
# for r_ind, r in enumerate(FC.roi_mask):
#     diff_brain[r] = diff_by_region[r_ind]
#
# zslices = np.linspace(5, 60, 8)
# lim = np.nanmax(np.abs(diff_brain.ravel()))
#
# fig3_x = plt.figure(figsize=(7, 3.5))
# for z_ind, z in enumerate(zslices):
#     ax = fig3_2.add_subplot(2, 4, z_ind+1)
#     img = ax.imshow(diff_brain[:, :, int(z)].T, cmap="RdBu", rasterized=False, vmin=-lim, vmax=lim)
#     ax.set_axis_off()
#     ax.set_aspect('equal')
#     ax.set_xlim([0, 102])
#     ax.set_ylim([107, 5])
#
# fig3_x, ax = plt.subplots(1, 1, figsize=(1, 3))
# ax.set_axis_off()
# cb = fig3_3.colorbar(img, ax=ax)
# cb.set_label(label='Region-average diff.', weight='bold', color='k')
# cb.ax.tick_params(labelsize=12, color='k')

# %% Average diffs within super-regions, look at fly-to-fly variability and compare super-regions
from scipy.stats import ttest_1samp, ttest_ind

regions = {'AL/LH': ['AL(R)', 'LH(R)'],
           'MB': ['MBCA(R)', 'MBML(R)', 'MBML(L)', 'MBPED(R)', 'MBVL(R)'],
           'CX': ['EB', 'FB', 'PB', 'NO'],
           'LX': ['BU(L)', 'BU(R)', 'LAL(R)'],
           'INP': ['CRE(L)', 'CRE(R)', 'SCL(R)', 'ICL(R)', 'IB', 'ATL(L)', 'ATL(R)'],
           'VMNP': ['VES(R)', 'EPA(R)', 'GOR(L)', 'GOR(R)', 'SPS(R)' ],
           'SNP': ['SLP(R)', 'SIP(R)', 'SMP(R)', 'SMP(L)'],
           'VLNP': ['AOTU(R)', 'AVLP(R)', 'PVLP(R)', 'PLP(R)', 'WED(R)'],
           # 'PENP': ['CAN(R)'],
         }

# log transform anatomical connectivity values
anatomical_mat = AC.getConnectivityMatrix('CellCount', diag=0).to_numpy().copy()
keep_inds_diff = np.where(anatomical_mat > 0)
anatomical_adj = np.log10(anatomical_mat[keep_inds_diff])

diff_by_region = []
for c_ind in range(FC.cmats.shape[2]):
    cmat = FC.cmats[:, :, c_ind]
    functional_adj = cmat[keep_inds_diff]

    F_zscore_fly = zscore(functional_adj)
    A_zscore_fly = zscore(anatomical_adj)

    diff = F_zscore_fly - A_zscore_fly

    diff_m = np.zeros_like(anatomical_mat)
    diff_m[keep_inds_diff] = diff
    diff_by_region.append(diff_m.mean(axis=0))

diff_by_region = np.vstack(diff_by_region).T  # region x fly

fig3_2, ax = plt.subplots(1, 1, figsize=(7, 3.5))
ax.axhline(0, color=[0.8, 0.8, 0.8], linestyle='-', zorder=0)

iterations = 100
p_vals = pd.DataFrame(data=np.zeros((1, len(regions))), columns=regions.keys())
DiffBySuperRegion = pd.DataFrame(data=np.zeros((20, len(regions))), columns=regions.keys())
ShuffledDiffByRegion = np.zeros(shape=(20, len(regions), iterations))
for r_ind, reg in enumerate(regions):
    in_inds = np.where([r in regions[reg] for r in FC.rois])[0]
    in_diffs = np.mean(diff_by_region[in_inds, :], axis=0)  #  mean across all regions in super-region, for each fly
    DiffBySuperRegion.loc[:, reg] = in_diffs

    # Shuffle super-region indices to do bootstrap comparison

    shuffle_diffs = []
    for it in range(iterations):
        shuffle_inds = np.random.choice(np.arange(0,len(FC.rois)), len(in_inds), replace=False)
        new_diffs = np.mean(diff_by_region[shuffle_inds, :], axis=0)  #  mean across all regions in super-region
        shuffle_diffs.append(new_diffs) # iterations x flies
    shuffle_diffs = np.vstack(shuffle_diffs)

    _, p = ttest_ind(shuffle_diffs.ravel(), in_diffs)

    p_vals.loc[:, reg] = p

# sort super regions by mean
sort_inds = DiffBySuperRegion.mean().sort_values().index[::-1]
DiffBySuperRegion_sorted = DiffBySuperRegion.reindex(sort_inds, axis=1)
p_vals_sorted = p_vals.reindex(sort_inds, axis=1)

print(p_vals_sorted)

sns.stripplot(data=DiffBySuperRegion_sorted, color='k')
sns.violinplot(data=DiffBySuperRegion_sorted, palette=sns.color_palette('deep', 8))

# ax.set_ylim([-1.2, 1.2])
ax.set_ylabel('Region avg. difference (FC - SC)')

colors = sns.color_palette('deep', 8)

sns.palplot(colors)
np.array(colors)
# fig3_2.savefig(os.path.join(analysis_dir, 'figpanels', 'fig3_2.svg'), format='svg', transparent=True)
# %%
from scipy.stats import ttest_1samp, ttest_ind

# log transform anatomical connectivity values
anatomical_mat = AC.getConnectivityMatrix('CellCount', diag=0).to_numpy().copy()
keep_inds_diff = np.where(anatomical_mat > 0)
anatomical_adj = np.log10(anatomical_mat[keep_inds_diff])

diff_by_region = []
for c_ind in range(FC.cmats.shape[2]):  #loop over fly
    cmat = FC.cmats[:, :, c_ind]
    functional_adj = cmat[keep_inds_diff]

    F_zscore_fly = zscore(functional_adj)
    A_zscore_fly = zscore(anatomical_adj)

    diff = F_zscore_fly - A_zscore_fly

    diff_m = np.zeros_like(anatomical_mat)
    diff_m[keep_inds_diff] = diff
    diff_by_region.append(diff_m.mean(axis=0))

diff_by_region = np.vstack(diff_by_region).T  # region x fly

colors = sns.color_palette('deep', 8)
fh, ax = plt.subplots(1, 1, figsize=(7, 3.5))
plot_ct = 0
for r_ind, reg in enumerate(regions):
    in_inds = np.where([r in regions[reg] for r in FC.rois])[0]
    # ax.annotate(reg, (plot_ct, 1.5))
    for i, included in enumerate(in_inds):
        new_mean = np.mean(diff_by_region[included,:])
        new_err = np.std(diff_by_region[included,:]) / np.sqrt(diff_by_region.shape[1])
        ax.plot(plot_ct, new_mean, linestyle='None', marker='o', color=colors[r_ind])
        ax.plot([plot_ct, plot_ct], [new_mean-new_err, new_mean+new_err], linestyle='-', linewidth=2, marker='None', color=colors[r_ind])
        ax.annotate(np.array(FC.rois)[included], (plot_ct-0.25, 1.1), rotation=90, fontsize=8)
        plot_ct+=1
    plot_ct +=1

ax.set_ylim([-1.5, 1.5])
ax.set_xticks([])
ax.spines['right'].set_visible(False)
ax.axhline(0, color=[0.8, 0.8, 0.8], linestyle='-', zorder=0)
ax.set_ylabel('Region avg. difference (FC - SC)')

# %%

anat_connect = AC.getConnectivityMatrix('CellCount', diag=None)
shortest_path_distance, shortest_path_steps, shortest_path_weight, hub_count = bridge.getShortestPathStats(anat_connect)

# %%

plt.plot(np.mean(shortest_path_distance, axis=1), np.mean(diff_by_region, axis=1), 'ko')

# %%
direct_dist = (1/AC.getConnectivityMatrix('CellCount', diag=None).to_numpy()).ravel()
fh, ax = plt.subplots(1, 2, figsize=(8, 4))
c = DifferenceMatrix.to_numpy().ravel()

sc = ax[0].scatter(direct_dist, shortest_path_distance.to_numpy().ravel(), c=c, alpha=1, cmap='Blues', marker='.')

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('Direct distance (1/cells)')
ax[0].set_ylabel('Shortest path distance (1/cells)');
ax[0].set_ylim([2e-4, 3e-2])
fh.colorbar(sc, ax=ax[0])


shortest_path_factor = direct_dist / shortest_path_distance.to_numpy().ravel()
x = shortest_path_factor
y = DifferenceMatrix.to_numpy().ravel()
keep_inds = np.where(x>1)

x = x[keep_inds]
y = y[keep_inds]


ax[1].plot(x, y, color=[0.8, 0.8, 0.8], marker='.', alpha=0.25, linestyle='None')
ax[1].set_xscale('log')
ax[1].set_xlabel('Shortest path factor')
ax[1].set_ylabel('Diff. (FC-SC)')
r, p = spearmanr(shortest_path_factor, DifferenceMatrix.to_numpy().ravel())
print('Spearman r is {}'.format(r))

num_bins = 18 # equally populated bins
points_per_bin = int(len(x)/num_bins)
for b_ind in range(num_bins):
    inds = np.argsort(x)[(b_ind*points_per_bin):(b_ind+1)*points_per_bin]
    bin_mean_x = x[inds].mean()
    bin_mean_y = y[inds].mean()
    ax[1].plot(bin_mean_x, bin_mean_y, color=plot_colors[0], marker='s', alpha=1, linestyle='none')

    err_x = x[inds].std()/np.sqrt(len(inds))
    ax[1].plot([bin_mean_x - err_x, bin_mean_x + err_x], [bin_mean_y, bin_mean_y], linestyle='-', marker='None', color=plot_colors[0], alpha=1, linewidth=2)

    err_y = y[inds].std()/np.sqrt(len(inds))
    ax[1].plot([bin_mean_x, bin_mean_x], [bin_mean_y - err_y, bin_mean_y + err_y], linestyle='-', marker='None', color=plot_colors[0], alpha=1, linewidth=2)


# %%
import networkx as nx
from node2vec import Node2Vec
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist, squareform


# 1) Embed anatomical graph using node2vec
max_path = 6
n_nodes = len(FC.rois)

adj = AC.getConnectivityMatrix('CellCount', symmetrize=True, diag=0).to_numpy()
# adj = FC.CorrelationMatrix.to_numpy()
# np.fill_diagonal(adj, 0)
adj[adj<0] = 0
G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
n2v = Node2Vec(graph=G, walk_length=2*max_path, num_walks=20*n_nodes, dimensions=n_nodes, q=0.5, p=1)
w2v = n2v.fit(sg=0, seed=1)
embedding_w2v = np.vstack([np.array(w2v[str(u)]) for u in sorted(G.nodes)]) # n_nodes x n_dimensions


# %%
# 2) do SVD/PCA on embeddings to visualize in 2D
c = DifferenceMatrix.mean().to_numpy()
lim = np.max(np.abs(c))
u, s, vh = np.linalg.svd(embedding_w2v, full_matrices=False)

fig3_4, ax = plt.subplots(1, 1, figsize=(4, 2))
ax.plot(s / np.sum(s), 'ko')
ax.set_xlabel('Mode')
ax.set_ylabel('Frac. var')
ax.set_xticks([])


fig3_5, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(u[:, 0], u[:, 1], c=c, cmap="RdBu", vmin=-lim, vmax=lim, s=80, edgecolors='k',)
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')



for r_ind, r in enumerate(AC.rois):
    ax.annotate(r, (u[r_ind, 0], u[r_ind, 1]), fontsize=8, fontweight='bold')

# %%
plt.plot(u[:, 0], c, 'ko')


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


fig4_0, ax = plt.subplots(1, 1, figsize=(1.6, 1.6))
lim = np.nanmax(np.abs(DifferenceMatrix.to_numpy().ravel()))
ax.scatter(10**anatomical_adjacency_diff, functional_adjacency_diff, alpha=1, color='k', marker='.')
ax.set_xscale('log')
ax.set_xlim([np.min(10**anatomical_adjacency_diff), np.max(10**anatomical_adjacency_diff)])
ax.set_xlabel('SC (cells)', fontsize=10)
ax.set_ylabel('FC (z)', fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=8)


fig4_1, ax = plt.subplots(1, 1, figsize=(1.6, 1.6))
lim = np.nanmax(np.abs(DifferenceMatrix.to_numpy().ravel()))
ax.scatter(A_zscore, F_zscore, alpha=1, c=diff, cmap="RdBu",  vmin=-lim, vmax=lim, marker='.')
ax.plot([-3.5, 3.5], [-3.5, 3.5], 'k-')
ax.axhline(color='k', zorder=0, alpha=0.5)
ax.axvline(color='k', zorder=0, alpha=0.5)
ax.set_xticks([-2, 2])
ax.set_yticks([-2, 2])
ax.set_xticklabels(['-2$\sigma$', '+2$\sigma$'])
ax.set_yticklabels(['-2$\sigma$', '+2$\sigma$'])
ax.set_xlabel('SC', fontsize=10)
ax.set_ylabel('FC', fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.set_aspect(1)

fig4_2, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.heatmap(sorted_diff, ax=ax, yticklabels=True, xticklabels=True, cbar_kws={'label': 'Difference (FC - SC)','shrink': .65}, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
ax.set_aspect('equal')
ax.tick_params(axis='both', which='major', labelsize=6)

fig4_0.savefig(os.path.join(analysis_dir, 'figpanels', 'fig4_0.svg'), format='svg', transparent=True)
fig4_1.savefig(os.path.join(analysis_dir, 'figpanels', 'fig4_1.svg'), format='svg', transparent=True)
fig4_2.savefig(os.path.join(analysis_dir, 'figpanels', 'fig4_2.svg'), format='svg', transparent=True)


# %% Average diff for each region, cluster and sort by super-regions

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
fig4_3, ax = plt.subplots(1, 1, figsize=(5.5, 3.0))
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

ax.set_ylim([-1.1, 1.1])
ax.spines['right'].set_visible(False)
ax.axhline(0, color=[0.8, 0.8, 0.8], linestyle='-', zorder=0)
ax.set_ylabel('Region avg. diff.\n(FC - SC)')

sns.palplot(colors)
np.array(colors)
fig4_3.savefig(os.path.join(analysis_dir, 'figpanels', 'fig4_3.svg'), format='svg', transparent=True)

# %%
# Shortest path distance:
anat_connect = AC.getConnectivityMatrix('CellCount', diag=None)
shortest_path_dist, shortest_path_steps, shortest_path_weight, hub_count = bridge.getShortestPathStats(anat_connect)

shortest_path_dist = shortest_path_dist.to_numpy()[~np.eye(36,dtype=bool)]

# Direct distance:
direct_dist = (1/AC.getConnectivityMatrix('CellCount', diag=0).to_numpy())[~np.eye(36,dtype=bool)]
direct_dist[np.isinf(direct_dist)] = np.nan

# FC-SC difference:
diff = DifferenceMatrix.to_numpy()[~np.eye(36,dtype=bool)]

fig4_4, ax = plt.subplots(1, 2, figsize=(6, 3))
lim = np.nanmax(np.abs(DifferenceMatrix.to_numpy().ravel()))
sc = ax[0].scatter(direct_dist, shortest_path_dist, c=diff, alpha=1, cmap='RdBu', marker='.', vmin=-lim, vmax=lim)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('Direct distance')
ax[0].set_ylabel('Shortest path distance');
ax[0].plot([2e-4, 1], [2e-4, 1], color='k', linewidth=2, alpha=1.0, linestyle='-', zorder=0)
ax[0].set_ylim([2e-4, 6e-2])
fig4_4.colorbar(sc, ax=ax[0])

shortest_path_factor = direct_dist / shortest_path_dist
x = shortest_path_factor
y = diff
keep_inds = np.where(x>1)

x = x[keep_inds]
y = y[keep_inds]

ax[1].scatter(x, y, c=diff[keep_inds], marker='.', alpha=1.0, linestyle='None', cmap='RdBu', vmin=-lim, vmax=lim)
ax[1].axhline(color='k', linestyle='--')
ax[1].set_xscale('log')
ax[1].set_xlabel(r'Indirect path factor: $\dfrac{D_{direct}}{D_{shortest}}$')
ax[1].set_ylabel('Diff. (FC-SC)')
r, p = spearmanr(x, y)
ax[1].annotate(r'$\rho$={:.2f}'.format(r), (90, 2.5))

bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 10)
num_bins = len(bins)-1

for b_ind in range(num_bins):
    b_start = bins[b_ind]
    b_end = bins[b_ind+1]
    inds = np.where(np.logical_and(x > b_start, x < b_end))
    bin_mean_x = x[inds].mean()
    bin_mean_y = y[inds].mean()
    ax[1].plot(bin_mean_x, bin_mean_y, color='k', marker='s', alpha=1, linestyle='none')

    err_x = x[inds].std()/np.sqrt(len(inds))
    ax[1].plot([bin_mean_x - err_x, bin_mean_x + err_x], [bin_mean_y, bin_mean_y], linestyle='-', marker='None', color='k', alpha=1, linewidth=2)

    err_y = y[inds].std()/np.sqrt(len(inds))
    ax[1].plot([bin_mean_x, bin_mean_x], [bin_mean_y - err_y, bin_mean_y + err_y], linestyle='-', marker='None', color='k', alpha=1, linewidth=2)

fig4_4.savefig(os.path.join(analysis_dir, 'figpanels', 'fig4_4.svg'), format='svg', transparent=True)

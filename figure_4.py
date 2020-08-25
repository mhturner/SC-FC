import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import os
from scipy.stats import zscore
import pandas as pd
import seaborn as sns

from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
import matplotlib
from matplotlib import rcParams
rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})

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
diff = A_zscore - F_zscore


diff_m = np.zeros_like(anatomical_mat)
diff_m[keep_inds_diff] = diff
DifferenceMatrix = pd.DataFrame(data=diff_m, index=FC.rois, columns=FC.rois)


# %% sort difference matrix by most to least different rois
diff_by_region = DifferenceMatrix.mean()
sort_inds = np.argsort(diff_by_region)
sort_keys = DifferenceMatrix.index[sort_inds]
sorted_diff = pd.DataFrame(data=np.zeros_like(DifferenceMatrix),columns=sort_keys, index=sort_keys)
for r_ind, r_key in enumerate(sort_keys):
    for c_ind, c_key in enumerate(sort_keys):
        sorted_diff.iloc[r_ind, c_ind]=DifferenceMatrix.loc[[r_key], [c_key]].to_numpy()

fig4_0, ax = plt.subplots(1, 1, figsize=(4, 4))
lim = np.nanmax(np.abs(DifferenceMatrix.to_numpy().ravel()))
ax.scatter(A_zscore, F_zscore, alpha=1, c=diff, cmap="RdBu",  vmin=-lim, vmax=lim, edgecolors='k', linewidths=0.5)
ax.plot([-3, 4], [-3, 4], 'k-')
ax.set_xlabel('Anatomical ajacency (z-score)')
ax.set_ylabel('Functional correlation (z-score)');
# ax.set_xticks([-2, 0, 3])
# ax.set_yticks([-3, 0, 3])

fig4_1, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.heatmap(sorted_diff, ax=ax, yticklabels=True, xticklabels=True, cbar_kws={'label': 'Difference (SC - FC)','shrink': .75}, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
ax.set_aspect('equal')
ax.tick_params(axis='both', which='major', labelsize=7)

diff_by_region = DifferenceMatrix.mean()
diff_brain = np.zeros(shape=FC.roi_mask[0].shape)
diff_brain[:] = np.nan
for r_ind, r in enumerate(FC.roi_mask):
    diff_brain[r] = diff_by_region[r_ind]


zslices = np.linspace(5, 60, 8)
lim = np.nanmax(np.abs(diff_brain.ravel()))

fig4_2 = plt.figure(figsize=(8, 4))
for z_ind, z in enumerate(zslices):
    ax = fig4_2.add_subplot(2, 4, z_ind+1)
    img = ax.imshow(diff_brain[:, :, int(z)].T, cmap="RdBu", rasterized=False, vmin=-lim, vmax=lim)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_xlim([0, 102])
    ax.set_ylim([107, 5])

fig4_3, ax = plt.subplots(1, 1, figsize=(1, 3))
ax.set_axis_off()
cb = fig4_3.colorbar(img, ax=ax)
cb.set_label(label='Region-average diff.', weight='bold', color='k')
cb.ax.tick_params(labelsize=12, color='k')


fig4_0.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig4_0.svg'), format='svg', transparent=True)
fig4_1.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig4_1.svg'), format='svg', transparent=True)
fig4_2.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig4_2.svg'), format='svg', transparent=True)
fig4_3.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig4_3.svg'), format='svg', transparent=True)

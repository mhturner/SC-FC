import os
import glob
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import nibabel as nib
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from region_connectivity import RegionConnectivity
import networkx as nx
from scipy.stats import pearsonr, spearmanr, ttest_1samp
from scipy.ndimage.measurements import center_of_mass

from operator import itemgetter

data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')
roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')
# %%

# # # # ATLAS STUFF AND REGION LOCATIONS # # # #
mapping = RegionConnectivity.getRoiMapping()
rois = list(mapping.keys())
rois.sort()

roi_mask, roi_size = RegionConnectivity.loadAtlasData(atlas_path=atlas_path, roinames_path=roinames_path, mapping=mapping)
# find center of mass for each roi
coms = np.vstack([center_of_mass(x) for x in roi_mask])
anat_position = {}
for r in range(len(coms)):
    anat_position[r] = coms[r, :]

# # # # STRUCTURAL ADJACENCY MATRIX # # # #
WeakConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'WeakConnections_computed_20200626.pkl'))
MediumConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'MediumConnections_computed_20200626.pkl'))
StrongConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'StrongConnections_computed_20200626.pkl'))
conn_mat = WeakConnections + MediumConnections + StrongConnections
# set diag to nan
tmp_mat = conn_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)
ConnectivityCount_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=conn_mat.index, columns=conn_mat.index)
ConnectivityCount = pd.DataFrame(data=tmp_mat, index=conn_mat.index, columns=conn_mat.index)
adjacency_anat = ConnectivityCount_Symmetrized.to_numpy().copy()
np.fill_diagonal(adjacency_anat, 0)

# # # # FUNCTIONAL ADJACENCY MATRIX  # # # #
response_filepaths = glob.glob(os.path.join(data_dir, 'region_responses') + '/' + '*.pkl')
roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')
CorrelationMatrix_Functional, cmats = RegionConnectivity.getFunctionalConnectivity(response_filepaths, cutoff=0.01, fs=1.2)
adjacency_fxn = CorrelationMatrix_Functional.to_numpy().copy()
np.fill_diagonal(adjacency_fxn, 0)

# significance test on fxnal cmat
upper_inds = np.triu_indices(CorrelationMatrix_Functional.shape[0], k=1) # k=1 excludes main diagonal
num_comparisons = len(upper_inds[0])
p_cutoff = 0.01 / num_comparisons # bonferroni
t, p = ttest_1samp(cmats, 0, axis=2) # ttest against 0
np.fill_diagonal(p, 1) # replace nans in diag with p=1
adjacency_fxn[p>p_cutoff] = 0 # set nonsig regions to 0
print('Ttest included {} significant of {} total regions in fxnal connectivity matrix'.format((p<p_cutoff).sum(), p.size))

# Plot clustering and degree using full adjacency to make graphs
G_anat = nx.from_numpy_matrix(adjacency_anat/adjacency_anat.max())
G_fxn = nx.from_numpy_matrix(adjacency_fxn/adjacency_fxn.max())

fh, ax = plt.subplots(1, 2, figsize=(8, 4))
clust_fxn = list(nx.clustering(G_fxn, weight='weight').values())
clust_anat = list(nx.clustering(G_anat, weight='weight').values())
r, p = spearmanr(clust_anat, clust_fxn)
ax[0].set_title('Clustering, r = {:.3f}'.format(r))
ax[0].plot(clust_anat, clust_fxn, 'ko')
ax[0].set_xlabel('Structural')
ax[0].set_ylabel('Functional')

deg_fxn = [val for (node, val) in G_fxn.degree(weight='weight')]
deg_anat = [val for (node, val) in G_anat.degree(weight='weight')]
r, p = spearmanr(deg_anat, deg_fxn)
ax[1].set_title('Degree, r = {:.3f}'.format(r))
ax[1].plot(deg_anat, deg_fxn, 'ko')
ax[1].set_xlabel('Structural')
ax[1].set_ylabel('Functional')

# # # # # plot network graph with top x% of connections
take_top_pct = 0.2 # top fraction to include in network graphs
roilabels_to_skip = ['LAL(R)', 'CRE(R)', 'CRE(L)', 'EPA(R)','BU(R)']
cmap = plt.get_cmap('Greys')


cutoff = np.quantile(adjacency_anat, 1-take_top_pct)
print('Threshold included {} of {} regions in anatomical connectivity matrix'.format((adjacency_anat>=cutoff).sum(), adjacency_anat.size))
temp_adj_anat = adjacency_anat.copy()
temp_adj_anat[temp_adj_anat<cutoff] = 0
G_anat = nx.from_numpy_matrix(temp_adj_anat/temp_adj_anat.max())



cutoff = np.quantile(adjacency_fxn[adjacency_fxn>0], 1-take_top_pct)
print('Threshold included {} of {} sig regions in functional connectivity matrix'.format((adjacency_fxn>=cutoff).sum(), (adjacency_fxn>0).sum()))
temp_adj_fxn = adjacency_fxn.copy()
temp_adj_fxn[temp_adj_fxn<cutoff] = 0
G_fxn = nx.from_numpy_matrix(temp_adj_fxn/temp_adj_fxn.max())

fh = plt.figure(figsize=(16,8))
ax_anat = fh.add_subplot(1, 2, 1, projection='3d')
ax_fxn = fh.add_subplot(1, 2, 2, projection='3d')

ax_anat.view_init(-145, -95)
ax_anat.set_axis_off()
ax_anat.set_title('Structural', fontweight='bold', fontsize=12)

ax_fxn.view_init(-145, -95)
ax_fxn.set_axis_off()
ax_fxn.set_title('Functional', fontweight='bold', fontsize=12)

for key, value in anat_position.items():
    xi = value[0]
    yi = value[1]
    zi = value[2]

    # Plot nodes
    ax_anat.scatter(xi, yi, zi, c='b', s=5+40*G_anat.degree(weight='weight')[key], edgecolors='k', alpha=0.25)
    ax_fxn.scatter(xi, yi, zi, c='b', s=5+20*G_fxn.degree(weight='weight')[key], edgecolors='k', alpha=0.25)
    if rois[key] not in roilabels_to_skip:
        ax_anat.text(xi, yi, zi+2, rois[key], zdir=(0,0,0), fontsize=8, fontweight='bold')
        ax_fxn.text(xi, yi, zi+2, rois[key], zdir=(0,0,0), fontsize=8, fontweight='bold')

    ctr = [15, 70, 60]
    dstep=10
    ax_anat.plot([ctr[0], ctr[0]+dstep], [ctr[1], ctr[1]], [ctr[2], ctr[2]], 'r') # x
    ax_anat.plot([ctr[0], ctr[0]], [ctr[1], ctr[1]-dstep], [ctr[2], ctr[2]], 'g') # y
    ax_anat.plot([ctr[0], ctr[0]], [ctr[1], ctr[1]], [ctr[2], ctr[2]-dstep], 'b') # z

    ax_fxn.plot([ctr[0], ctr[0]+dstep], [ctr[1], ctr[1]], [ctr[2], ctr[2]], 'r') # x
    ax_fxn.plot([ctr[0], ctr[0]], [ctr[1], ctr[1]-dstep], [ctr[2], ctr[2]], 'g') # y
    ax_fxn.plot([ctr[0], ctr[0]], [ctr[1], ctr[1]], [ctr[2], ctr[2]-dstep], 'b') # z


# plot connections
for i,j in enumerate(G_anat.edges()):
    x = np.array((anat_position[j[0]][0], anat_position[j[1]][0]))
    y = np.array((anat_position[j[0]][1], anat_position[j[1]][1]))
    z = np.array((anat_position[j[0]][2], anat_position[j[1]][2]))

    # Plot the connecting lines
    line_wt = (G_anat.get_edge_data(j[0], j[1], default={'weight':0})['weight'] + G_anat.get_edge_data(j[1], j[0], default={'weight':0})['weight'])/2
    color = cmap(line_wt)
    ax_anat.plot(x, y, z, c=color, alpha=line_wt, linewidth=2)

    line_wt = (G_fxn.get_edge_data(j[0], j[1], default={'weight':0})['weight'] + G_fxn.get_edge_data(j[1], j[0], default={'weight':0})['weight'])/2
    color = cmap(line_wt)
    ax_fxn.plot(x, y, z, c=color, alpha=line_wt, linewidth=2)

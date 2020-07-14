import os
import glob
import pandas as pd
import numpy as np
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
mapping = RegionConnectivity.getRoiMapping()
rois = list(mapping.keys())
rois.sort()

# 1) ConnectivityCount
WeakConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'WeakConnections_computed_20200626.pkl'))
MediumConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'MediumConnections_computed_20200626.pkl'))
StrongConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'StrongConnections_computed_20200626.pkl'))
conn_mat = WeakConnections + MediumConnections + StrongConnections
# set diag to nan
tmp_mat = conn_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)
ConnectivityCount_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=conn_mat.index, columns=conn_mat.index)
ConnectivityCount = pd.DataFrame(data=tmp_mat, index=conn_mat.index, columns=conn_mat.index)

theta_anat = 0.0
D_anat = ConnectivityCount_Symmetrized.to_numpy().copy()
np.fill_diagonal(D_anat, 0)
D_anat = D_anat / D_anat.max()
D_anat[D_anat<theta_anat] = 0
G_anat = nx.from_numpy_matrix(D_anat)



#   FXNAL
theta_fxn = 0.0
response_filepaths = glob.glob(os.path.join(data_dir, 'region_responses') + '/' + '*.pkl')
roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')
fs = 1.2 # Hz
cutoff = 0.01 # Hz

upper_inds = np.triu_indices(ConnectivityCount_Symmetrized.shape[0], k=1) # k=1 excludes main diagonal
num_comparisons = len(upper_inds[0])
p_cutoff = 0.001 / num_comparisons
CorrelationMatrix_Functional, cmats = RegionConnectivity.getFunctionalConnectivity(response_filepaths, cutoff=cutoff, fs=fs)
t, p = ttest_1samp(cmats, 0, axis=2)
CorrelationMatrix_Functional[p>p_cutoff] = 0
print('Ttest Excluded {} of {} regions in fxnal connectivity matrix'.format((p>p_cutoff).sum(), p.size))

roi_mask, roi_size = RegionConnectivity.loadAtlasData(atlas_path=atlas_path, roinames_path=roinames_path, mapping=mapping)
# find center of mass for each roi
coms = np.vstack([center_of_mass(x) for x in roi_mask])

D_fxn = CorrelationMatrix_Functional.to_numpy().copy()
np.fill_diagonal(D_fxn, 0)
D_fxn = D_fxn / D_fxn.max()
D_fxn[D_fxn<theta_fxn] = 0
G_fxn = nx.from_numpy_matrix(D_fxn)


# %%
anat_position = {}
for r in range(D_anat.shape[0]):
    anat_position[r] = coms[r, :]

# %%

cmap = plt.get_cmap('Greys')

with plt.style.context(('ggplot')):

    fig_anat = plt.figure(figsize=(10,7))
    ax_anat = Axes3D(fig_anat)

    fig_fxn = plt.figure(figsize=(10,7))
    ax_fxn = Axes3D(fig_fxn)

    for key, value in anat_position.items():
        xi = value[0]
        yi = value[1]
        zi = value[2]

        # Plot nodes
        ax_anat.scatter(xi, yi, zi, c='b', s=40*G_anat.degree(weight='weight')[key], edgecolors='k', alpha=0.25)
        ax_anat.text(xi, yi, zi+2, rois[key], zdir=(1,1,0))

        ax_fxn.scatter(xi, yi, zi, c='b', s=20*G_fxn.degree(weight='weight')[key], edgecolors='k', alpha=0.25)
        ax_fxn.text(xi, yi, zi+2, rois[key], zdir=(1,1,0))

    # plot lines
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


# %%



# %%
node_and_degree = G_fxn.degree()
(largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]
hub_ego = nx.ego_graph(G_fxn, largest_hub)
pos = nx.spring_layout(hub_ego)

nx.draw(hub_ego, pos, node_color='b', node_size=50, with_labels=True, font_weight='bold', font_color='r')

# %%



fh, ax = plt.subplots(1, 3, figsize=(12, 4))
clust_fxn = list(nx.clustering(G_fxn, weight='weight').values())
clust_anat = list(nx.clustering(G_anat, weight='weight').values())
r, p = spearmanr(clust_anat, clust_fxn)
ax[0].set_title('Clustering, r = {:.3f}'.format(r))
ax[0].plot(clust_anat, clust_fxn, 'ko')

deg_fxn = [val for (node, val) in G_fxn.degree(weight='weight')]
deg_anat = [val for (node, val) in G_anat.degree(weight='weight')]

r, p = spearmanr(deg_anat, deg_fxn)
ax[1].set_title('Degree, r = {:.3f}'.format(r))
ax[1].plot(deg_anat, deg_fxn, 'ko')


# %%

nx.flow_hierarchy(G_anat, weight='weight')

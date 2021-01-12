"""
Turner, Mann, Clandinin: scratch

https://github.com/mhturner/SC-FC
"""

import matplotlib.pyplot as plt
from neuprint import Client, fetch_neurons, NeuronCriteria
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

mp = bridge.getRoiMapping()
mp

# Get FunctionalConnectivity object
FC = functional_connectivity.FunctionalConnectivity(data_dir=data_dir, fs=1.2, cutoff=0.01, mapping=bridge.getRoiMapping())

# Get AnatomicalConnectivity object
AC = anatomical_connectivity.AnatomicalConnectivity(data_dir=data_dir, neuprint_client=neuprint_client, mapping=bridge.getRoiMapping())
# %%
Neur, Syn = fetch_neurons(NeuronCriteria(inputRois='AL(R)', outputRois='LH(R)', status='Traced'))
np.unique(Neur.bodyId).shape



# %%
import os
import pandas as pd
computed_date = '20210108'
WeakConnections = pd.read_pickle(os.path.join(AC.data_dir, 'connectome_connectivity', 'uncropped_WeakConnections_computed_{}.pkl'.format(computed_date)))
MediumConnections = pd.read_pickle(os.path.join(AC.data_dir, 'connectome_connectivity', 'uncropped_MediumConnections_computed_{}.pkl'.format(computed_date)))
StrongConnections = pd.read_pickle(os.path.join(AC.data_dir, 'connectome_connectivity', 'uncropped_StrongConnections_computed_{}.pkl'.format(computed_date)))

conn_mat_u = WeakConnections + MediumConnections + StrongConnections

# %%
computed_date = '20200909'
WeakConnections = pd.read_pickle(os.path.join(AC.data_dir, 'connectome_connectivity', 'WeakConnections_computed_{}.pkl'.format(computed_date)))
MediumConnections = pd.read_pickle(os.path.join(AC.data_dir, 'connectome_connectivity', 'MediumConnections_computed_{}.pkl'.format(computed_date)))
StrongConnections = pd.read_pickle(os.path.join(AC.data_dir, 'connectome_connectivity', 'StrongConnections_computed_{}.pkl'.format(computed_date)))

conn_mat = WeakConnections + MediumConnections + StrongConnections

# %%

fh, ax = plt.subplots(1, 1, figsize=(4,4))
ax.plot(conn_mat, conn_mat_u, 'ko');
ax.plot([0, 8000], [0, 8000])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Connecting cells')
ax.set_ylabel('Connecting cells (uncropped only)')

r, p = pearsonr(conn_mat.to_numpy().ravel(), conn_mat_u.to_numpy().ravel())
r
ax.annotate('r={:.2f}'.format(r), (1,6e3));

# %%
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
r
anatomical_adjacency.shape


# %% HUB score analysis

# Shortest path distance:
anat_connect = AC.getConnectivityMatrix('CellCount', diag=None)
shortest_path_dist, shortest_path_steps, shortest_path_weight, hub_count = bridge.getShortestPathStats(anat_connect)

# %%
hub_count
hub_count.sort_values('count', ascending=False)



hub_count
fh1, ax1 = plt.subplots(1, 1, figsize=(8, 4))
sns.barplot(x=hub_count.sort_values('count', ascending=False).index, y=np.squeeze(hub_count.sort_values('count', ascending=False).values) / shortest_path_steps.size, ax=ax1)
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
    tick.set_fontsize(12)
ax1.set_ylabel('Hub score')
ax1.set_ylim([0, 0.5])
# %%

conn = AC.getConnectivityMatrix('CellCount', diag=np.nan)
fh2, ax2 = plt.subplots(1, 2, figsize=(8, 4))

region_1 = 'PVLP(R)'
region_2 = 'WED(R)'

ax2[0].plot(conn[region_1], conn[region_2], 'ko')
ax2[0].set_xlabel('Outgoing from {}'.format(region_1))
ax2[0].set_ylabel('Outgoing from {}'.format(region_2))
for r_ind, r in enumerate(FC.rois):
    ax2[0].annotate(r, (conn[region_1][r_ind]+100, conn[region_2][r_ind]-20), fontsize=8, fontweight='bold')

ax2[0].set_xlim([0, 4000])

region_1 = 'AL(R)'
region_2 = 'WED(R)'
ax2[1].plot(conn[region_1], conn[region_2], 'ko')
ax2[1].set_xlabel('Outgoing from {}'.format(region_1))
ax2[1].set_ylabel('Outgoing from {}'.format(region_2))
for r_ind, r in enumerate(FC.rois):
    ax2[1].annotate(r, (conn[region_1][r_ind]+5, conn[region_2][r_ind]-20), fontsize=8, fontweight='bold')


fh1.savefig(os.path.join(analysis_dir, 'figpanels', 'fig_hub_1.svg'), format='svg', transparent=True, dpi=400)
fh2.savefig(os.path.join(analysis_dir, 'figpanels', 'fig_hub_2.svg'), format='svg', transparent=True, dpi=400)

# %%

import matplotlib.pyplot as plt
from neuprint import Client, fetch_neurons, NeuronCriteria
import numpy as np
import networkx as nx
from scipy.stats import pearsonr, spearmanr
import os
import socket

from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
import matplotlib
from matplotlib import rcParams
rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})
rcParams.update({'axes.spines.right': False})
rcParams.update({'axes.spines.top': False})
rcParams['svg.fonttype'] = 'none' # let illustrator handle the font type

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


# %%
# %% regression model on shortest path steps
fig2_6, ax = plt.subplots(1, 3, figsize=(9, 3.5))

rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=0)

# 1: Cell count
anat_connect = AC.getConnectivityMatrix('CellCount', diag=None)
shortest_path_distance, shortest_path_steps, shortest_path_weight, hub_count = bridge.getShortestPathStats(anat_connect)
x = np.log10(((shortest_path_distance.T + shortest_path_distance.T)/2).to_numpy()[FC.upper_inds])
y = FC.CorrelationMatrix.to_numpy()[FC.upper_inds]
x = x.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(x, y);
pred = regressor.predict(x)
cv_results = cross_validate(regressor, x, y, cv=rkf, scoring='r2');
avg_r2 = cv_results['test_score'].mean()
err = cv_results['test_score'].std()
print('r2 = {:.2f}+/-{:.2f}'.format(avg_r2, err))
ax[0].plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
ax[0].plot(pred, y, 'ko', alpha=0.25)
ax[0].annotate('$r^2$={:.2f}'.format(avg_r2), (-0.15, 0.95))
ax[0].set_ylabel('Measured FC (z)')
ax[0].set_xlim([-0.2, 1.0])
ax[0].set_title('Cell count', fontsize=10)
ax[0].set_aspect('equal')

# 2: Synapse count
anat_connect = AC.getConnectivityMatrix('WeightedSynapseCount', diag=None)
shortest_path_distance, shortest_path_steps, shortest_path_weight, hub_count = bridge.getShortestPathStats(anat_connect)
x = np.log10(((shortest_path_distance.T + shortest_path_distance.T)/2).to_numpy()[FC.upper_inds])
y = FC.CorrelationMatrix.to_numpy()[FC.upper_inds]
x = x.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(x, y);
pred = regressor.predict(x)
cv_results = cross_validate(regressor, x, y, cv=rkf, scoring='r2');
avg_r2 = cv_results['test_score'].mean()
err = cv_results['test_score'].std()
print('r2 = {:.2f}+/-{:.2f}'.format(avg_r2, err))
ax[1].plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
ax[1].plot(pred, y, 'ko', alpha=0.25)
ax[1].annotate('$r^2$={:.2f}'.format(avg_r2), (-0.15, 0.95))
ax[1].set_xlabel('Predicted FC (z)')
ax[1].set_xlim([-0.2, 1.0])
ax[1].set_title('Weighted synapse count', fontsize=10)
ax[1].set_aspect('equal')

# 3: Tbars
anat_connect = AC.getConnectivityMatrix('TBars', diag=None)
shortest_path_distance, shortest_path_steps, shortest_path_weight, hub_count = bridge.getShortestPathStats(anat_connect)
x = np.log10(((shortest_path_distance.T + shortest_path_distance.T)/2).to_numpy()[FC.upper_inds])
y = FC.CorrelationMatrix.to_numpy()[FC.upper_inds]
x = x.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(x, y);
pred = regressor.predict(x)
cv_results = cross_validate(regressor, x, y, cv=rkf, scoring='r2');
avg_r2 = cv_results['test_score'].mean()
err = cv_results['test_score'].std()
print('r2 = {:.2f}+/-{:.2f}'.format(avg_r2, err))
ax[2].plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
ax[2].plot(pred, y, 'ko', alpha=0.25)
ax[2].annotate('$r^2$={:.2f}'.format(avg_r2), (-0.15, 0.95))
ax[2].set_xlim([-0.2, 1.0])
ax[2].set_title('T-Bar count', fontsize=10)
ax[2].set_aspect('equal')

fig2_6.suptitle('Shortest path connectivity')
# fig2_6.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig2_6.pdf'), format='pdf', transparent=True)

# %% multiple regression model to try to get highest r2

fh, ax = plt.subplots(1, 1, figsize=(4, 4))
# Cell ct + tbars
sp_cell_count, _, _, _ = bridge.getShortestPathStats(AC.getConnectivityMatrix('CellCount', diag=None))
sp_tbars, _, _, _ = bridge.getShortestPathStats(AC.getConnectivityMatrix('TBars', diag=None))

x = np.vstack([
               np.log10(((sp_cell_count.T + sp_cell_count.T)/2).to_numpy()[FC.upper_inds]),
               np.log10(((sp_tbars.T + sp_tbars.T)/2).to_numpy()[FC.upper_inds]),
               FC.SizeMatrix.to_numpy()[FC.upper_inds],
               FC.DistanceMatrix.to_numpy()[FC.upper_inds],
               ]).T


y = FC.CorrelationMatrix.to_numpy()[FC.upper_inds]
# x = x.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(x, y);
pred = regressor.predict(x)
cv_results = cross_validate(regressor, x, y, cv=rkf, scoring='r2');
avg_r2 = cv_results['test_score'].mean()
err = cv_results['test_score'].std()
print('r2 = {:.2f}+/-{:.2f}'.format(avg_r2, err))
ax.plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
ax.plot(pred, y, 'ko', alpha=0.25)
ax.annotate('$r^2$={:.2f}'.format(avg_r2), (-0.15, 0.95))
ax.set_xlim([-0.2, 1.0])
ax.set_title('Multiple regression model', fontsize=10)
ax.set_aspect('equal')

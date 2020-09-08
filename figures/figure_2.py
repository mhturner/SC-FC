import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_validate, RepeatedKFold
import socket

from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
import matplotlib
from matplotlib import rcParams
rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})
rcParams.update({'axes.spines.right': False})
rcParams.update({'axes.spines.top': False})
rcParams['svg.fonttype'] = 'none' # let illustrator handle the font type
rcParams['pdf.fonttype'] = 42

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

fig2_0, ax = plt.subplots(1, 2, figsize=(10, 5))
df = AC.getConnectivityMatrix('CellCount', diag=np.nan)
sns.heatmap(np.log10(AC.getConnectivityMatrix('CellCount', diag=np.nan)).replace([np.inf, -np.inf], 0), ax=ax[0], yticklabels=True, xticklabels=True, cmap="cividis", rasterized=True, cbar=False)
cb = fig2_0.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.SymLogNorm(vmin=1, vmax=np.nanmax(df.to_numpy()), base=10, linthresh=0.1, linscale=1), cmap="cividis"), ax=ax[0], shrink=0.75, label='Connecting cells')
cb.outline.set_linewidth(0)
ax[0].set_xlabel('Target');
ax[0].set_ylabel('Source');
ax[0].set_aspect('equal')
ax[0].tick_params(axis='both', which='major', labelsize=8)
sns.heatmap(FC.CorrelationMatrix, ax=ax[1], yticklabels=True, xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .75}, cmap="cividis", rasterized=True)
ax[1].set_aspect('equal')
ax[1].tick_params(axis='both', which='major', labelsize=8)

# Make adjacency matrices
# Log transform anatomical connectivity
anatomical_adjacency, keep_inds = AC.getAdjacency('CellCount', do_log=True)
functional_adjacency = FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]

r, p = pearsonr(anatomical_adjacency, functional_adjacency)
coef = np.polyfit(anatomical_adjacency, functional_adjacency, 1)
linfit = np.poly1d(coef)

fig2_1, ax = plt.subplots(1,1,figsize=(4, 4))
ax.plot(10**anatomical_adjacency, functional_adjacency, color='k', marker='o', linestyle='none', alpha=0.25)
xx = np.linspace(anatomical_adjacency.min(), anatomical_adjacency.max(), 100)
ax.plot(10**xx, linfit(xx), color='k', linewidth=2, marker=None)
ax.set_xscale('log')
ax.set_xlabel('Anatomical adjacency (cells)')
ax.set_ylabel('Functional correlation (z)')
ax.annotate('r = {:.2f}'.format(r), xy=(1, 1.0));

r_vals = []
for c_ind in range(FC.cmats.shape[2]):
    cmat = FC.cmats[:, :, c_ind]
    functional_adjacency_new = cmat[FC.upper_inds][keep_inds]

    r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_new)
    r_vals.append(r_new)

fig2_2, ax = plt.subplots(1, 1, figsize=(1.75, 3.5))
fig2_2.tight_layout(pad=4)
sns.stripplot(x=np.ones_like(r_vals), y=r_vals, color='k')
sns.violinplot(y=r_vals)
ax.set_ylabel('Structure-function corr. (z)')
ax.set_xticks([])
ax.set_ylim([0, 1])


fig2_0.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig2_0.svg'), format='svg', transparent=True)
fig2_1.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig2_1.svg'), format='svg', transparent=True)
fig2_2.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig2_2.svg'), format='svg', transparent=True)

# %%
anatomical_adjacency, keep_inds = AC.getAdjacency('CellCount', do_log=True)
functional_adjacency = FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]
r_shuffle = []
for it in range(1000):
    fa = np.random.permutation(functional_adjacency.copy())

    r_new, _ = pearsonr(anatomical_adjacency, fa)
    r_shuffle.append(r_new)

plt.hist(r_shuffle, 20);
# %% single linear regression with different connectivity metrics
fig2_5, ax = plt.subplots(1, 3, figsize=(9, 3.5))

rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=0)

# 1: Cell count
x, keep_inds = AC.getAdjacency('CellCount', do_log=True)
y = FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]
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
ax[0].set_title('Direct: cell count', fontsize=10)
ax[0].set_aspect('equal')

# 2: Synapse count
x, keep_inds = AC.getAdjacency('WeightedSynapseCount', do_log=True)
y = FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]
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
ax[1].set_title('Direct: synapse count', fontsize=10)
ax[1].set_aspect('equal')

# 3: Shortest path length
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
ax[2].plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
ax[2].plot(pred, y, 'ko', alpha=0.25)
ax[2].annotate('$r^2$={:.2f}'.format(avg_r2), (-0.15, 0.95))
ax[2].set_xlim([-0.2, 1.0])
ax[2].set_title('Shortest path distance', fontsize=10)
ax[2].set_aspect('equal')

fig2_5.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig2_5.pdf'), format='pdf', transparent=True)

# %% Basic SC-FC with synapse count
figS2_0, ax = plt.subplots(1, 2, figsize=(10, 5))
df = AC.getConnectivityMatrix('WeightedSynapseCount', diag=np.nan)
sns.heatmap(np.log10(AC.getConnectivityMatrix('WeightedSynapseCount', diag=np.nan)).replace([np.inf, -np.inf], 0), ax=ax[0], yticklabels=True, xticklabels=True, cmap="cividis", rasterized=True, cbar=False)
cb = figS2_0.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.SymLogNorm(vmin=1, vmax=np.nanmax(df.to_numpy()), base=10, linthresh=0.1, linscale=1), cmap="cividis"), ax=ax[0], shrink=0.75, label='Weighted synapse count')
cb.outline.set_linewidth(0)
ax[0].set_xlabel('Target');
ax[0].set_ylabel('Source');
ax[0].set_aspect('equal')
ax[0].tick_params(axis='both', which='major', labelsize=8)
sns.heatmap(FC.CorrelationMatrix, ax=ax[1], yticklabels=True, xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .75}, cmap="cividis", rasterized=True)
ax[1].set_aspect('equal')
ax[1].tick_params(axis='both', which='major', labelsize=8)

# Make adjacency matrices
# Log transform anatomical connectivity
anatomical_adjacency, keep_inds = AC.getAdjacency('WeightedSynapseCount', do_log=True)
functional_adjacency = FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]

r, p = pearsonr(anatomical_adjacency, functional_adjacency)
coef = np.polyfit(anatomical_adjacency, functional_adjacency, 1)
linfit = np.poly1d(coef)

figS2_1, ax = plt.subplots(1,1,figsize=(4, 4))
ax.plot(10**anatomical_adjacency, functional_adjacency, color='k', marker='o', linestyle='none', alpha=0.25)
xx = np.linspace(anatomical_adjacency.min(), anatomical_adjacency.max(), 100)
ax.plot(10**xx, linfit(xx), color='k', linewidth=2, marker=None)
ax.set_xscale('log')
ax.set_xlabel('Anatomical adjacency (synapses)')
ax.set_ylabel('Functional correlation (z)')
ax.annotate('r = {:.2f}'.format(r), xy=(1, 1.0));

r_vals = []
for c_ind in range(FC.cmats.shape[2]):
    cmat = FC.cmats[:, :, c_ind]
    functional_adjacency_new = cmat[FC.upper_inds][keep_inds]

    r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_new)
    r_vals.append(r_new)

figS2_2, ax = plt.subplots(1,1,figsize=(1.75, 3.5))
figS2_2.tight_layout(pad=4)
sns.stripplot(x=np.ones_like(r_vals), y=r_vals, color='k')
sns.violinplot(y=r_vals)
ax.set_ylabel('Structure-function corr. (z)')
ax.set_xticks([])
ax.set_ylim([0, 1]);

figS2_0.savefig(os.path.join(analysis_dir, 'figpanels', 'FigS2_0.svg'), format='svg', transparent=True)
figS2_1.savefig(os.path.join(analysis_dir, 'figpanels', 'FigS2_1.svg'), format='svg', transparent=True)
figS2_2.savefig(os.path.join(analysis_dir, 'figpanels', 'FigS2_2.svg'), format='svg', transparent=True)

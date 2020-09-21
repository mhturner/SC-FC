import matplotlib.pyplot as plt
from neuprint import Client, fetch_neurons, NeuronCriteria
import numpy as np
import networkx as nx
from scipy.stats import pearsonr, spearmanr
import os
import socket
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_validate
from scipy.stats import norm

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

import pandas as pd
from scfc.rate_model import RateModel

AC = anatomical_connectivity.AnatomicalConnectivity(data_dir=data_dir, neuprint_client=None, mapping=bridge.getRoiMapping())
tmp = AC.getConnectivityMatrix('CellCount').to_numpy().copy()
np.fill_diagonal(tmp, 0)

keep_inds = np.where(tmp > 0)
C = np.zeros_like(tmp)
base = 10
C[keep_inds] = np.log(tmp[keep_inds]) / np.log(base)

RM = RateModel(C=C, tau_i=2, tau_e=10, w_e=2, w_i=4, w_internode=0.50)
t, r_e, r_i = RM.solve(tdim=3000, r0=None, pulse_size=5, spike_rate=5, stimulus=None)


cmat = np.arctanh(np.corrcoef(r_e[100:, :].T))
np.fill_diagonal(cmat, np.nan)
pred_cmat = pd.DataFrame(data=cmat, index=FC.rois, columns=FC.rois)

fh, ax = plt.subplots(1, 1, figsize=(4, 2))
ax.plot(t, r_e)

y = FC.CorrelationMatrix.to_numpy()[FC.upper_inds]
y_hat = pred_cmat.to_numpy()[FC.upper_inds]

r2 = 1 - np.var(y-y_hat)/np.var(y)

r, _ = pearsonr(y, y_hat)
fh, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(pred_cmat.to_numpy()[FC.upper_inds], FC.CorrelationMatrix.to_numpy()[FC.upper_inds], 'ko')
ax.plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Measured')
ax.set_title('r2 = {:.3f}'.format(r2));

# %%
# # plot responses
events = []
for e in range(nez_e.shape[0]):
    events.append(np.where(nez_e[e, :] > 0)[0])

fig1, ax = plt.subplots(3, 1, figsize=(8,4))
ax[0].eventplot(events, color='k')
ax[0].set_xlim([1000, 2000])
ax[0].set_ylabel('Noise')
ax[0].set_xticks([])

ax[1].plot(t, r_e, linewidth=2)
ax[1].set_xlim([1000, 2000])
ax[1].set_ylabel('r exc')
ax[1].set_xticks([])

ax[2].plot(t, r_i, linewidth=2)

ax[2].set_xlim([1000, 2000])
ax[2].set_ylabel('r inh')

cmat = np.arctanh(np.corrcoef(r_e[100:, :].T))
np.fill_diagonal(cmat, np.nan)
pred_cmat = pd.DataFrame(data=cmat, index=FC.rois, columns=FC.rois)

vmin = np.nanmin(FC.CorrelationMatrix.to_numpy())
vmax = np.nanmax(FC.CorrelationMatrix.to_numpy())

fig2, ax = plt.subplots(1, 2, figsize=(14, 7))
sns.heatmap(pred_cmat, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .75}, cmap="cividis", rasterized=True)
ax[0].set_aspect('equal')
ax[0].set_title('Predicted')
sns.heatmap(FC.CorrelationMatrix, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .75}, cmap="cividis", rasterized=True, vmin=vmin, vmax=vmax)
ax[1].set_aspect('equal')
ax[1].set_title('Measured')

r2 = explained_variance_score(pred_cmat.to_numpy()[FC.upper_inds], FC.CorrelationMatrix.to_numpy()[FC.upper_inds])
r, _ = pearsonr(pred_cmat.to_numpy()[FC.upper_inds], FC.CorrelationMatrix.to_numpy()[FC.upper_inds])
fig3, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(pred_cmat.to_numpy()[FC.upper_inds], FC.CorrelationMatrix.to_numpy()[FC.upper_inds], 'ko')
ax.plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Measured')
ax.set_title('r = {:.3f}'.format(r));


# %% pulse and measure predictions
target_region = 'MBML(R)'
tdim = 500
stim_amplitude = 2
stim_node_index = np.where(np.array(FC.rois)==target_region)[0][0]
stimulus = np.zeros_like(nez_e)
nez_e = np.zeros_like(nez_e) # cut the noise

stimulus[stim_node_index, 200:250] = stim_amplitude

# solve
t = np.arange(0, tdim) # sec
X = odeint(dXdt, r0, t, args=(nez_e, stimulus))
r_e = X[:, :n_nodes]
r_i = X[:, n_nodes:2*n_nodes]

node_responses = np.max(r_e[200:250, :], axis=0)
sort_inds = np.argsort(node_responses)[::-1]

# plot responses
fig4, ax = plt.subplots(3, 1, figsize=(8,5))

ax[0].plot(t, r_e, linewidth=2)
ax[0].set_ylabel('r exc')
ax[0].annotate('Inject into {}'.format(target_region), (400, 1))
ax[0].set_xlim([50, 500])
ax[0].set_xticks([])

ax[1].plot(t, r_i, linewidth=2)
ax[1].set_ylabel('r inh')
ax[1].set_xlim([50, 500])

ax[2].plot(node_responses[sort_inds][1:], 'kx')
ax[2].set_xticks(list(range(n_nodes-1)))
ax[2].set_xticklabels(np.array(FC.rois)[sort_inds][1:], rotation=90);
ax[2].set_ylim([0, 1.25*node_responses[sort_inds][1]])
ax[2].set_ylabel('Peak response (a.u.)')

# %% save figs

figs_to_save = [fig1, fig2, fig3, fig4]
for f_ind, fh in enumerate(figs_to_save):
    fh.savefig(os.path.join(analysis_dir, 'figpanels', 'RateModelFig{}.svg'.format(f_ind)))

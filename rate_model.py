import numpy as np
from scipy.integrate import odeint
from scipy.special import erf
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import glob
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score

from scfc import functional_connectivity, bridge, anatomical_connectivity
from matplotlib import rcParams
rcParams['svg.fonttype'] = 'none'

"""
See coupled ei model here:
https://elifesciences.org/articles/22425#s4

"""

data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC'

tdim = 4000

tau_i = 2 # msec (2)
tau_e = 10 # msec (10)
# w_: within subnetwork weights
w_ee = 2 # e self drive (2)
w_ei = 4 # i to e (4)
w_ie = 4 # e to i (4)

 # scaling of weights between excitatory populations
 # CellCount: ~2
  # CellCount, log: ~0.5
 # WeightedSynapseCount: ~7
w_internode = 0.51
do_log = True

pulse_size = 5 # (5)
spike_rate = 5 #hz (5)

# Get FunctionalConnectivity object
FC = functional_connectivity.FunctionalConnectivity(data_dir=data_dir, fs=1.2, cutoff=0.01, mapping=bridge.getRoiMapping())

# Get AnatomicalConnectivity object
AC = anatomical_connectivity.AnatomicalConnectivity(data_dir=data_dir, neuprint_client=None, mapping=bridge.getRoiMapping())
tmp = AC.getConnectivityMatrix('CellCount').to_numpy().copy()
np.fill_diagonal(tmp, 0)
if do_log:
    keep_inds = np.where(tmp > 0)
    C = np.zeros_like(tmp)
    base = 10
    C[keep_inds] = np.log(tmp[keep_inds]) / np.log(base)
else:
    C = tmp

C = C / C.max()


# C = np.random.rand(C.shape[0], C.shape[1])
# np.fill_diagonal(C, 0)

n_nodes = C.shape[0]

# poisson noise in all nodes
cutoff_prob = spike_rate / 1000 # spikes per msec bin
spikes = np.random.uniform(low=0, high=1, size=(n_nodes, tdim)) <= cutoff_prob
nez_e = pulse_size * spikes

stimulus = None

C_internode = w_internode * C

# IC stats from end of run
# r_i[-1, :].std()
r0 = np.hstack([0.3+0.3*np.random.rand(n_nodes), # exc initial conditions
                3+3*np.random.rand(n_nodes)]) # inh initial conditions

def threshlinear(input):
    output = np.array(input)
    output[output < 0] = 0
    return output

def sigmoid(input, thresh=0, scale=5):
    output = scale*erf((np.array(input)-thresh)/scale)
    output[output < 0] = 0
    return output

def dXdt(X, t, nez_e, stimulus):
    """
    Input is X := r_exc, r_inh... at time t
    Output is dX/dt
    """
    r_e = X[:n_nodes]
    r_i = X[n_nodes:2*n_nodes]
    if stimulus is not None:
        stim = stimulus[:, int(t)]
    else:
        stim = 0

    internode_inputs = C_internode.T @ r_e
    exc_inputs = w_ee*r_e - w_ei*r_i
    edot = (-r_e + sigmoid(internode_inputs + exc_inputs) + nez_e[:, int(t)] + stim) / tau_e

    inh_inputs = w_ie*r_e
    idot = (-r_i + sigmoid(inh_inputs) + stim) / tau_i

    return np.hstack([edot, idot])

# solve
t = np.arange(0, tdim) # sec
X = odeint(dXdt, r0, t, args=(nez_e, stimulus))
r_e = X[:, :n_nodes]
r_i = X[:, n_nodes:2*n_nodes]

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

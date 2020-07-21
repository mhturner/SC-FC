import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import glob
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score
from region_connectivity import RegionConnectivity
from matplotlib.backends.backend_pdf import PdfPages
import datetime

"""
See coupled ei model here:
https://elifesciences.org/articles/22425#s4

"""

data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC'

do_log_transform = False # (True, False)
tdim = 10000

tau_i = 2 # msec (2, 2)
tau_e = 10 # msec (10, 10)
# w_: within subnetwork weights
w_ee = 2 # e self drive (2, 2)
w_ei = 2 # i to e (2, 2)
w_ie = 2 # e to i (2, 2)

 # scaling of weights between excitatory populations (1.10, 4)
 #  Log weights: 0.13 = close to measured
 #  Raw weigihts: 0.52
w_internode = 0.52

pulse_size = 25
spike_rate = 2 #hz

# # # load measured fxnal connectivity
roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')
response_filepaths = glob.glob(os.path.join(data_dir, 'region_responses') + '/' + '*.pkl')
meas_cmat, _ = RegionConnectivity.getFunctionalConnectivity(response_filepaths, cutoff=0.01, fs=1.2)
upper_inds = np.triu_indices(meas_cmat.shape[0], k=1) # k=1 excludes main diagonal

# # # anatomical connectivity matrix
WeakConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'WeakConnections_computed_20200626.pkl'))
MediumConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'MediumConnections_computed_20200626.pkl'))
StrongConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'StrongConnections_computed_20200626.pkl'))
conn_mat = WeakConnections + MediumConnections + StrongConnections
C = conn_mat.to_numpy()
if do_log_transform:
    C = np.log10(C)
    C[np.isinf(C)] = 0
np.fill_diagonal(C, 0)
C = C / C.max()
n_nodes = C.shape[0]

# poisson noise in all nodes
cutoff_prob = spike_rate / 1000 # spikes per msec bin
spikes = np.random.uniform(low=0, high=1, size=(n_nodes, tdim)) <= cutoff_prob
nez_e = pulse_size * spikes

stimulus = None

C_internode = w_internode * C

# IC stats from end of run
# r_i[-1, :].mean()
r0 = np.hstack([1+1*np.random.rand(n_nodes), # exc initial conditions
                8+8*np.random.rand(n_nodes)]) # inh initial conditions


def threshlinear(input):
    output = np.array(input)
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
    exc_inputs = w_ee*r_e - w_ei*r_i + nez_e[:, int(t)] + stim
    edot = (-r_e + threshlinear(internode_inputs + exc_inputs)) / tau_e

    inh_inputs = w_ie*r_e + stim
    idot = (-r_i + threshlinear(inh_inputs)) / tau_i

    return np.hstack([edot, idot])

# solve
t = np.arange(0, tdim) # sec
X = odeint(dXdt, r0, t, args=(nez_e, stimulus))
r_e = X[:, :n_nodes]
r_i = X[:, n_nodes:2*n_nodes]

# # plot responses
fig1, ax = plt.subplots(3, 1, figsize=(12,6))
ax[0].imshow(nez_e, rasterized=True)
ax[0].set_xlim([0, tdim])
ax[0].set_aspect(20)

ax[1].plot(t, r_e, linewidth=2)
ax[1].set_ylabel('r exc')

ax[2].plot(t, r_i, linewidth=2)
ax[2].set_ylabel('r inh')

cmat = np.arctanh(np.corrcoef(r_e[100:, :].T))
np.fill_diagonal(cmat, np.nan)
pred_cmat = pd.DataFrame(data=cmat, index=conn_mat.index, columns=conn_mat.columns)

fig2, ax = plt.subplots(1, 2, figsize=(14,6))
sns.heatmap(pred_cmat, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True)
sns.heatmap(meas_cmat, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True)

r2 = explained_variance_score(pred_cmat.to_numpy()[upper_inds], meas_cmat.to_numpy()[upper_inds])
r, _ = pearsonr(pred_cmat.to_numpy()[upper_inds], meas_cmat.to_numpy()[upper_inds])
fig3, ax = plt.subplots(1, 1, figsize=(6,6))
ax.plot(pred_cmat.to_numpy()[upper_inds], meas_cmat.to_numpy()[upper_inds], 'ko')
ax.plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Measured')
ax.set_title('r = {:.3f}'.format(r));


# %% pulse and measure predictions
target_region = 'MBML(R)'
tdim = 500
stim_amplitude = 500
stim_node_index = np.where(conn_mat.index==target_region)[0][0]
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
fig4, ax = plt.subplots(3, 1, figsize=(12,6))

ax[0].plot(t, r_e, linewidth=2)
ax[0].set_ylabel('r exc')

ax[1].plot(t, r_i, linewidth=2)
ax[1].set_ylabel('r inh')

ax[2].plot(node_responses[sort_inds][1:], 'kx')
ax[2].set_xticks(list(range(n_nodes-1)))
ax[2].set_xticklabels(conn_mat.index[sort_inds][1:], rotation=90);
ax[2].set_ylim([0, 1.25*node_responses[sort_inds][1]])
ax[2].set_ylabel('Peak response (a.u.)')

# %% save figs
with PdfPages(os.path.join(analysis_dir, 'rate_model_figs.pdf')) as pdf:
    pdf.savefig(fig1)
    pdf.savefig(fig2)
    pdf.savefig(fig3)
    pdf.savefig(fig4)


    d = pdf.infodict()
    d['Author'] = 'Max Turner'
    d['ModDate'] = datetime.datetime.today()

plt.close('all')

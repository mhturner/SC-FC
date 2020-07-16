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
"""
See coupled ei model here:
https://elifesciences.org/articles/22425#s4

"""

data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'

tdim = 500

tau_i = 2 # msec
tau_e = 10 # msec
beta = 1.0 # adaptation current in exc populations
w_ee = 5 # w_ are within subnetwork weights
w_ei = 1
w_ie = 10
w_ii = 0.4

w_internode = 0.4 # weights between excitatory populations

nez_mean = 0.0
nez_scale = 5

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
np.fill_diagonal(C, 0)
C = C / C.max()
n_nodes = C.shape[0]


nez_e = np.random.normal(loc=nez_mean, scale=nez_scale, size=(n_nodes, tdim))
nez_i = np.random.normal(loc=nez_mean, scale=nez_scale, size=(n_nodes, tdim))

C_internode = w_internode * C

r0 = np.hstack([nez_scale*np.random.rand(n_nodes),
      nez_scale*np.random.rand(n_nodes),
      0 * np.ones(n_nodes)])

def threshlinear(input):
    output = np.array(input)
    output[output < 0] = 0
    return output

def dXdt(X, t, tau_e, tau_i):
    """
    Input is X := r_exc, r_inh, and a... at time t
    Output is dX/dt
    """
    r_e = X[:n_nodes]
    r_i = X[n_nodes:2*n_nodes]
    a = X[2*n_nodes:]

    internode_inputs = C_internode.T @ r_e
    exc_inputs = w_ee*r_e - w_ei*r_i + a + nez_e[:, int(t)]
    edot = (-r_e + threshlinear(internode_inputs + exc_inputs)) / tau_e

    inh_inputs = w_ie*r_e - w_ii*r_i + nez_i[:, int(t)]
    idot = (-r_i + threshlinear(inh_inputs)) / tau_i

    adot = (-a + beta * r_e)

    return np.hstack([edot, idot, adot])

t = np.arange(0, tdim) # sec

# solve ODEs
X = odeint(dXdt, r0, t, args=(tau_e, tau_i,))
r_e = X[:, :n_nodes]
r_i = X[:, n_nodes:2*n_nodes]
a = X[:, 2*n_nodes:]


# %% plot responses
fh, ax = plt.subplots(3, 1, figsize=(12,6))
ax[0].plot(t, r_e, linewidth=2)
ax[0].set_ylabel('r exc')

ax[1].plot(t, nez_e[0,:], linewidth=2)
ax[1].set_ylabel('r inh')

ax[2].plot(t, a, linewidth=2)
ax[2].set_ylabel('a')

cmat = np.arctanh(np.corrcoef(r_e.T))
np.fill_diagonal(cmat, np.nan)
pred_cmat = pd.DataFrame(data=cmat, index=conn_mat.index, columns=conn_mat.columns)



fh, ax = plt.subplots(1, 2, figsize=(14,6))
sns.heatmap(pred_cmat, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True)
sns.heatmap(meas_cmat, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True)

r2 = explained_variance_score(pred_cmat.to_numpy()[upper_inds], meas_cmat.to_numpy()[upper_inds])
r, _ = pearsonr(pred_cmat.to_numpy()[upper_inds], meas_cmat.to_numpy()[upper_inds])
fh, ax = plt.subplots(1, 1, figsize=(6,6))
ax.plot(pred_cmat.to_numpy()[upper_inds], meas_cmat.to_numpy()[upper_inds], 'ko')
ax.plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Measured')
ax.set_title('r = {:.3f}'.format(r))

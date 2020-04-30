import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score
import pickle

"""

Galan 2008 (PLoS One)
"""

analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/hemibrain_analysis/roi_connectivity'

CorrelationMatrix_Functional = pd.read_pickle(os.path.join(analysis_dir, 'data', 'CorrelationMatrix_Functional.pkl'))
# with open(os.path.join(analysis_dir, 'data', 'NeuronCount_computed_20200422.pkl'), 'rb') as f:
#     Conn = pickle.load(f)[0].to_numpy()

Conn = pd.read_pickle(os.path.join(analysis_dir,'data', 'ConnectivityMatrix_computed_20200422.pkl')).to_numpy()

C = Conn / Conn.max()
self_couplings = -C.diagonal()
np.fill_diagonal(C, 0)
C += np.diag(self_couplings)


dt = 0.1
timevector = np.arange(0, 5000, dt)

def getV(C, timevector, dt, alpha, scale=1):
    nodes = C.shape[0]
    V = np.zeros(shape=(nodes, len(timevector)))
    V[:, 0] = np.random.normal(loc=0.0, scale=scale, size=nodes)

    for t in range(1, len(timevector)):
        # V[:, t] = (1 - (alpha * dt)) * V[:, t-1] + (C @ V[:, t-1]) * dt + np.random.normal(loc=0.0, scale=scale, size=nodes) * dt
        V[:, t] = dt * (-alpha * V[:, t-1] + (C @ V[:, t-1]) + np.random.normal(loc=0.0, scale=scale, size=nodes))
    return V


# def getV(C, timevector, dt, alpha, tau, scale=1):
#     nodes = C.shape[0]
#     V = np.zeros(shape=(nodes, len(timevector)))
#     V[:, 0] = np.random.normal(loc=0.0, scale=scale, size=nodes)
#
#     for t in range(1, len(timevector)):
#         V[:, t] = dt * (-alpha + (C @ V[:, t-1]) + np.random.normal(loc=0.0, scale=scale, size=nodes)) / tau
#         # (1 - (alpha * dt)) * V[:, t-1] + (C @ V[:, t-1]) * dt + np.random.normal(loc=0.0, scale=scale, size=nodes) * dt
#     return V

# %%
# alphas = np.arange(1, 4, 0.1) # count connectivity
alphas = np.arange(0, 0.5, 0.01) # weight connectivity
corrs = np.zeros(len(alphas))

for a_ind, alpha in enumerate(alphas):
    V = getV(C, timevector, dt, alpha)

    pred_corr = np.corrcoef(V)
    np.fill_diagonal(pred_corr, 0)
    # apply fischer transform
    pred_corr = np.arctanh(pred_corr)

    # compare measured functional and predicted functional correlation matrices
    upper_inds = np.triu_indices(CorrelationMatrix_Functional.shape[0], k=1) #k=1 excludes main diagonal
    functional_adjacency = CorrelationMatrix_Functional.to_numpy()[upper_inds]
    pred_functional_adjacency = pred_corr[upper_inds]

    if np.any(np.isinf(pred_functional_adjacency)):
        continue
    elif np.any(np.isnan(pred_functional_adjacency)):
        continue
    else:
        # r, p = pearsonr(pred_functional_adjacency, functional_adjacency)
        r2 = explained_variance_score(functional_adjacency, pred_functional_adjacency)

        corrs[a_ind] = r2



# %%
corrs[np.where(corrs<0)] = 0
fh, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(alphas, corrs, 'k-o')
ax.set_xlabel('Alpha')
ax.set_ylabel('Frac. var explained');



# %%
# with open(os.path.join(analysis_dir, 'data', 'NeuronCount_computed_20200422.pkl'), 'rb') as f:
#     Conn = pickle.load(f)[0].to_numpy()

Conn = pd.read_pickle(os.path.join(analysis_dir,'data', 'ConnectivityMatrix_computed_20200422.pkl')).to_numpy()

C = Conn / Conn.max()
# self_couplings = -C.diagonal()
np.fill_diagonal(C, 0)
# C += np.diag(self_couplings)
C = C * 35

dt = 0.1
timevector = np.arange(0, 10000, dt)
alpha = 2



V = getV(C, timevector, dt, alpha, scale=1)

# plt.plot(timevector, V.T);

pred_corr = np.corrcoef(V)
np.fill_diagonal(pred_corr, 0)
# apply fischer transform
pred_corr = np.arctanh(pred_corr)

pred_corr_df = pd.DataFrame(data=pred_corr, index=CorrelationMatrix_Functional.index, columns=CorrelationMatrix_Functional.columns)

# compare measured functional and predicted functional correlation matrices
upper_inds = np.triu_indices(CorrelationMatrix_Functional.shape[0], k=1) #k=1 excludes main diagonal
functional_adjacency = CorrelationMatrix_Functional.to_numpy()[upper_inds]
pred_functional_adjacency = pred_corr_df.to_numpy()[upper_inds]


r, p = pearsonr(pred_functional_adjacency, functional_adjacency)
r2 = explained_variance_score(functional_adjacency, pred_functional_adjacency)


fh, ax = plt.subplots(1, 3, figsize=(18, 6))
sns.heatmap(pred_corr_df, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Corr (z)'}, cmap="cividis", rasterized=True)
ax[0].set_aspect('equal')
ax[0].set_title('R = {}'.format(r2))

sns.heatmap(CorrelationMatrix_Functional, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Corr (z)'}, cmap="cividis", rasterized=True)
ax[1].set_aspect('equal')


ax[2].plot(pred_functional_adjacency, functional_adjacency, 'ko')
ax[2].plot([-0.2, 1.4], [-0.2, 1.4], 'k--')
ax[2].set_xlabel('Predicted corr (z)')
ax[2].set_ylabel('Measured corr (z)');


 #%%
# %% try analytical thing from Galan 2008
CorrelationMatrix_Functional = pd.read_pickle(os.path.join(analysis_dir, 'data', 'CorrelationMatrix_Functional.pkl'))
with open(os.path.join(analysis_dir, 'data', 'NeuronCount_computed_20200422.pkl'), 'rb') as f:
    Conn = pickle.load(f)[0].to_numpy()

C = Conn / Conn.max()
np.fill_diagonal(C, 0) #replace diagonal nan with 0

sigma = 1
alpha = 0.9
dt = 0.01

A = -alpha * np.eye(C.shape[0]) + C

val, L = np.linalg.eig(A)
Q = sigma * np.eye(C.shape[0])
Q_hat = np.linalg.inv(L) @ Q @ np.linalg.inv(L).conj().T
P = Q_hat / (1 - np.outer(val, val.conj()))

Cov = L @ P @ L.conj().T

Cov
D = np.outer(np.sqrt(np.diagonal(Cov)), np.sqrt(np.diagonal(Cov)))
Corr = Cov / D

Corr[Cov==0] = 0


np.fill_diagonal(Corr, 0)
# apply fischer transform
Corr = np.arctanh(Corr)
pred_corr_df = pd.DataFrame(data=Corr, index=CorrelationMatrix_Functional.index, columns=CorrelationMatrix_Functional.columns)


upper_inds = np.triu_indices(CorrelationMatrix_Functional.shape[0], k=1) #k=1 excludes main diagonal
functional_adjacency = CorrelationMatrix_Functional.to_numpy()[upper_inds]
pred_functional_adjacency = pred_corr_df.to_numpy()[upper_inds]
pred_functional_adjacency
keep_inds = np.where(~np.isnan(pred_functional_adjacency))
r, p = pearsonr(pred_functional_adjacency[keep_inds], functional_adjacency[keep_inds])
# r2 = explained_variance_score(functional_adjacency, pred_functional_adjacency)

fh, ax = plt.subplots(1, 2, figsize=(18, 9))
sns.heatmap(pred_corr_df, ax=ax[0], xticklabels=True, cbar_kws={'label': 'computed'}, cmap="viridis", rasterized=True)
ax[0].set_aspect('equal')
ax[0].set_title('R = {}'.format(r))

sns.heatmap(CorrelationMatrix_Functional, ax=ax[1], xticklabels=True, cbar_kws={'label': 'computed'}, cmap="viridis", rasterized=True)
ax[1].set_aspect('equal')

fh, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(functional_adjacency[keep_inds], pred_functional_adjacency[keep_inds], 'ko')


# %%
from scipy.signal import butter, lfilter

import nest
nest.ResetKernel()

dt = 1.0 # msec
dt_rec = 50.0 # msec
time_stop = 10000  #msec
gi = 0
ge = 750

nest.SetKernelStatus({'resolution': dt})

# Conn = pd.read_pickle(os.path.join(analysis_dir,'data', 'ConnectivityMatrix_computed_20200422.pkl')).to_numpy()
with open(os.path.join(analysis_dir, 'data', 'NeuronCount_computed_20200422.pkl'), 'rb') as f:
    Conn = pickle.load(f)[0].to_numpy()

C = Conn / Conn.max()

nodes = C.shape[0]

node_pops = nest.Create("gif_pop_psc_exp", nodes, {"tau_m": 20.0, "len_kernel": -1})

voltmeter = nest.Create('voltmeter')
nez = nest.Create('noise_generator', nodes)

for n_ind, nod in enumerate(node_pops):
    nest.SetStatus([nez[n_ind]], {'mean': 0.0, 'std': 1.0, 'dt': 100.0})
    nest.Connect([nez[n_ind]], [nod])


for i, nest_i in enumerate(node_pops):
    for j, nest_j in enumerate(node_pops):
        if i == j:
            nest.Connect([nest_i], [nest_j], conn_spec='all_to_all', syn_spec={'model': 'static_synapse', 'weight': -gi*C[i, j], 'delay': 6.0})
        else:
            nest.Connect([nest_i], [nest_j], conn_spec='all_to_all', syn_spec={'model': 'static_synapse', 'weight': ge*C[i, j], 'delay': 3.0})

nest.Connect(voltmeter, node_pops)

nest.Simulate(time_stop)

ev = nest.GetStatus(voltmeter)[0]['events']

def lpfilter(signal, cut):
    b, a = butter(3, cut, btype='low')
    y = lfilter(b, a, signal)
    return y

cut_freq = 20 # hz
fs = (1 / cut_freq) * 1e3 #msec
cut_frac = dt / fs # number of dts in the cutoff sample period

tt = np.arange(0, time_stop, dt)[:-1] / 1e3 #sec
U = np.zeros((nodes, len(tt)))
for n_ind, nod in enumerate(node_pops):
    new_resp = ev['V_m'][ev['senders'] == nod]
    U[n_ind, :] = lpfilter(new_resp, cut=cut_frac)


fh, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(tt, U.T);
ax.set_xlabel('Time (s)')
ax.set_ylabel('Mean node response (a.u.)')
# ax.set_ylim([0, 10])

# %%
pred_corr = np.corrcoef(U)
np.fill_diagonal(pred_corr, 0)
# apply fischer transform
pred_corr = np.arctanh(pred_corr)

pred_corr_df = pd.DataFrame(data=pred_corr, index=CorrelationMatrix_Functional.index, columns=CorrelationMatrix_Functional.columns)

# compare measured functional and predicted functional correlation matrices
upper_inds = np.triu_indices(CorrelationMatrix_Functional.shape[0], k=1) #k=1 excludes main diagonal
functional_adjacency = CorrelationMatrix_Functional.to_numpy()[upper_inds]
pred_functional_adjacency = pred_corr_df.to_numpy()[upper_inds]


r, p = pearsonr(pred_functional_adjacency, functional_adjacency)
r2 = explained_variance_score(functional_adjacency, pred_functional_adjacency)


fh, ax = plt.subplots(1, 3, figsize=(27, 9))
sns.heatmap(pred_corr_df, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Corr (z)'}, cmap="cividis", rasterized=True)
ax[0].set_aspect('equal')
ax[0].set_title('R = {:.2f}'.format(r))

sns.heatmap(CorrelationMatrix_Functional, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Corr (z)'}, cmap="cividis", rasterized=True)
ax[1].set_aspect('equal')


ax[2].plot(pred_functional_adjacency, functional_adjacency, 'ko')
ax[2].plot([-0.2, 1.4], [-0.2, 1.4], 'k--')
ax[2].set_xlabel('Predicted corr (z)')
ax[2].set_ylabel('Measured corr (z)');

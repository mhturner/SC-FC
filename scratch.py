import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import os

data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'

tdim = 100
tau_s = 4

noise_mean = 0.0
noise_scale = 0.1

# # anatomical connectivity matrix
WeakConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'WeakConnections_computed_20200626.pkl'))
MediumConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'MediumConnections_computed_20200626.pkl'))
StrongConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'StrongConnections_computed_20200626.pkl'))
conn_mat = WeakConnections + MediumConnections + StrongConnections
C = conn_mat.to_numpy()
C = C / C.max()

np.fill_diagonal(C, 0)
n_nodes = C.shape[0]

s0 = np.random.rand(n_nodes)

def f_r(r):
    """
    Compute synaptic output of each cell given its firing rate
    """
    s = r.copy()
    s[s<0] = 0
    return s

def f_s(s):
    """
    Compute firing rate of each cell given all of the synaptic inputs
    """
    # transpose here is b/c C is source x target.
    # this matmul sums all the synaptic inputs into each cell
    summed_inputs = C.T @ s
    summed_inputs[summed_inputs<0] = 0 # rectify
    firing_rate = summed_inputs + np.random.normal(loc=noise_mean, scale=noise_scale, size=len(summed_inputs))
    # firing_rate = summed_inputs

    firing_rate[firing_rate>100] = 100 # saturate

    return firing_rate

def dsdt(s, t, tau_s):
    r = f_s(s) # firing rate depends instantaneously on synaptic inputs
    # print(s)
    # print(f_r(r))
    sdot = (-s + f_r(r)) / tau_s

    return sdot

t = np.arange(0, tdim) # sec

# solve ODEs
S = odeint(dsdt, s0, t, args=(tau_s,))
R = [f_s(S[x,:]) for x in range(tdim)]
S[0,:]

fh, ax = plt.subplots(2, 1, figsize=(10,5))
ax[0].plot(t, R, linewidth=2)
ax[0].set_ylabel('R')

ax[1].plot(t, S, linewidth=2)
ax[1].set_xlabel('time')
ax[1].set_ylabel('S');


corrmat = np.corrcoef(R)

# %%
from neuprint import (Client, fetch_neurons, fetch_adjacencies, NeuronCriteria)
import numpy as np

neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1heHdlbGxob2x0ZXR1cm5lckBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpMHJRX0M4akliX0ZrS2h2OU5DSElsWlpnRDY5YUMtVGdNLWVWM3lRP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzY2MTk1MzcwfQ.Q-57D4tX2sXMjWym2LFhHaUGHgHiUsIM_JI9xekxw_0')

ra = 'AL(R)'
rb = 'LH(R)'

Neur_a, _ = fetch_neurons(NeuronCriteria(outputRois=ra, status='Traced'))
Neur_b, _ = fetch_neurons(NeuronCriteria(outputRois=rb, status='Traced'))

np.intersect1d(Neur_a.bodyId, Neur_b.bodyId).shape
len(Neur_a.bodyId)

Neur_ab, _ = fetch_neurons(NeuronCriteria(outputRois=[ra, rb], status='Traced', roi_req='all'))

Neur_a.shape

Neur_a, _ = fetch_neurons(NeuronCriteria(outputRois=ra, status='Traced'))
Neur_a.shape
Neur_a, _  = fetch_neurons(NeuronCriteria(outputRois=ra, status='Traced', min_roi_outputs=10))
Neur_a.shape

Neur_a.loc[0].roiInfo

Neur_ab, _ = fetch_neurons(NeuronCriteria(outputRois=[ra, rb], status='Traced', min_roi_outputs=10, roi_req='all'))
# %%
Neur_ab.shape
Neur_a, _ = fetch_neurons(NeuronCriteria(outputRois=[ra, rb], status='Traced'))
Neur_a.shape
Neur_a

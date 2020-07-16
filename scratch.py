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

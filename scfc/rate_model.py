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

"""
See coupled ei model here:
https://elifesciences.org/articles/22425#s4

"""

class RateModel:
    def __init__(self, C, tau_i=2, tau_e=10, w_e=2, w_i=4, w_internode=0.5):
        self.tau_i = tau_i
        self.tau_e = tau_e
        self.w_e = w_e
        self.w_i = w_i
        self.w_internode = w_internode

        # Connectivity matrix
        C = C / C.max()
        self.n_nodes = C.shape[0]
        self.C_internode = w_internode * C

    def solve(self, tdim=4000, r0=None, pulse_size=5, spike_rate=5, stimulus=None):
        # poisson noise in all nodes
        cutoff_prob = spike_rate / 1000 # spikes per msec bin
        spikes = np.random.uniform(low=0, high=1, size=(self.n_nodes, tdim)) <= cutoff_prob
        nez_e = pulse_size * spikes

        if r0 is None:
            r0 = np.hstack([0.3+0.3*np.random.rand(self.n_nodes), # exc initial conditions
                            3+3*np.random.rand(self.n_nodes)]) # inh initial conditions

        # solve
        t = np.arange(0, tdim) # sec
        X = odeint(self.dXdt, r0, t, args=(nez_e, stimulus))
        r_e = X[:, :self.n_nodes]
        r_i = X[:, self.n_nodes:2*self.n_nodes]

        self.nez_e = nez_e

        return t, r_e, r_i

    def dXdt(self, X, t, nez_e, stimulus):
        """
        Input is X := r_exc, r_inh... at time t
        Output is dX/dt
        """
        r_e = X[:self.n_nodes]
        r_i = X[self.n_nodes:2*self.n_nodes]
        if stimulus is not None:
            stim = stimulus[:, int(t)]
        else:
            stim = 0

        internode_inputs = self.C_internode.T @ r_e
        exc_inputs = self.w_e*r_e - self.w_i*r_i
        edot = (-r_e + sigmoid(internode_inputs + exc_inputs) + nez_e[:, int(t)] + stim) / self.tau_e

        inh_inputs = self.w_i*r_e
        idot = (-r_i + sigmoid(inh_inputs) + stim) / self.tau_i

        return np.hstack([edot, idot])




def threshlinear(input):
    output = np.array(input)
    output[output < 0] = 0
    return output

def sigmoid(input, thresh=0, scale=5):
    output = scale*erf((np.array(input)-thresh)/scale)
    output[output < 0] = 0
    return output

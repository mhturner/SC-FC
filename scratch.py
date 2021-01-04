"""
Turner, Mann, Clandinin: Figure generation script: Fig. 2.

https://github.com/mhturner/SC-FC
"""

import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import os
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns
import glob
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score, contingency_matrix
from munkres import Munkres


from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
import matplotlib
from matplotlib import rcParams
rcParams.update({'font.size': 12})
rcParams.update({'axes.spines.right': False})
rcParams.update({'axes.spines.top': False})
rcParams['svg.fonttype'] = 'none' # let illustrator handle the font type

data_dir = bridge.getUserConfiguration()['data_dir']
analysis_dir = bridge.getUserConfiguration()['analysis_dir']
token = bridge.getUserConfiguration()['token']

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token=token)

mp = bridge.getRoiMapping()
mp

# Get FunctionalConnectivity object
FC = functional_connectivity.FunctionalConnectivity(data_dir=data_dir, fs=1.2, cutoff=0.01, mapping=bridge.getRoiMapping())

# Get AnatomicalConnectivity object
AC = anatomical_connectivity.AnatomicalConnectivity(data_dir=data_dir, neuprint_client=neuprint_client, mapping=bridge.getRoiMapping())



# %%
# Make adjacency matrices
# Log transform anatomical connectivity
anatomical_adjacency, keep_inds = AC.getAdjacency('CellCount', do_log=True)
functional_adjacency = FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]

r, p = pearsonr(anatomical_adjacency, functional_adjacency)
coef = np.polyfit(anatomical_adjacency, functional_adjacency, 1)
linfit = np.poly1d(coef)

fig2_4, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.plot(10**anatomical_adjacency, functional_adjacency, color='k', marker='.', linestyle='none', alpha=1.0)
xx = np.linspace(anatomical_adjacency.min(), anatomical_adjacency.max(), 100)
ax.plot(10**xx, linfit(xx), color='k', linewidth=2, marker=None)
ax.set_xscale('log')
ax.set_xlabel('Cell Count')
ax.set_ylabel('Functional correlation (z)')
ax.annotate('r = {:.2f}'.format(r), xy=(0.8, 1.1))

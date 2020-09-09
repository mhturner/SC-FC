import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import networkx as nx
from scipy.stats import pearsonr, spearmanr
import os
import socket

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

ConnectivityMatrix = AC.getConnectivityMatrix(type='TBars', symmetrize=True, diag=np.nan)
CorrelationMatrix = FC.CorrelationMatrix.copy()
roi_completeness = anatomical_connectivity.getRoiCompleteness(neuprint_client, bridge.getRoiMapping())
roi_completeness
# completeness threshold
thresh = 0.0
exclude_rois = np.where(roi_completeness.loc[:, 'frac_pre'] < thresh)[0]
ConnectivityMatrix.iloc[exclude_rois, :] = np.nan
ConnectivityMatrix.iloc[:, exclude_rois] = np.nan

CorrelationMatrix.iloc[exclude_rois, :] = np.nan
CorrelationMatrix.iloc[:, exclude_rois] = np.nan

keep_inds = np.where(ConnectivityMatrix.to_numpy()[AC.upper_inds] > 0) # for log-transforming anatomical connectivity, toss zero values
# sc_adjacency = np.log10(ConnectivityMatrix.to_numpy().copy()[AC.upper_inds][keep_inds])
sc_adjacency = np.log10(ConnectivityMatrix.to_numpy().copy()[AC.upper_inds][keep_inds])

fc_adjacency = CorrelationMatrix.to_numpy().copy()[FC.upper_inds][keep_inds]

r, p = pearsonr(sc_adjacency, fc_adjacency)

fh, ax = plt.subplots(1, 3, figsize=(12,4))
ax[0].imshow(ConnectivityMatrix)
ax[1].imshow(CorrelationMatrix)
ax[2].plot(sc_adjacency, fc_adjacency, 'ko', alpha=0.25)
ax[2].set_title('r={:.2f}'.format(r))

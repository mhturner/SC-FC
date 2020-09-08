import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import networkx as nx
from scipy.stats import pearsonr
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

src = 0
trg = 18

anat_connect = AC.getConnectivityMatrix('CellCount', diag=0)
adj = anat_connect.to_numpy()

graph = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)

flows = np.zeros((len(AC.rois), len(AC.rois)))
for r in range(len(AC.rois)):
    for c in range(len(AC.rois)):
        if r != c:
            flow_val, flow_dict = nx.flow.maximum_flow(graph, _s=r, _t=c, capacity='weight')
            flows[r, c] = flow_val

# %%
f_conn = (flows + flows.T)/2
f_conn = f_conn[AC.upper_inds]
fc = FC.CorrelationMatrix.to_numpy()[FC.upper_inds]

plt.plot(f_conn, fc, 'ko')

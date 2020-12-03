
import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import os
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import seaborn as sns
import socket
import glob

from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
import matplotlib
from matplotlib import rcParams
rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})
rcParams.update({'axes.spines.right': False})
rcParams.update({'axes.spines.top': False})
rcParams['svg.fonttype'] = 'none' # let illustrator handle the font type
rcParams['pdf.fonttype'] = 42

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
SynapseCount = pd.DataFrame(columns=['source', 'target', 'cell', 'num_tbars', 'weighted_tbars'])

SynapseCount
# %%
FC.roi_size

FC.CorrelationMatrix.mean()

fh, ax = plt.subplots(1, 2, figsize=(8,4))
ax[0].plot(FC.roi_size, FC.CorrelationMatrix.mean(), 'ko')
ax[0].set_xlabel('Roi size')
ax[0].set_ylabel('Region avg. FC')
ax[1].plot(FC.roi_size, AC.getConnectivityMatrix('CellCount').mean(), 'ko')
ax[1].set_xlabel('Roi size')
ax[1].set_ylabel('Region avg. SC')



# %%

ConnectivityCount = AC.getConnectivityMatrix('CellCount')
ConnectivityTBars = AC.getConnectivityMatrix('TBars')

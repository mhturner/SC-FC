
import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import os
from scipy.stats import pearsonr
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
roi_completeness = anatomical_connectivity.getRoiCompleteness(neuprint_client, bridge.getRoiMapping())
np.mean(roi_completeness['frac_pre'])
np.sum(FC.roi_size)

# %%

ConnectivityCount = AC.getConnectivityMatrix('CellCount')
ConnectivityTBars = AC.getConnectivityMatrix('TBars')


r = pearsonr(ConnectivityCount.to_numpy().ravel(), ConnectivityTBars.to_numpy().ravel())

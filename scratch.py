import matplotlib.pyplot as plt
from neuprint import Client, fetch_neurons, NeuronCriteria
import numpy as np
import networkx as nx
from scipy.stats import pearsonr, spearmanr
import os
import socket
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_validate
from scipy.stats import norm

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

# %% multiple regression model on direct + shortest path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_validate

rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=0)

# 1: direct connectivity
direct_connect, keep_inds = AC.getAdjacency('TBars', do_log=True)

# 2: shortest path
anat_connect = AC.getConnectivityMatrix('TBars', diag=None)
measured_sp, measured_steps, _, measured_hub = bridge.getShortestPathStats(anat_connect)
shortest_path = np.log10(((measured_sp.T + measured_sp.T)/2).to_numpy()[FC.upper_inds][keep_inds])


x = np.vstack([direct_connect,
               shortest_path]).T
# x = shortest_path
# x = x.reshape(-1, 1)

measured_fc = FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]
regressor = LinearRegression()
regressor.fit(x, measured_fc);

pred = regressor.predict(x)
cv_results = cross_validate(regressor, x, measured_fc, cv=rkf, scoring='r2');
avg_r2 = cv_results['test_score'].mean()
err = cv_results['test_score'].std()
print('r2 = {:.2f}+/-{:.2f}'.format(avg_r2, err))
fh, ax = plt.subplots(1, 1, figsize=(6,6))
ax.plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
ax.plot(pred, measured_fc, 'ko', alpha=0.25)
ax.annotate('$r^2$={:.2f}'.format(avg_r2), (-0.15, 0.95))
ax.set_ylabel('Measured FC (z)')
ax.set_xlabel('Predicted FC (z)')
ax.set_xlim([-0.2, 1.0])
ax.set_aspect('equal')

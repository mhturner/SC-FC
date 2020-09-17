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


# %% DIRECT CONN MODEL


rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=0)

# 1: Cell count measured
ConnectivityMatrix = AC.getConnectivityMatrix('CellCount', diag=0, symmetrize=True)
keep_inds = np.where(ConnectivityMatrix.to_numpy()[AC.upper_inds] > 0) # for log-transforming anatomical connectivity, toss zero values
x = np.log10(ConnectivityMatrix.to_numpy().copy()[AC.upper_inds][keep_inds])
measured_fc = FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]
x = x.reshape(-1, 1)
regressor = LinearRegression()
regressor.fit(x, measured_fc);

pred = regressor.predict(x)
cv_results = cross_validate(regressor, x, measured_fc, cv=rkf, scoring='r2');
avg_r2 = cv_results['test_score'].mean()
err = cv_results['test_score'].std()
print('r2 = {:.2f}+/-{:.2f}'.format(avg_r2, err))
fh, ax = plt.subplots(1, 1, figsize=(3,3))
ax.plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
ax.plot(pred, measured_fc, 'ko', alpha=0.25)
ax.annotate('$r^2$={:.2f}'.format(avg_r2), (-0.15, 0.95))
ax.set_ylabel('Measured FC (z)')
ax.set_xlabel('Predicted FC (z)')
ax.set_xlim([-0.2, 1.0])
ax.set_aspect('equal')

# 2 norm random model
norm_model = norm(loc=np.mean(ConnectivityMatrix.to_numpy()), scale=np.std(ConnectivityMatrix.to_numpy()))
norm_vals = norm_model.rvs(size=(len(AC.rois), len(AC.rois)))
norm_vals[norm_vals<0] = 0
keep_inds = np.where(norm_vals[AC.upper_inds] > 0) # for log-transforming anatomical connectivity, toss zero values
x = np.log10(norm_vals[AC.upper_inds][keep_inds])
x = x.reshape(-1, 1)
norm_fc = regressor.predict(x)

# 3 lognorm model
conns = ConnectivityMatrix.to_numpy().copy()
lognorm_model = norm(loc=np.mean(np.log10(conns[conns>0])), scale=np.std(np.log10(conns[conns>0])))
lognorm_vals = lognorm_model.rvs(size=(len(AC.rois), len(AC.rois)))
lognorm_vals[lognorm_vals<0] = 0
lognorm_vals = 10**lognorm_vals
keep_inds = np.where(lognorm_vals[AC.upper_inds] > 0) # for log-transforming anatomical connectivity, toss zero values
x = np.log10(lognorm_vals[AC.upper_inds][keep_inds])
x = x.reshape(-1, 1)
lognorm_fc = regressor.predict(x)

nbins = 25
fh, ax = plt.subplots(1, 1, figsize=(8,4))
vals, bins = np.histogram(measured_fc, bins=nbins, density=True)
ax.plot(bins[:-1], vals/np.sum(vals), 'k')

vals, bins = np.histogram(norm_fc, bins=nbins, density=True)
ax.plot(bins[:-1], vals/np.sum(vals), 'b')

vals, bins = np.histogram(lognorm_fc, bins=nbins, density=True)
ax.plot(bins[:-1], vals/np.sum(vals), 'r')

# %% SHORTEST PATH MODEL
import pandas as pd

rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=0)

# 1: Cell count measured
anat_connect = AC.getConnectivityMatrix('CellCount', diag=None)
measured_sp, measured_steps, _, measured_hub = bridge.getShortestPathStats(anat_connect)
x = np.log10(((measured_sp.T + measured_sp.T)/2).to_numpy()[FC.upper_inds])
measured_fc = FC.CorrelationMatrix.to_numpy()[FC.upper_inds]
x = x.reshape(-1, 1)
regressor = LinearRegression()
regressor.fit(x, measured_fc);

pred = regressor.predict(x)
cv_results = cross_validate(regressor, x, measured_fc, cv=rkf, scoring='r2');
avg_r2 = cv_results['test_score'].mean()
err = cv_results['test_score'].std()
print('r2 = {:.2f}+/-{:.2f}'.format(avg_r2, err))
fh, ax = plt.subplots(1, 1, figsize=(3,3))
ax.plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
ax.plot(pred, measured_fc, 'ko', alpha=0.25)
ax.annotate('$r^2$={:.2f}'.format(avg_r2), (-0.15, 0.95))
ax.set_ylabel('Measured FC (z)')
ax.set_xlabel('Predicted FC (z)')
ax.set_xlim([-0.2, 1.0])
ax.set_aspect('equal')

# 2 norm random model
norm_model = norm(loc=np.mean(ConnectivityMatrix.to_numpy()), scale=np.std(ConnectivityMatrix.to_numpy()))
norm_vals = norm_model.rvs(size=(len(AC.rois), len(AC.rois)))
norm_vals[norm_vals<0] = 0
norm_df = pd.DataFrame(data=norm_vals, index=AC.rois, columns=AC.rois)
norm_sp, norm_steps, _, norm_hub = bridge.getShortestPathStats(norm_df)
x = np.log10(((norm_sp.T + norm_sp.T)/2).to_numpy()[FC.upper_inds])
x = x.reshape(-1, 1)
norm_fc = regressor.predict(x)

# 3 lognorm model
conns = ConnectivityMatrix.to_numpy().copy()
lognorm_model = norm(loc=np.mean(np.log10(conns[conns>0])), scale=np.std(np.log10(conns[conns>0])))
lognorm_vals = lognorm_model.rvs(size=(len(AC.rois), len(AC.rois)))
lognorm_vals[lognorm_vals<0] = 0
lognorm_vals = 10**lognorm_vals
lognorm_df = pd.DataFrame(data=lognorm_vals, index=AC.rois, columns=AC.rois)
lognorm_sp, lognorm_steps, _, lognorm_hub = bridge.getShortestPathStats(lognorm_df)
x = np.log10(((lognorm_sp.T + lognorm_sp.T)/2).to_numpy()[FC.upper_inds])
x = x.reshape(-1, 1)
lognorm_fc = regressor.predict(x)

nbins = 25
fh, ax = plt.subplots(1, 1, figsize=(8,4))
vals, bins = np.histogram(measured_fc, bins=nbins, density=True)
ax.plot(bins[:-1], vals/np.sum(vals), 'k', label='Measured')

vals, bins = np.histogram(norm_fc, bins=nbins, density=True)
ax.plot(bins[:-1], vals/np.sum(vals), color=plot_colors[0], label='norm. model')

vals, bins = np.histogram(lognorm_fc, bins=nbins, density=True)
ax.plot(bins[:-1], vals/np.sum(vals), color=plot_colors[1], label='lognorm. model')
ax.set_xlabel('Functional connectivity (z)')
ax.set_ylabel('Prob.')
ax.legend()

# %%
nbin=5
fh, ax = plt.subplots(1, 1, figsize=(3,3))
vals, bins = np.histogram(measured_steps.to_numpy().ravel(), bins=nbins, density=True)
ax.plot(bins[:-1], vals/np.sum(vals), label='Measured', color='k')

vals, bins = np.histogram(norm_steps.to_numpy().ravel(), bins=nbins, density=True)
ax.plot(bins[:-1], vals/np.sum(vals), label='Norm. model', color=plot_colors[0])

vals, bins = np.histogram(lognorm_steps.to_numpy().ravel(), bins=nbins, density=True)
ax.plot(bins[:-1], vals/np.sum(vals), label='Lognorm. model', color=plot_colors[1])
ax.legend()


nbin=25
fh, ax = plt.subplots(1, 1, figsize=(3,3))
vals, bins = np.histogram(measured_hub.to_numpy().ravel(), bins=nbins, density=True)
ax.plot(bins[:-1], vals/np.sum(vals), label='Measured', color='k')

vals, bins = np.histogram(norm_hub.to_numpy().ravel(), bins=nbins, density=True)
ax.plot(bins[:-1], vals/np.sum(vals), label='Norm. model', color=plot_colors[0])

vals, bins = np.histogram(lognorm_hub.to_numpy().ravel(), bins=nbins, density=True)
ax.plot(bins[:-1], vals/np.sum(vals), label='Lognorm. model', color=plot_colors[1])

ax.legend()

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

# %% SHORTEST PATH MODEL

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


iterations = 10
# 2 norm random model
norm_fc = []
norm_sp = []
norm_dc = []
for it in range(iterations):
    norm_adj = AC.makeModelAdjacency(type='CellCount', model='norm', by_row=True)
    sp, norm_steps, _, norm_hub = bridge.getShortestPathStats(norm_adj)
    x = np.log10(((sp.T + sp.T)/2).to_numpy()[FC.upper_inds])
    x = x.reshape(-1, 1)
    norm_fc.append(regressor.predict(x))
    norm_sp.append(sp)
    norm_dc.append(norm_adj.to_numpy().ravel())
norm_fc = np.vstack(norm_fc)
norm_sp = np.vstack(norm_sp)
norm_dc = np.vstack(norm_dc)

iterations = 10
# 3 norm random model
uniform_fc = []
uniform_sp = []
uniform_dc = []
for it in range(iterations):
    norm_adj = AC.makeModelAdjacency(type='CellCount', model='uniform', by_row=True)
    sp, norm_steps, _, norm_hub = bridge.getShortestPathStats(norm_adj)
    x = np.log10(((sp.T + sp.T)/2).to_numpy()[FC.upper_inds])
    x = x.reshape(-1, 1)
    uniform_fc.append(regressor.predict(x))
    uniform_sp.append(sp)
    uniform_dc.append(norm_adj.to_numpy().ravel())
uniform_fc = np.vstack(uniform_fc)
uniform_sp = np.vstack(uniform_sp)
uniform_dc = np.vstack(uniform_dc)

# 4 lognorm model
lognorm_fc = []
lognorm_sp = []
lognorm_dc = []
for it in range(iterations):
    lognorm_adj = AC.makeModelAdjacency(type='CellCount', model='lognorm', by_row=True)
    sp, lognorm_steps, _, lognorm_hub = bridge.getShortestPathStats(lognorm_adj)
    x = np.log10(((sp.T + sp.T)/2).to_numpy()[FC.upper_inds])
    x = x.reshape(-1, 1)
    lognorm_fc.append(regressor.predict(x))
    lognorm_sp.append(sp)
    lognorm_dc.append(lognorm_adj.to_numpy().ravel())
lognorm_fc = np.vstack(lognorm_fc)
lognorm_sp = np.vstack(lognorm_sp)
lognorm_dc = np.vstack(lognorm_dc)


# %% PLOT

def plotHistograms(measured, norm, uniform, lognorm, bins, ax):
    vals, bins = np.histogram(measured, bins=bins, density=True)
    ax.plot(bins[:-1], vals/np.sum(vals), 'k-o', label='Measured')

    vals = []
    for it in range(iterations):
        new_vals, _ = np.histogram(norm[it, :], bins=bins, density=True)
        vals.append(new_vals/np.sum(new_vals))
    vals = np.vstack(vals)
    avg = np.mean(vals, axis=0)
    err = np.std(vals, axis=0) / np.sqrt(iterations)
    ax.fill_between(bins[:-1], avg-err, avg+err, color=plot_colors[0], alpha=0.25)
    ax.plot(bins[:-1], avg, color=plot_colors[0], label='norm. model')

    vals = []
    for it in range(iterations):
        new_vals, bins = np.histogram(uniform[it, :], bins=bins, density=True)
        vals.append(new_vals/np.sum(new_vals))
    vals = np.vstack(vals)
    avg = np.mean(vals, axis=0)
    err = np.std(vals, axis=0) / np.sqrt(iterations)
    ax.fill_between(bins[:-1], avg-err, avg+err, color=plot_colors[1], alpha=0.25)
    ax.plot(bins[:-1], avg, color=plot_colors[1], label='uniform model')

    vals = []
    for it in range(iterations):
        new_vals, bins = np.histogram(lognorm[it, :], bins=bins, density=True)
        vals.append(new_vals/np.sum(new_vals))
    vals = np.vstack(vals)
    avg = np.mean(vals, axis=0)
    err = np.std(vals, axis=0) / np.sqrt(iterations)
    ax.fill_between(bins[:-1], avg-err, avg+err, color=plot_colors[2], alpha=0.25)
    ax.plot(bins[:-1], avg, color=plot_colors[2], label='lognorm. model')


fh, ax = plt.subplots(1, 3, figsize=(12,4))
# direct conn.
bins = np.logspace(0.1, 4, 20)
measured_dc = AC.getConnectivityMatrix('CellCount')
plotHistograms(measured_dc, norm_dc, uniform_dc, lognorm_dc, bins, ax[0])
ax[0].set_xlabel('Direct connectivity (cells)')
ax[0].set_xscale('log')
ax[0].set_ylabel('Prob.')
ax[0].legend()

# shortest path distances
bins = np.logspace(-3.7, -1.75, 20)
plotHistograms(measured_sp, norm_sp, uniform_sp, lognorm_sp, bins, ax[1])
ax[1].set_xlabel('Shortest path distance')
ax[1].set_xscale('log')
ax[1].set_ylabel('Prob.')

# FC
bins = 20
# plotHistograms(pred, norm_fc, uniform_fc, lognorm_fc, bins, ax[2])
plotHistograms(measured_fc, norm_fc, uniform_fc, lognorm_fc, bins, ax[2])
ax[2].set_xlabel('Functional connectivity (z)')
ax[2].set_ylabel('Prob.')

# %%
# TODO: need a stat model for connectivity that reproduces FC well with this regression model

# %% DIRECT CELL COUNT MODEL

rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=0)

# 1: Cell count measured

anat_connect, keep_inds = AC.getAdjacency('CellCount', do_log=True)
x = anat_connect
measured_fc = FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]
x = x.reshape(-1, 1)
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

iterations = 10
# 2 norm random model
norm_fc = []
norm_dc = []
for it in range(iterations):
    norm_adj = AC.makeModelAdjacency(type='CellCount', model='norm', by_row=True)
    keep_inds = np.where(norm_adj.to_numpy()[AC.upper_inds] > 0)
    x = np.log10(norm_adj.to_numpy()[FC.upper_inds][keep_inds])
    x = x.reshape(-1, 1)
    norm_fc.append(regressor.predict(x))
    norm_dc.append(norm_adj.to_numpy().ravel())

# 3 uniform random model
uniform_fc = []
uniform_sp = []
uniform_dc = []
for it in range(iterations):
    uniform_adj = AC.makeModelAdjacency(type='CellCount', model='uniform', by_row=True)
    keep_inds = np.where(uniform_adj.to_numpy()[AC.upper_inds] > 0)
    x = np.log10(uniform_adj.to_numpy()[FC.upper_inds][keep_inds])
    x = x.reshape(-1, 1)
    uniform_fc.append(regressor.predict(x))
    uniform_dc.append(norm_adj.to_numpy().ravel())


# 4 lognorm model
lognorm_fc = []
lognorm_sp = []
lognorm_dc = []
for it in range(iterations):
    lognorm_adj = AC.makeModelAdjacency(type='CellCount', model='lognorm', by_row=True)
    keep_inds = np.where(lognorm_adj.to_numpy()[AC.upper_inds] > 0)
    x = np.log10(lognorm_adj.to_numpy()[FC.upper_inds][keep_inds])
    x = x.reshape(-1, 1)
    lognorm_fc.append(regressor.predict(x))
    lognorm_dc.append(lognorm_adj.to_numpy().ravel())



# %% PLOT

def plotHistograms(measured, norm, uniform, lognorm, bins, ax):
    vals, bins = np.histogram(measured, bins=bins, density=True)
    ax.plot(bins[:-1], vals/np.sum(vals), 'k-o', label='Measured')

    vals = []
    for it in range(iterations):
        new_vals, _ = np.histogram(norm[it], bins=bins, density=True)
        vals.append(new_vals/np.sum(new_vals))
    vals = np.vstack(vals)
    avg = np.mean(vals, axis=0)
    err = np.std(vals, axis=0) / np.sqrt(iterations)
    ax.fill_between(bins[:-1], avg-err, avg+err, color=plot_colors[0], alpha=0.25)
    ax.plot(bins[:-1], avg, color=plot_colors[0], label='norm. model')

    vals = []
    for it in range(iterations):
        new_vals, bins = np.histogram(uniform[it], bins=bins, density=True)
        vals.append(new_vals/np.sum(new_vals))
    vals = np.vstack(vals)
    avg = np.mean(vals, axis=0)
    err = np.std(vals, axis=0) / np.sqrt(iterations)
    ax.fill_between(bins[:-1], avg-err, avg+err, color=plot_colors[1], alpha=0.25)
    ax.plot(bins[:-1], avg, color=plot_colors[1], label='uniform model')

    vals = []
    for it in range(iterations):
        new_vals, bins = np.histogram(lognorm[it], bins=bins, density=True)
        vals.append(new_vals/np.sum(new_vals))
    vals = np.vstack(vals)
    avg = np.mean(vals, axis=0)
    err = np.std(vals, axis=0) / np.sqrt(iterations)
    ax.fill_between(bins[:-1], avg-err, avg+err, color=plot_colors[2], alpha=0.25)
    ax.plot(bins[:-1], avg, color=plot_colors[2], label='lognorm. model')


fh, ax = plt.subplots(1, 2, figsize=(14,7))
# direct conn.
bins = np.logspace(0.1, 4, 20)
measured_dc = AC.getConnectivityMatrix('CellCount')
plotHistograms(measured_dc, norm_dc, uniform_dc, lognorm_dc, bins, ax[0])
ax[0].set_xlabel('Direct connectivity (cells)')
ax[0].set_xscale('log')
ax[0].set_ylabel('Prob.')
ax[0].legend()

# FC
bins = 20
plotHistograms(measured_fc, norm_fc, uniform_fc, lognorm_fc, bins, ax[1])
ax[1].set_xlabel('Functional connectivity (z)')
ax[1].set_ylabel('Prob.')

# %%

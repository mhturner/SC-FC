import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import os
from scipy.stats import norm, zscore, kstest
import socket

from scfc import bridge, anatomical_connectivity, functional_connectivity
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
save_dpi = 400


# %% ~Lognormal distribtution of connection strengths
ConnectivityCount = AC.getConnectivityMatrix('CellCount')
ConnectivityTBars = AC.getConnectivityMatrix('TBars')


pull_region = 'AL(R)'

fig1_0, ax = plt.subplots(2, 1, figsize=(4.5, 4.5))
ax = ax.ravel()
fig1_0.tight_layout(w_pad=2, h_pad=8)

figS1_0, axS1 = plt.subplots(9, 4, figsize=(8, 9))
axS1 = axS1.ravel()

z_scored_cell = []
z_scored_tbar = []
for p_ind, pr in enumerate(ConnectivityCount.index):
    # # # # CELL COUNT:
    outbound = ConnectivityCount.loc[pr, :]
    outbound = outbound.sort_values(ascending=False)
    ki = np.where(outbound > 0)
    ct = outbound.iloc[ki]
    z_scored_cell.append(zscore(np.log10(ct)))

    lognorm_model = norm(loc=np.mean(np.log10(ct)), scale=np.std(np.log10(ct)))
    iterations = 1000
    samples = []
    for it in range(iterations):
        new_samples = np.sort(lognorm_model.rvs(size=len(ct)))[::-1]
        samples.append(new_samples)
    samples = np.vstack(samples)
    # Note order of operations matters here, get mean and std before going back out of log
    mod_mean = 10**np.mean(samples, axis=0)
    err_down = 10**(np.mean(samples, axis=0) - 2*np.std(samples, axis=0))
    err_up = 10**(np.mean(samples, axis=0) + 2*np.std(samples, axis=0))

    axS1[p_ind].fill_between(list(range(len(mod_mean))), err_up, err_down, color='k', alpha=0.4, rasterized=False)
    axS1[p_ind].plot(mod_mean, 'k--', rasterized=False)
    axS1[p_ind].plot(ct, marker='.', linestyle='none', rasterized=False)

    axS1[p_ind].set_xticks([])
    axS1[p_ind].annotate('{}'.format(pr), (12, 6e3), fontsize=8)
    axS1[p_ind].set_yscale('log')
    axS1[p_ind].set_ylim([0.2, 5e4])
    axS1[p_ind].set_yticks([1e0, 1e2, 1e4])

    if pr == pull_region:
        ax[0].fill_between(list(range(len(mod_mean))), err_up, err_down, color='k', alpha=0.4)
        ax[0].plot(mod_mean, 'k--')
        ax[0].plot(ct, marker='o', linestyle='none')

        ax[0].set_xticks(list(range(len(ct))))
        ax[0].tick_params(axis='both', which='major', labelsize=10)
        ax[0].set_xticklabels(ct.index)
        ax[0].set_yscale('log')
        ax[0].set_ylim([0.05, 5e4])
        for tick in ax[0].get_xticklabels():
            tick.set_rotation(90)
            tick.set_fontsize(8)
        ax[0].set_ylabel('Cells')
        ax[0].annotate('Source: {}'.format(pr), (12, 1e4), fontsize=14)


    # # # # TBAR COUNT:
    outbound = ConnectivityTBars.loc[pr, :]
    outbound = outbound.sort_values(ascending=False)
    ki = np.where(outbound > 0)
    ct = outbound.iloc[ki]
    z_scored_tbar.append(zscore(np.log10(ct)))

    if pr == pull_region:
        lognorm_model = norm(loc=np.mean(np.log10(ct)), scale=np.std(np.log10(ct)))
        iterations = 1000
        samples = []
        for it in range(iterations):
            new_samples = np.sort(lognorm_model.rvs(size=len(ct)))[::-1]
            samples.append(new_samples)
        samples = np.vstack(samples)

        # Note order of operations matters here, get mean and std before going back out of log
        mod_mean = 10**np.mean(samples, axis=0)
        err_down = 10**(np.mean(samples, axis=0) - 2*np.std(samples, axis=0))
        err_up = 10**(np.mean(samples, axis=0) + 2*np.std(samples, axis=0))
        ax[1].fill_between(list(range(len(mod_mean))), err_up, err_down, color='k', alpha=0.4)
        ax[1].plot(mod_mean, 'k--')
        ax[1].plot(ct, marker='o', linestyle='none')

        ax[1].set_xticks(list(range(len(ct))))
        ax[1].tick_params(axis='both', which='major', labelsize=10)
        ax[1].set_xticklabels(ct.index)
        ax[1].set_yscale('log')
        ax[1].set_ylim([0.05, 5e6])
        ax[1].tick_params(axis='y', which='minor')
        for tick in ax[1].get_xticklabels():
            tick.set_rotation(90)
            tick.set_fontsize(8)
        ax[1].set_ylabel('T-Bars')

figS1_0.text(-0.01, 0.5, 'Connections from source region (cells)', va='center', rotation='vertical', fontsize=14)

fig1_0.savefig(os.path.join(analysis_dir, 'figpanels', 'fig1_0.svg'), format='svg', transparent=True, dpi=save_dpi)
figS1_0.savefig(os.path.join(analysis_dir, 'figpanels', 'figS1_0.svg'), format='svg', transparent=True, dpi=save_dpi)

# %% Summary across all regions: zscore within each outgoing and compare to lognorm
# CELL COUNT:
p_vals = []
for arr in z_scored_cell:
    _, p = kstest(arr, 'norm')
    p_vals.append(p)

data = np.hstack(z_scored_cell)

# fit norm model on log transformed data
params = norm.fit(data)
norm_model = norm(loc=params[0], scale=params[1])
theory_distr = []
for iter in range(100):
    theory_distr.append(norm_model.rvs(size=len(data)))
theory_distr = np.vstack(theory_distr)

fig1_1, ax = plt.subplots(1, 1, figsize=(3.5, 2.25))
val, bin = np.histogram(data, 20, density=True)
bin_ctrs = bin[:-1]
xx = np.linspace(-3.5, 3.5)
ax.plot(10**xx, norm_model.pdf(xx), linewidth=2, color='k', linestyle='--')
ax.plot(10**bin_ctrs, val, linewidth=3)
ax.set_xscale('log')
ax.set_xlabel('Cells (z-score)')
ax.set_ylabel('Prob.')
ax.set_xticks([1e-2, 1, 1e2])


# Q-Q plot of log-transformed data vs. fit normal distribution
fig1_2, ax = plt.subplots(1, 1, figsize=(1.2, 1.2))
ax.plot([10**-4, 10**4], [10**-4, 10**4], 'k-')
quants = np.linspace(0, 1, 20)
for q in quants:
    th_pts = np.quantile(theory_distr, q, axis=1)  # quantile value for each iteration
    ax.plot([10**np.quantile(data, q), 10**np.quantile(data, q)], [10**(np.mean(th_pts) - 2*np.std(th_pts)), 10**(np.mean(th_pts) + 2*np.std(th_pts))], color=plot_colors[0], alpha=1)
    ax.plot(10**np.quantile(data, q), 10**np.mean(th_pts), marker='o', color=plot_colors[0], alpha=1)
# ax.set_xlabel('Cells', fontsize=8)
# ax.set_ylabel('Lognorm.', fontsize=8)
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xticks([1e-2, 1e2])
ax.set_yticks([1e-2, 1e2])
ax.set_xticklabels(['-2$\sigma$', '+2$\sigma$'])
ax.set_yticklabels(['-2$\sigma$', '+2$\sigma$'])
ax.axhline(y=1, color='k', zorder=0, alpha=0.5)
ax.axvline(x=1, color='k', zorder=0, alpha=0.5)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(8)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# # # # TBARS # # # # # # # # # # # # # # # #:
p_vals = []
for arr in z_scored_tbar:
    _, p = kstest(arr, 'norm')
    p_vals.append(p)

data = np.hstack(z_scored_tbar)

# fit norm model on log transformed data
params = norm.fit(data)
norm_model = norm(loc=params[0], scale=params[1])
theory_distr = []
for iter in range(100):
    theory_distr.append(norm_model.rvs(size=len(data)))
theory_distr = np.vstack(theory_distr)

fig1_3, ax = plt.subplots(1, 1, figsize=(3.5, 2.25))
val, bin = np.histogram(data, 20, density=True)
bin_ctrs = bin[:-1]
xx = np.linspace(-3.5, 3.5)
ax.plot(10**xx, norm_model.pdf(xx), linewidth=2, color='k', linestyle='--')
ax.plot(10**bin_ctrs, val, linewidth=3)
ax.set_xscale('log')
ax.set_xlabel('T-Bars (z-score)')
ax.set_ylabel('Prob.')
ax.set_xticks([1e-2, 1, 1e2])


# Q-Q plot of log-transformed data vs. fit normal distribution
fig1_4, ax = plt.subplots(1, 1, figsize=(1.2, 1.2))
ax.plot([10**-4, 10**4], [10**-4, 10**4], 'k-')
quants = np.linspace(0, 1, 20)
for q in quants:
    th_pts = np.quantile(theory_distr, q, axis=1)  # quantile value for each iteration
    ax.plot([10**np.quantile(data, q), 10**np.quantile(data, q)], [10**(np.mean(th_pts) - 2*np.std(th_pts)), 10**(np.mean(th_pts) + 2*np.std(th_pts))], color=plot_colors[0], alpha=1)
    ax.plot(10**np.quantile(data, q), 10**np.mean(th_pts), marker='o', color=plot_colors[0], alpha=1)
# ax.set_xlabel('Cells', fontsize=8)
# ax.set_ylabel('Lognorm.', fontsize=8)
ax.set_xscale('log')
ax.set_yscale('log')


ax.set_xticks([1e-2, 1e2])
ax.set_yticks([1e-2, 1e2])
ax.set_xticklabels(['-2$\sigma$', '+2$\sigma$'])
ax.set_yticklabels(['-2$\sigma$', '+2$\sigma$'])
ax.axhline(y=1, color='k', zorder=0, alpha=0.5)
ax.axvline(x=1, color='k', zorder=0, alpha=0.5)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(8)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(8)


fig1_1.savefig(os.path.join(analysis_dir, 'figpanels', 'fig1_1.svg'), format='svg', transparent=True, dpi=save_dpi)
fig1_2.savefig(os.path.join(analysis_dir, 'figpanels', 'fig1_2.svg'), format='svg', transparent=True, dpi=save_dpi)
fig1_3.savefig(os.path.join(analysis_dir, 'figpanels', 'fig1_3.svg'), format='svg', transparent=True, dpi=save_dpi)
fig1_4.savefig(os.path.join(analysis_dir, 'figpanels', 'fig1_4.svg'), format='svg', transparent=True, dpi=save_dpi)

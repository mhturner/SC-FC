import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import os
from scipy.stats import norm, zscore, kstest, pearsonr
import pandas as pd
import seaborn as sns
import socket

from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
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


# %% ~Lognormal distribtution of connection strengths
ConnectivityCount = AC.getConnectivityMatrix('CellCount')

pull_regions = ['AL(R)', 'LH(R)']
pull_inds = [np.where(np.array(FC.rois) == x)[0][0] for x in pull_regions]

fig1_0, ax = plt.subplots(2, 1, figsize=(5, 5))
ax = ax.ravel()
fig1_0.tight_layout(w_pad=2, h_pad=8)

figS1_0, axS1 = plt.subplots(4, 9, figsize=(18, 6))
axS1 = axS1.ravel()

z_scored_data = []
for p_ind, pr in enumerate(ConnectivityCount.index):
    outbound = ConnectivityCount.loc[pr, :]
    outbound = outbound.sort_values(ascending=False)
    ki = np.where(outbound > 0)
    ct = outbound.iloc[ki]

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

    z_scored_data.append(zscore(np.log10(ct)))

    axS1[p_ind].fill_between(list(range(len(mod_mean))), err_up, err_down, color='k', alpha=0.4)
    axS1[p_ind].plot(mod_mean, 'k--')
    axS1[p_ind].plot(ct, marker='o', linestyle='none')

    axS1[p_ind].set_xticks([])
    axS1[p_ind].annotate('{}'.format(pr), (12, 6e3), fontsize=8)
    axS1[p_ind].set_yscale('log')
    axS1[p_ind].set_ylim([0.2, 5e4])

    if pr in pull_regions:
        eg_ind = np.where(pr == np.array(pull_regions))[0][0]
        ax[eg_ind].fill_between(list(range(len(mod_mean))), err_up, err_down, color='k', alpha=0.4)
        ax[eg_ind].plot(mod_mean, 'k--')
        ax[eg_ind].plot(ct, marker='o', linestyle='none')

        ax[eg_ind].set_xticks(list(range(len(ct))))
        ax[eg_ind].tick_params(axis='both', which='major', labelsize=10)
        ax[eg_ind].set_xticklabels(ct.index)
        ax[eg_ind].set_yscale('log')
        ax[eg_ind].set_ylim([0.05, 5e4])
        for tick in ax[eg_ind].get_xticklabels():
            tick.set_rotation(90)
            tick.set_fontsize(8)

        ax[eg_ind].annotate('Source: {}'.format(pr), (12, 1e4), fontsize=14)

fig1_0.text(-0.01, 0.6, 'Connecting cells', va='center', rotation='vertical', fontsize=14)
figS1_0.text(-0.01, 0.5, 'Connections from source region (cells)', va='center', rotation='vertical', fontsize=14)

frac_inside_shading = np.sum(np.abs(np.hstack(z_scored_data)) <= 2) / np.hstack(z_scored_data).size

p_vals = []
for arr in z_scored_data:
    _, p = kstest(arr, 'norm')
    p_vals.append(p)

print(p_vals)
fig1_1, ax = plt.subplots(2, 1, figsize=(3, 5))

data = np.hstack(z_scored_data)

# fit norm model on log transformed data
params = norm.fit(data)
norm_model = norm(loc=params[0], scale=params[1])
theory_distr = []
for iter in range(100):
    theory_distr.append(norm_model.rvs(size=len(data)))
theory_distr = np.vstack(theory_distr)

val, bin = np.histogram(data, 20, density=True)
bin_ctrs = bin[:-1]
xx = np.linspace(-3.5, 3.5)
ax[0].plot(10**xx, norm_model.pdf(xx), linewidth=2, color='k', linestyle='--')
ax[0].plot(10**bin_ctrs, val, linewidth=3)
ax[0].set_xscale('log')
ax[0].set_xlabel('Cell count (z-scored)')
ax[0].set_ylabel('Probability')
ax[0].set_xscale('log')
ax[0].set_xticks([1e-2, 1, 1e2])

# Q-Q plot of log-transformed data vs. fit normal distribution
ax[1].plot([10**-4, 10**4], [10**-4, 10**4], 'k--')
quants = np.linspace(0, 1, 20)
for q in quants:
    th_pts = np.quantile(theory_distr, q, axis=1)  # quantile value for each iteration
    ax[1].plot([10**np.quantile(data, q), 10**np.quantile(data, q)], [10**(np.mean(th_pts) - 2*np.std(th_pts)), 10**(np.mean(th_pts) + 2*np.std(th_pts))], color=plot_colors[0])
    ax[1].plot(10**np.quantile(data, q), 10**np.mean(th_pts), marker='o', color=plot_colors[0])
ax[1].set_xlabel('Data quantile')
ax[1].set_ylabel('Lognormal quantile')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ticks = [1e-2, 1, 1e2]
ax[1].set_xticks(ticks)
ax[1].set_yticks(ticks)

fig1_0.savefig(os.path.join(analysis_dir, 'figpanels', 'fig1_0.svg'), format='svg', transparent=True)
fig1_1.savefig(os.path.join(analysis_dir, 'figpanels', 'fig1_1.svg'), format='svg', transparent=True)
figS1_0.savefig(os.path.join(analysis_dir, 'figpanels', 'figS1_0.svg'), format='svg', transparent=True)
# %% Eg region traces and cross corrs
pull_regions = ['AL(R)', 'CAN(R)', 'LH(R)', 'SPS(R)']
pull_inds = [np.where(np.array(FC.rois) == x)[0][0] for x in pull_regions]

resp_fp = os.path.join(data_dir, 'region_responses', '2018-11-03_5.pkl')
voxel_size = [3, 3, 3]  # um, xyz

brain_str = '2018-11-03_5'
brain_fn = 'func_volreg_{}_meanbrain.nii'.format(brain_str)
atlas_fn = 'vfb_68_{}.nii.gz'.format(brain_str)
brain_fp = os.path.join(data_dir, 'region_responses', brain_fn)
atlas_fp = os.path.join(data_dir, 'region_responses', atlas_fn)

# load eg meanbrain and region masks
meanbrain = FC.getMeanBrain(brain_fp)
all_masks, _ = FC.loadAtlasData(atlas_fp)
masks = list(np.array(all_masks)[pull_inds])

cmap = plt.get_cmap('Set2')
colors = cmap(np.arange(len(pull_regions))/len(pull_regions))

zslices = [12, 45]
fig1_2 = plt.figure(figsize=(1.5, 4))
for z_ind, z in enumerate(zslices):
    ax = fig1_2.add_subplot(3, 1, z_ind+2)
    ax.annotate('z={} $ \mu m$'.format(z*voxel_size[2]), (1, 14), color='w', fontsize=10)

    overlay = plotting.overlayImage(meanbrain, masks, 0.5, colors=colors, z=z) + 60  # arbitrary brighten here for visualization

    img = ax.imshow(np.swapaxes(overlay, 0, 1), rasterized=False)
    ax.set_axis_off()
    ax.set_aspect('equal')

ax = fig1_2.add_subplot(3, 1, 1)
ax.imshow(np.mean(meanbrain, axis=2).T, cmap='inferno')
ax.annotate('Mean proj.', (23, 14), color='w', fontsize=10)
ax.set_axis_off()
ax.set_aspect('equal')

dx = 100  # um
dx_pix = int(dx / voxel_size[0])
ax.plot([5, dx_pix], [120, 120], 'w-')

# # TODO: put this df/f processing stuff in functional_connectivity
fs = 1.2  # Hz
cutoff = 0.01

x_start = 200
dt = 300  # datapts
timevec = np.arange(0, dt) / fs  # sec

file_id = resp_fp.split('/')[-1].replace('.pkl', '')
region_response = pd.read_pickle(resp_fp)
# convert to dF/F
dff = (region_response.to_numpy() - np.mean(region_response.to_numpy(), axis=1)[:, None]) / np.mean(region_response.to_numpy(), axis=1)[:, None]

# trim and filter
resp = functional_connectivity.filterRegionResponse(dff, cutoff=cutoff, fs=fs)
resp = functional_connectivity.trimRegionResponse(file_id, resp)
region_dff = pd.DataFrame(data=resp, index=region_response.index)

fig1_3, ax = plt.subplots(4, 1, figsize=(6, 6))
fig1_3.tight_layout(pad=2)
ax = ax.ravel()
[x.set_axis_off() for x in ax]
[x.set_ylim([-0.2, 0.29]) for x in ax]
[x.set_xlim([-15, timevec[-1]]) for x in ax]
for p_ind, pr in enumerate(pull_regions):
    ax[p_ind].plot(timevec, region_dff.loc[pr, x_start:(x_start+dt-1)], color=colors[p_ind])
    ax[p_ind].annotate(pr, (-10, 0) , rotation=90)

plotting.addScaleBars(ax[0], dT=5, dF=0.10, T_value=-2.5, F_value=-0.10)

fig1_4, ax = plt.subplots(3, 3, figsize=(4, 4))
fig1_4.tight_layout(h_pad=4, w_pad=4)
[x.set_xticks([]) for x in ax.ravel()]
[x.set_yticks([]) for x in ax.ravel()]
for ind_1, eg1 in enumerate(pull_regions):
    for ind_2, eg2 in enumerate(pull_regions):
        if ind_1 > ind_2:

            r, p = pearsonr(region_dff.loc[eg1, :], region_dff.loc[eg2, :])
            # print('{}/{}: r = {}'.format(eg2, eg1, r))

            # normed xcorr plot
            window_size = 180
            total_len = len(region_dff.loc[eg1, :])

            a = (region_dff.loc[eg1, :] - np.mean(region_dff.loc[eg1, :])) / (np.std(region_dff.loc[eg1, :]) * len(region_dff.loc[eg1, :]))
            b = (region_dff.loc[eg2, :] - np.mean(region_dff.loc[eg2, :])) / (np.std(region_dff.loc[eg2, :]))
            c = np.correlate(a, b, 'same')
            time = np.arange(-window_size/2, window_size/2) / fs # sec
            ax[ind_1-1, ind_2].plot(time, c[int(total_len/2-window_size/2): int(total_len/2+window_size/2)], 'k')
            ax[ind_1-1, ind_2].set_ylim([-0.2, 1])
            ax[ind_1-1, ind_2].axhline(0, color='k', alpha=0.5, linestyle='-')
            ax[ind_1-1, ind_2].axvline(0, color='k', alpha=0.5, linestyle='-')
            if ind_2==0:
                ax[ind_1-1, ind_2].set_ylabel(eg1)
            if ind_1==3:
                ax[ind_1-1, ind_2].set_xlabel(eg2)

plotting.addScaleBars(ax[0, 0], dT=-30, dF=0.25, T_value=time[-1], F_value=-0.15)
sns.despine(top=True, right=True, left=True, bottom=True)
fig1_2.savefig(os.path.join(analysis_dir, 'figpanels', 'fig1_2.svg'), format='svg', transparent=True)
fig1_3.savefig(os.path.join(analysis_dir, 'figpanels', 'fig1_3.svg'), format='svg', transparent=True)
fig1_4.savefig(os.path.join(analysis_dir, 'figpanels', 'fig1_4.svg'), format='svg', transparent=True)

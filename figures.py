from neuprint import Client
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import pearsonr, ttest_1samp, spearmanr
from scipy.stats import zscore
from scipy.stats import kstest, lognorm, norm
from scipy.signal import correlate
from dominance_analysis import Dominance

import scipy
import networkx as nx
from matplotlib import rcParams
rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})

rcParams['svg.fonttype'] = 'none'

from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting

"""
References:
https://connectome-neuprint.github.io/neuprint-python/docs/index.html
https://github.com/connectome-neuprint/neuprint-python

"""

data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC'

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1heHdlbGxob2x0ZXR1cm5lckBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpMHJRX0M4akliX0ZrS2h2OU5DSElsWlpnRDY5YUMtVGdNLWVWM3lRP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzY2MTk1MzcwfQ.Q-57D4tX2sXMjWym2LFhHaUGHgHiUsIM_JI9xekxw_0')

# Get FunctionalConnectivity object
FC = functional_connectivity.FunctionalConnectivity(data_dir=data_dir, fs=1.2, cutoff=0.01, mapping=bridge.getRoiMapping())

# Get AnatomicalConnectivity object
AC = anatomical_connectivity.AnatomicalConnectivity(data_dir=data_dir, neuprint_client=neuprint_client, mapping=bridge.getRoiMapping())

plot_colors = plt.get_cmap('tab10')(np.arange(8)/8)
# %% symmetry of connectivity
fh, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].plot(AC.getConnectivityMatrix('CellCount').to_numpy().copy()[AC.upper_inds],
           AC.getConnectivityMatrix('CellCount').to_numpy().copy().T[AC.upper_inds], 'ko')
ax[0].set_xlabel('Cell count, A->B')
ax[0].set_ylabel('Cell count, B->A')

ax[1].plot(AC.getConnectivityMatrix('WeightedSynapseCount').to_numpy().copy()[AC.upper_inds],
           AC.getConnectivityMatrix('WeightedSynapseCount').to_numpy().copy().T[AC.upper_inds], 'ko')
ax[1].set_xlabel('Weighted synapses, A->B')
ax[1].set_ylabel('Weighted synapses, B->A')

# %%
fh, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(AC.getConnectivityMatrix('CellCount').to_numpy().copy()[AC.upper_inds],
           AC.getConnectivityMatrix('CommonInputFraction').to_numpy().copy().T[AC.upper_inds], 'ko')
ax[0].set_xlabel('Direct connectivity (cells)')
ax[0].set_ylabel('Common input fraction')

ax[1].plot(AC.getConnectivityMatrix('CommonInputFraction'),
           FC.CorrelationMatrix, 'ko')

ax[1].set_xlabel('Common input fraction')
ax[1].set_ylabel('Functional correlation')

# %% ~Lognormal distribtution of connection strengths
ConnectivityCount = AC.getConnectivityMatrix('CellCount')

pull_regions = ['AL(R)', 'LH(R)']
pull_inds = [np.where(np.array(FC.rois) == x)[0][0] for x in pull_regions]

fig1_0, ax = plt.subplots(2, 1, figsize=(6, 6))
ax = ax.ravel()
fig1_0.tight_layout(w_pad=2, h_pad=8)

figS1, axS1 = plt.subplots(4, 9, figsize=(18,6))
axS1 = axS1.ravel()

z_scored_data = []
for p_ind, pr in enumerate(AC.getConnectivityMatrix('CellCount').index):
    outbound = AC.getConnectivityMatrix('CellCount').loc[pr,:]
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
        eg_ind = np.where(pr==np.array(pull_regions))[0][0]
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

        ax[eg_ind].annotate('Source: {}'.format(pr), (12, mod_mean[0]), fontsize=14)

fig1_0.text(-0.01, 0.6, 'Connecting cells', va='center', rotation='vertical', fontsize=14)
figS1.text(-0.01, 0.5, 'Connections from source region (cells)', va='center', rotation='vertical', fontsize=14)

frac_inside_shading = np.sum(np.abs(np.hstack(z_scored_data)) <=2) / np.hstack(z_scored_data).size

p_vals = []
for arr in z_scored_data:
    _, p = kstest(arr, 'norm')
    p_vals.append(p)

print(p_vals)
fig1_1, ax = plt.subplots(2, 1, figsize=(3.5, 6))

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
ticks = [1e-2, 1, 1e2]
ax[0].set_xticks(ticks)
ax[0].set_xlabel('Cell count (z-scored)')
ax[0].set_ylabel('Probability')
ax[0].set_xscale('log')

# Q-Q plot of log-transformed data vs. fit normal distribution

ax[1].plot([10**-4, 10**4], [10**-4, 10**4], 'k--')
quants = np.linspace(0, 1, 20)
for q in quants:
    th_pts = np.quantile(theory_distr, q, axis=1) # quantile value for each iteration
    ax[1].plot([10**np.quantile(data, q), 10**np.quantile(data, q)], [10**(np.mean(th_pts) - 2*np.std(th_pts)), 10**(np.mean(th_pts) + 2*np.std(th_pts))], color=plot_colors[0])
    ax[1].plot(10**np.quantile(data, q), 10**np.mean(th_pts), marker='o', color=plot_colors[0])
ax[1].set_xlabel('Data quantile')
ax[1].set_ylabel('Lognormal quantile')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ticks = [1e-4, 1e-2, 1, 1e2, 1e4]
ax[1].set_xticks(ticks)
ax[1].set_yticks(ticks)

# %% Eg region traces and cross corrs
pull_regions = ['AL(R)', 'CAN(R)', 'LH(R)', 'SPS(R)']
pull_inds = [np.where(np.array(FC.rois) == x)[0][0] for x in pull_regions]

ind = 11 # eg brain ind
resp_fp = FC.response_filepaths[ind]
brain_fn = 'func_volreg_2018-11-03_5.nii.gz'
brain_fp = os.path.join(os.path.split(resp_fp)[0], brain_fn)
suffix = brain_fp.split('func_volreg_')[-1]
atlas_fp = os.path.join(os.path.split(resp_fp)[0], 'vfb_68_' + suffix)

# load eg meanbrain and region masks
meanbrain = FC.getMeanBrain(brain_fp)
all_masks, _ = FC.loadAtlasData(atlas_fp)
masks = list(np.array(all_masks)[pull_inds])

cmap = plt.get_cmap('Set2')
colors = cmap(np.arange(len(pull_regions))/len(pull_regions))

zslices = [12, 45]
fig1_2 = plt.figure(figsize=(2,4))
for z_ind, z in enumerate(zslices):
    ax = fig1_2.add_subplot(2, 1, z_ind+1)

    overlay = plotting.overlayImage(meanbrain, masks, 0.5, colors=colors, z=z)

    img = ax.imshow(np.swapaxes(overlay, 0, 1), rasterized=False)
    ax.set_axis_off()
    ax.set_aspect('equal')

# # TODO: put this df/f processing stuff in functional_connectivity
fs = 1.2 # Hz
cutoff = 0.01

x_start = 200
dt = 300 #datapts
timevec = np.arange(0, dt) / fs # sec

file_id = resp_fp.split('/')[-1].replace('.pkl', '')
region_response = pd.read_pickle(resp_fp)
# convert to dF/F
dff = (region_response.to_numpy() - np.mean(region_response.to_numpy(), axis=1)[:, None]) / np.mean(region_response.to_numpy(), axis=1)[:, None]

# trim and filter
resp = functional_connectivity.filterRegionResponse(dff, cutoff=cutoff, fs=fs)
resp = functional_connectivity.trimRegionResponse(file_id, resp)
region_dff = pd.DataFrame(data=resp, index=region_response.index)

fig1_3, ax = plt.subplots(4, 1, figsize=(9, 6))
fig1_3.tight_layout(pad=2)
ax = ax.ravel()
[x.set_axis_off() for x in ax]
[x.set_ylim([-0.2, 0.29]) for x in ax]
[x.set_xlim([-15, timevec[-1]]) for x in ax]
for p_ind, pr in enumerate(pull_regions):
    ax[p_ind].plot(timevec, region_dff.loc[pr, x_start:(x_start+dt-1)], color=colors[p_ind])
    ax[p_ind].annotate(pr, (-10, 0) , rotation=90)

plotting.addScaleBars(ax[0], dT=5, dF=0.10, T_value=-2.5, F_value=-0.10)

fig1_4, ax = plt.subplots(4, 4, figsize=(6, 6))
fig1_4.tight_layout(h_pad=4, w_pad=4)
[x.set_xticks([]) for x in ax.ravel()]
[x.set_yticks([]) for x in ax.ravel()]
for ind_1, eg1 in enumerate(pull_regions):
    for ind_2, eg2 in enumerate(pull_regions):
        if ind_1 > ind_2:

            r, p = pearsonr(region_dff.loc[eg1, :], region_dff.loc[eg2, :])
            print('{}/{}: r = {}'.format(eg2, eg1, r))

            # normed xcorr plot
            window_size = 180
            total_len = len(region_dff.loc[eg1, :])

            a = (region_dff.loc[eg1, :] - np.mean(region_dff.loc[eg1, :])) / (np.std(region_dff.loc[eg1, :]) * len(region_dff.loc[eg1, :]))
            b = (region_dff.loc[eg2, :] - np.mean(region_dff.loc[eg2, :])) / (np.std(region_dff.loc[eg2, :]))
            c = np.correlate(a, b, 'same')
            time = np.arange(-window_size/2, window_size/2) / fs # sec
            ax[ind_1, ind_2].plot(time, c[int(total_len/2-window_size/2): int(total_len/2+window_size/2)], 'k')
            ax[ind_1, ind_2].set_ylim([-0.2, 1])
            ax[ind_1, ind_2].axhline(0, color='k', alpha=0.5, linestyle='-')
            ax[ind_1, ind_2].axvline(0, color='k', alpha=0.5, linestyle='-')
            if ind_2==0:
                ax[ind_1, ind_2].set_ylabel(eg1)
            if ind_1==3:
                ax[ind_1, ind_2].set_xlabel(eg2)

plotting.addScaleBars(ax[3, 0], dT=30, dF=0.25, T_value=time[0], F_value=-0.15)
sns.despine(top=True, right=True, left=True, bottom=True)


# %%

fig2_0, ax = plt.subplots(1, 2, figsize=(10, 5))
df = AC.getConnectivityMatrix('CellCount', diag=np.nan)
sns.heatmap(np.log10(AC.getConnectivityMatrix('CellCount', diag=np.nan)).replace([np.inf, -np.inf], 0), ax=ax[0], yticklabels=True, xticklabels=True, cmap="cividis", rasterized=True, cbar=False)
cb = fig2_0.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.SymLogNorm(vmin=1, vmax=np.nanmax(df.to_numpy()), base=10, linthresh=0.1, linscale=1), cmap="cividis"), ax=ax[0], shrink=0.75, label='Connecting cells')
cb.outline.set_linewidth(0)
ax[0].set_xlabel('Target');
ax[0].set_ylabel('Source');
ax[0].set_aspect('equal')
ax[0].tick_params(axis='both', which='major', labelsize=8)
sns.heatmap(FC.CorrelationMatrix, ax=ax[1], yticklabels=True, xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .75}, cmap="cividis", rasterized=True)
ax[1].set_aspect('equal')
ax[1].tick_params(axis='both', which='major', labelsize=8)

# Make adjacency matrices
# Log transform anatomical connectivity
anatomical_adjacency, keep_inds = AC.getAdjacency('CellCount', do_log=True)
functional_adjacency = FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]

r, p = pearsonr(anatomical_adjacency, functional_adjacency)
coef = np.polyfit(anatomical_adjacency, functional_adjacency, 1)
linfit = np.poly1d(coef)

fig2_1, ax = plt.subplots(1,1,figsize=(3.5, 3.5))
ax.plot(10**anatomical_adjacency, functional_adjacency, color='k', marker='o', linestyle='none')
xx = np.linspace(anatomical_adjacency.min(), anatomical_adjacency.max(), 100)
ax.plot(10**xx, linfit(xx), color='k', linewidth=2, marker=None)
ax.set_xscale('log')
ax.set_xlabel('Anatomical adjacency (cells)')
ax.set_ylabel('Functional correlation (z)')
ax.annotate('r = {:.2f}'.format(r), xy=(1, 1.0));

r_vals = []
for c_ind in range(FC.cmats.shape[2]):
    cmat = FC.cmats[:, :, c_ind]
    functional_adjacency_new = cmat[FC.upper_inds][keep_inds]

    r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_new)
    r_vals.append(r_new)

fig2_2, ax = plt.subplots(1,1,figsize=(1.75, 3.15))
fig2_2.tight_layout(pad=4)
sns.stripplot(x=np.ones_like(r_vals), y=r_vals, color='k')
sns.violinplot(y=r_vals)
ax.set_ylabel('Structure-function corr. (z)')
ax.set_xticks([])
ax.set_ylim([0, 1]);

# %%
thresh = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6]

r_vals = []
for th in thresh:
    anatomical_adjacency, keep_inds = AC.getAdjacency('CellCount', do_log=True, thresh=th)
    functional_adjacency = FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]

    r, p = pearsonr(anatomical_adjacency, functional_adjacency)
    r_vals.append(r)

print(r_vals)
# %%

roilabels_to_show = ['BU(R)', 'AVLP(R)', 'MBML(R)', 'PVLP(R)', 'AL(R)', 'LH(R)', 'EB', 'PLP(R)', 'AOTU(R)']

anat_position = {}
for r in range(len(FC.coms)):
    anat_position[r] = FC.coms[r, :]

adjacency_anat = AC.getConnectivityMatrix('CellCount', symmetrize=True, diag=0).to_numpy()
adjacency_fxn = FC.CorrelationMatrix.to_numpy().copy()
np.fill_diagonal(adjacency_fxn, 0)

# significance test on fxnal cmat
num_comparisons = len(FC.upper_inds[0])
p_cutoff = 0.01 / num_comparisons # bonferroni
t, p = ttest_1samp(FC.cmats, 0, axis=2) # ttest against 0
np.fill_diagonal(p, 1) # replace nans in diag with p=1
adjacency_fxn[p>p_cutoff] = 0 # set nonsig regions to 0
print('Ttest included {} significant of {} total edges in fxnal connectivity matrix'.format((p<p_cutoff).sum(), p.size))

# Plot clustering and degree using full adjacency to make graphs
G_anat = nx.from_numpy_matrix(adjacency_anat)
G_fxn = nx.from_numpy_matrix(adjacency_fxn)

nx.get_edge_attributes(G_anat,'weight')

fig3_0, ax = plt.subplots(1, 2, figsize=(8, 4))
deg_fxn = np.array([val for (node, val) in G_fxn.degree(weight='weight')])
deg_anat = np.array([val for (node, val) in G_anat.degree(weight='weight')])
plotting.addLinearFit(ax[0], deg_anat, deg_fxn, alpha=0.5)
ax[0].plot(deg_anat, deg_fxn, alpha=1.0, marker='o', linestyle='none')
for r_ind, r in enumerate(FC.rois):
    if r in roilabels_to_show:
        ax[0].annotate(r, (deg_anat[r_ind]+500, deg_fxn[r_ind]-0.2), fontsize=8, fontweight='bold')

ax[0].set_xlabel('Structural')
ax[0].set_ylabel('Functional')

clust_fxn = np.array(list(nx.clustering(G_fxn, weight='weight').values()))
clust_anat = np.array(list(nx.clustering(G_anat, weight='weight').values()))
plotting.addLinearFit(ax[1], clust_anat, clust_fxn, alpha=0.5)
ax[1].plot(clust_anat, clust_fxn, alpha=1.0, marker='o', linestyle='none')
for r_ind, r in enumerate(FC.rois):
    if r in roilabels_to_show:
        ax[1].annotate(r, (clust_anat[r_ind]+0.002, clust_fxn[r_ind]-0.003), fontsize=8, fontweight='bold')
ax[1].set_xlabel('Structural')
ax[1].set_ylabel('Functional')


# %%

# Illustration schematic: node degree

G = nx.Graph()
G.add_edge(1, 2, weight=1)
G.add_edge(1, 3, weight=1)
G.add_edge(3, 2, weight=10)

fig3_1, ax = plt.subplots(1, 3, figsize=(4, 1.25))
[x.set_xlim([-1, 1]) for x in ax.ravel()]
[x.set_ylim([-1, 1]) for x in ax.ravel()]
# graph 1: low degree
position = nx.circular_layout(G, scale=0.75)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
deg = [val for (node, val) in G.degree(weight='weight')]
nx.draw(G, ax=ax[0], pos=position, width=np.array(weights)/3, node_color=['r', 'k', 'k'], node_size=75)
nx.draw_networkx_edge_labels(G, ax=ax[0], pos=position, edge_labels=nx.get_edge_attributes(G,'weight'))
ax[0].annotate(deg[0], position[1] + [-0.0, 0.2], fontsize=12, weight='bold')
# graph 2: mod degree
G.add_edge(1, 2, weight=0)
G.add_edge(1, 3, weight=10)
G.add_edge(3, 2, weight=10)
position = nx.circular_layout(G, scale=0.75)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
deg = [val for (node, val) in G.degree(weight='weight')]
nx.draw(G, ax=ax[1], pos=position, width=np.array(weights)/3, node_color=['r', 'k', 'k'], node_size=75)
nx.draw_networkx_edge_labels(G, ax=ax[1], pos=position, edge_labels=nx.get_edge_attributes(G,'weight'))
ax[1].annotate(deg[0], position[1] + [-0.0, 0.2], fontsize=12, weight='bold')

# graph 3: high degree
G.add_edge(1, 2, weight=10)
G.add_edge(1, 3, weight=10)
G.add_edge(3, 2, weight=1)
position = nx.circular_layout(G, scale=0.75)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
deg = [val for (node, val) in G.degree(weight='weight')]
nx.draw(G, ax=ax[2], pos=position, width=np.array(weights)/3, node_color=['r', 'k', 'k'], node_size=75)
nx.draw_networkx_edge_labels(G, ax=ax[2], pos=position, edge_labels=nx.get_edge_attributes(G,'weight'))
ax[2].annotate(deg[0], position[1] + [-0.0, 0.2], fontsize=12, weight='bold');
# Illustration schematic: clustering coefficient
high = 10
low = 1

G = nx.Graph()
G.add_edge(1, 2, weight=high)
G.add_edge(1, 3, weight=high)
G.add_edge(3, 2, weight=0)
G.add_edge(3, 4, weight=0)
G.add_edge(2, 4, weight=0)
G.add_edge(1, 4, weight=high)


fig3_2, ax = plt.subplots(1, 3, figsize=(4, 1.25))
[x.set_xlim([-1, 1]) for x in ax.ravel()]
[x.set_ylim([-1, 1]) for x in ax.ravel()]
# graph 1: low clustering
position = nx.circular_layout(G, scale=0.75)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
clust = list(nx.clustering(G, weight='weight').values())
nx.draw(G, ax=ax[0], pos=position, width=np.array(weights)/3, node_color=['r', 'k', 'k', 'k'], node_size=75)
ax[0].annotate('{:.2f}'.format(clust[0]), position[1] + [-0.0, 0.2], fontsize=12, weight='bold')

# graph 2: mod clustering
G = nx.Graph()
G.add_edge(1, 2, weight=high)
G.add_edge(1, 3, weight=high)
G.add_edge(3, 2, weight=low)
G.add_edge(3, 4, weight=low)
G.add_edge(2, 4, weight=low)
G.add_edge(1, 4, weight=high)
position = nx.circular_layout(G, scale=0.75)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
clust = list(nx.clustering(G, weight='weight').values())
nx.draw(G, ax=ax[1], pos=position, width=np.array(weights)/3, node_color=['r', 'k', 'k', 'k'], node_size=75)
ax[1].annotate('{:.2f}'.format(clust[0]), position[1] + [-0.0, 0.2], fontsize=12, weight='bold')
high
# graph 3: high clustering
G = nx.Graph()
G.add_edge(1, 2, weight=high)
G.add_edge(1, 3, weight=high)
G.add_edge(3, 2, weight=high)
G.add_edge(3, 4, weight=high)
G.add_edge(2, 4, weight=high)
G.add_edge(1, 4, weight=high)
position = nx.circular_layout(G, scale=0.75)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
clust = list(nx.clustering(G, weight='weight').values())
nx.draw(G, ax=ax[2], pos=position, width=np.array(weights)/3, node_color=['r', 'k', 'k', 'k'], node_size=75)
ax[2].annotate('{:.2f}'.format(clust[0]), position[1] + [-0.0, 0.2], fontsize=12, weight='bold');
# %%
# # # # # plot network graph with top x% of connections
take_top_pct = 0.2 # top fraction to include in network graphs
roilabels_to_skip = ['ATL(R)', 'IB', 'MPED(R)', 'SIP(R)', 'PLP(R)', 'SPS(R)', 'GOR(R)', 'GOR(L)', 'ICL(R)','BU(L)', 'BU(R)', 'SCL(R)']

cmap = plt.get_cmap('Blues')

cutoff = np.quantile(adjacency_anat, 1-take_top_pct)
print('Threshold included {} of {} edges in anatomical connectivity matrix'.format((adjacency_anat>=cutoff).sum(), adjacency_anat.size))
temp_adj_anat = adjacency_anat.copy()
temp_adj_anat[temp_adj_anat<cutoff] = 0
G_anat = nx.from_numpy_matrix(temp_adj_anat/temp_adj_anat.max())

cutoff = np.quantile(adjacency_fxn[adjacency_fxn>0], 1-take_top_pct)
print('Threshold included {} of {} sig edges in functional connectivity matrix'.format((adjacency_fxn>=cutoff).sum(), (adjacency_fxn>0).sum()))
temp_adj_fxn = adjacency_fxn.copy()
temp_adj_fxn[temp_adj_fxn<cutoff] = 0
G_fxn = nx.from_numpy_matrix(temp_adj_fxn/temp_adj_fxn.max())

fig3_3 = plt.figure(figsize=(12,6))
ax_anat = fig3_3.add_subplot(1, 2, 1, projection='3d')
ax_fxn = fig3_3.add_subplot(1, 2, 2, projection='3d')

ax_anat.view_init(-70, -95)
ax_anat.set_axis_off()
# ax_anat.set_title('Structural', fontweight='bold', fontsize=12)

ax_fxn.view_init(-70, -95)
ax_fxn.set_axis_off()
# ax_fxn.set_title('Functional', fontweight='bold', fontsize=12)

for key, value in anat_position.items():
    xi = value[0]
    yi = value[1]
    zi = value[2]

    # Plot nodes
    ax_anat.scatter(xi, yi, zi, c='b', s=5+40*G_anat.degree(weight='weight')[key], edgecolors='k', alpha=0.25)
    ax_fxn.scatter(xi, yi, zi, c='b', s=5+20*G_fxn.degree(weight='weight')[key], edgecolors='k', alpha=0.25)
    if FC.rois[key] not in roilabels_to_skip:
        ax_anat.text(xi, yi, zi+2, FC.rois[key], zdir=(0,0,0), fontsize=8, fontweight='bold')
        ax_fxn.text(xi, yi, zi+2, FC.rois[key], zdir=(0,0,0), fontsize=8, fontweight='bold')

# ctr = [5, 80, 60]
# dstep = 10
# ax_anat.plot([ctr[0], ctr[0]+dstep], [ctr[1], ctr[1]], [ctr[2], ctr[2]], 'r') # x
# ax_anat.plot([ctr[0], ctr[0]], [ctr[1], ctr[1]-dstep], [ctr[2], ctr[2]], 'g') # y
# ax_anat.plot([ctr[0], ctr[0]], [ctr[1], ctr[1]], [ctr[2], ctr[2]-dstep], 'b') # z
#
# ax_fxn.plot([ctr[0], ctr[0]+dstep], [ctr[1], ctr[1]], [ctr[2], ctr[2]], 'r') # x
# ax_fxn.plot([ctr[0], ctr[0]], [ctr[1], ctr[1]-dstep], [ctr[2], ctr[2]], 'g') # y
# ax_fxn.plot([ctr[0], ctr[0]], [ctr[1], ctr[1]], [ctr[2], ctr[2]-dstep], 'b') # z


# plot connections
for i,j in enumerate(G_anat.edges()):
    x = np.array((anat_position[j[0]][0], anat_position[j[1]][0]))
    y = np.array((anat_position[j[0]][1], anat_position[j[1]][1]))
    z = np.array((anat_position[j[0]][2], anat_position[j[1]][2]))

    # Plot the connecting lines
    line_wt = (G_anat.get_edge_data(j[0], j[1], default={'weight':0})['weight'] + G_anat.get_edge_data(j[1], j[0], default={'weight':0})['weight'])/2
    color = cmap(line_wt)
    ax_anat.plot(x, y, z, c=color, alpha=line_wt, linewidth=2)

    line_wt = (G_fxn.get_edge_data(j[0], j[1], default={'weight':0})['weight'] + G_fxn.get_edge_data(j[1], j[0], default={'weight':0})['weight'])/2
    color = cmap(line_wt)
    ax_fxn.plot(x, y, z, c=color, alpha=line_wt, linewidth=2)


# %% Dominance analysis
cell_ct, _ = AC.getAdjacency('CellCount')
synapse_count, _ = AC.getAdjacency('WeightedSynapseCount')
commoninput, _ = AC.getAdjacency('CommonInputFraction')
path_length = AC.getShortestPathLength('CellCount').to_numpy()[AC.upper_inds]

completeness = (AC.CompletenessMatrix.to_numpy() + AC.CompletenessMatrix.to_numpy().T) / 2

# X = np.vstack([connectivity,
#                commoninput[keep_inds],
#                AC.getShortestPathLength('WeightedSynapseCount').to_numpy()[AC.upper_inds][keep_inds],
#                FC.SizeMatrix.to_numpy()[FC.upper_inds][keep_inds],
#                FC.DistanceMatrix.to_numpy()[FC.upper_inds][keep_inds],
#                AC.CompletenessMatrix.to_numpy()[AC.upper_inds][keep_inds],
#                FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]]).T


# X = np.vstack([cell_ct,
#                synapse_count,
#                commoninput,
#                path_length,
#                FC.SizeMatrix.to_numpy()[FC.upper_inds],
#                FC.DistanceMatrix.to_numpy()[FC.upper_inds],
#                AC.CompletenessMatrix.to_numpy()[AC.upper_inds],
#                FC.CorrelationMatrix.to_numpy()[FC.upper_inds]]).T

X = np.vstack([cell_ct,
               synapse_count,
               commoninput,
               path_length,
               FC.SizeMatrix.to_numpy()[FC.upper_inds],
               FC.DistanceMatrix.to_numpy()[FC.upper_inds],
               completeness[AC.upper_inds],
               FC.CorrelationMatrix.to_numpy()[FC.upper_inds]]).T

fig4_1, ax = plt.subplots(1, 1, figsize=(2, 2.2))
# linear regression model prediction:
regressor = LinearRegression()
regressor.fit(X[:, :-1], X[:, -1]);
pred = regressor.predict(X[:, :-1])
score = regressor.score(X[:, :-1], X[:, -1])
ax.plot(pred, X[:, -1], 'k.')
ax.plot([-0.2, 1.1], [-0.2, 1.1], 'k--')
ax.set_title('$r^2$={:.2f}'.format(score));
ax.set_xlabel('Predicted', fontsize=10)
ax.set_ylabel('Measured', fontsize=10)
ax.set_xticks([0, 1.0])
ax.set_yticks([0, 1.0])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

r, p = pearsonr(pred, X[:, -1])


fc_df = pd.DataFrame(data=X, columns=['Cell count', 'Synapse count', 'Common Input', 'Path length', 'ROI size', 'ROI Distance', 'Completeness', 'fc'])
dominance_regression=Dominance(data=fc_df,target='fc',objective=1)

incr_variable_rsquare=dominance_regression.incremental_rsquare()
keys = np.array(list(incr_variable_rsquare.keys()))
vals = np.array(list(incr_variable_rsquare.values()))
s_inds = np.argsort(vals)[::-1]

fig4_2, ax = plt.subplots(1, 1, figsize=(4.75, 3.5))
sns.barplot(x=[x.replace(' ','\n') for x in keys[s_inds]], y=vals[s_inds], ax=ax, color=plot_colors[0])
ax.set_ylabel('Incremental $r^2$')
ax.tick_params(axis='both', which='major', labelsize=8)

 #%%
SP = AC.getShortestPathLength('WeightedSynapseCount')

x = SP.to_numpy()[AC.upper_inds]
y = FC.CorrelationMatrix.to_numpy()[FC.upper_inds]

r, p = spearmanr(x, y)
fh, ax = plt.subplots(1, 1, figsize=(6,4))
ax.plot(x, y, 'ko')
ax.set_title('$\\rho$ = {:.2f}'.format(r));
ax.set_xlabel('Shortest path distance')
ax.set_ylabel('Functional connectivity (z)')




# %%
import collections

anat_connect = AC.getConnectivityMatrix('CellCount', diag=None).to_numpy()
G_anat = nx.from_numpy_matrix(anat_connect)

for e in G_anat.edges:
    G_anat.edges[e]['distance'] = 1/G_anat.edges[e]['weight']


inter_nodes = []
for row in range(anat_connect.shape[0]):
    for col in range(anat_connect.shape[1]):
        step_nodes = list(nx.algorithms.shortest_path(G_anat, source=row, target=col, weight='distance'))
        if len(step_nodes) > 2:
            inter_nodes.append(step_nodes)
            # %%
len(inter_nodes)
inter_nodes


hub_dict = collections.Counter(np.hstack(inter_nodes))

for k in hub_dict:
    print('{}: {}; {}'.format(AC.rois[k] , hub_dict[k] , deg_anat[k]))



# %% Difference matrix

# # compute difference matrix using original, asymmetric anatomical connectivity matrix
anatomical_mat = AC.getConnectivityMatrix('CellCount', diag=0).to_numpy().copy()
functional_mat = FC.CorrelationMatrix.to_numpy().copy()
np.fill_diagonal(functional_mat, 0)

# log transform anatomical connectivity values
keep_inds_diff = np.where(anatomical_mat > 0)
functional_adjacency_diff = functional_mat[keep_inds_diff]
anatomical_adjacency_diff = np.log10(anatomical_mat[keep_inds_diff])

F_zscore = zscore(functional_adjacency_diff)
A_zscore = zscore(anatomical_adjacency_diff)
diff = A_zscore - F_zscore


diff_m = np.zeros_like(anatomical_mat)
diff_m[keep_inds_diff] = diff
DifferenceMatrix = pd.DataFrame(data=diff_m, index=FC.rois, columns=FC.rois)


# %% sort difference matrix by most to least different rois
diff_by_region = DifferenceMatrix.mean()
sort_inds = np.argsort(diff_by_region)
sort_keys = DifferenceMatrix.index[sort_inds]
sorted_diff = pd.DataFrame(data=np.zeros_like(DifferenceMatrix),columns=sort_keys, index=sort_keys)
for r_ind, r_key in enumerate(sort_keys):
    for c_ind, c_key in enumerate(sort_keys):
        sorted_diff.iloc[r_ind, c_ind]=DifferenceMatrix.loc[[r_key], [c_key]].to_numpy()

fig5_0, ax = plt.subplots(1, 1, figsize=(4, 4))
lim = np.nanmax(np.abs(DifferenceMatrix.to_numpy().ravel()))
ax.scatter(A_zscore, F_zscore, alpha=1, c=diff, cmap="RdBu",  vmin=-lim, vmax=lim, edgecolors='k', linewidths=0.5)
ax.plot([-3, 4], [-3, 4], 'k-')
ax.set_xlabel('Anatomical ajacency (z-score)')
ax.set_ylabel('Functional correlation (z-score)');
# ax.set_xticks([-2, 0, 3])
# ax.set_yticks([-3, 0, 3])

fig5_1, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.heatmap(sorted_diff, ax=ax, yticklabels=True, xticklabels=True, cbar_kws={'label': 'Difference (SC - FC)','shrink': .75}, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
ax.set_aspect('equal')
ax.tick_params(axis='both', which='major', labelsize=7)

diff_by_region = DifferenceMatrix.mean()
diff_brain = np.zeros(shape=FC.roi_mask[0].shape)
diff_brain[:] = np.nan
for r_ind, r in enumerate(FC.roi_mask):
    diff_brain[r] = diff_by_region[r_ind]


zslices = np.linspace(5, 60, 8)
lim = np.nanmax(np.abs(diff_brain.ravel()))

fig5_2 = plt.figure(figsize=(8, 4))
for z_ind, z in enumerate(zslices):
    ax = fig5_2.add_subplot(2, 4, z_ind+1)
    img = ax.imshow(diff_brain[:, :, int(z)].T, cmap="RdBu", rasterized=False, vmin=-lim, vmax=lim)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_xlim([0, 102])
    ax.set_ylim([107, 5])

fig5_3, ax = plt.subplots(1, 1, figsize=(1, 3))
ax.set_axis_off()
cb = fig5_3.colorbar(img, ax=ax)
cb.set_label(label='Region-average diff.', weight='bold', color='k')
cb.ax.tick_params(labelsize=12, color='k')

# %% subsampled region cmats and SC-FC corr
anatomical_adjacency, keep_inds = AC.getAdjacency('CellCount', do_log=True)

bins = np.arange(np.floor(np.min(FC.roi_size)), np.ceil(np.max(FC.roi_size)))
values, base = np.histogram(FC.roi_size, bins=bins, density=True)
cumulative = np.cumsum(values)

# Load precomputed subsampled Cmats for each brain
load_fn = os.path.join(data_dir, 'functional_connectivity', 'subsampled_cmats_20200626.npy')
(cmats_pop, CorrelationMatrix_Full, subsampled_sizes) = np.load(load_fn, allow_pickle=True)

# mean cmat over brains for each subsampledsize and iteration
cmats_popmean = np.mean(cmats_pop, axis=4) # roi x roi x iterations x sizes
scfc_r = np.zeros(shape=(cmats_popmean.shape[2], cmats_popmean.shape[3])) # iterations x sizes
for s_ind, sz in enumerate(subsampled_sizes):
    for it in range(cmats_popmean.shape[2]):
        functional_adjacency_tmp = cmats_popmean[:, :, it, s_ind][FC.upper_inds][keep_inds]
        new_r, _ = pearsonr(anatomical_adjacency, functional_adjacency_tmp)
        scfc_r[it, s_ind] = new_r

# plot mean+/-SEM results on top of region size cumulative histogram
err_y = np.std(scfc_r, axis=0)
mean_y = np.mean(scfc_r, axis=0)

figS2, ax1 = plt.subplots(1, 1, figsize=(5,5))
ax1.plot(subsampled_sizes, mean_y, 'ko')
ax1.errorbar(subsampled_sizes, mean_y, yerr=err_y, color='k')
ax1.hlines(mean_y[-1], subsampled_sizes.min(), subsampled_sizes.max(), color='k', linestyle='--')
ax1.set_xlabel('Region size (voxels)')
ax1.set_ylabel('Correlation with anatomical connectivity')
ax1.set_xscale('log')
ax2 = ax1.twinx()
ax2.plot(base[:-1], cumulative)
ax2.set_ylabel('Cumulative fraction')
ax2.set_ylim([0, 1.05])

# %%
figs_to_save = [fig1_0, fig1_1, fig1_2, fig1_3, fig1_4,
                fig2_0, fig2_1, fig2_2,
                fig3_0, fig3_1, fig3_2, fig3_3,
                fig4_1, fig4_2,
                fig5_0, fig5_1, fig5_2, fig5_3,
                figS1, figS2]
for f_ind, fh in enumerate(figs_to_save):
    fh.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig{}.svg'.format(f_ind)), format='svg')

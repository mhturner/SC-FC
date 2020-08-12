from neuprint import Client
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, ttest_1samp
from scipy.stats import zscore
from scipy.stats import kstest, lognorm, norm
from scipy.signal import correlate
from dominance_analysis import Dominance
import networkx as nx
from matplotlib import rcParams
rcParams['svg.fonttype'] = 'none'
rcParams.update({'figure.autolayout': True})

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

# %% ~Lognormal distribtution of connection strengthsto_numpy().copy().T[AC.upper_inds]
ConnectivityCount = AC.getConnectivityMatrix('CellCount')

pull_regions = ['AL(R)', 'CAN(R)', 'LH(R)', 'SPS(R)']

fig1_0, ax = plt.subplots(int(len(pull_regions)/2), 2, figsize=(12,6))
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

    axS1[p_ind].plot(ct, 'bo')
    axS1[p_ind].fill_between(list(range(len(mod_mean))), err_up, err_down, color='k', alpha=0.4)
    axS1[p_ind].plot(mod_mean, 'k--')
    axS1[p_ind].set_xticks([])
    axS1[p_ind].annotate('{}'.format(pr), (12, 6e3), fontsize=8)
    axS1[p_ind].set_yscale('log')
    axS1[p_ind].set_ylim([0.2, 5e4])

    if pr in pull_regions:
        eg_ind = np.where(pr==np.array(pull_regions))[0][0]
        ax[eg_ind].plot(ct, 'bo')
        ax[eg_ind].fill_between(list(range(len(mod_mean))), err_up, err_down, color='k', alpha=0.4)
        ax[eg_ind].plot(mod_mean, 'k--')

        ax[eg_ind].set_xticks(list(range(len(ct))))
        ax[eg_ind].set_xticklabels(ct.index)
        ax[eg_ind].set_yscale('log')
        for tick in ax[eg_ind].get_xticklabels():
            tick.set_rotation(90)

        ax[eg_ind].annotate('Source: {}'.format(pr), (12, mod_mean[0]))

fig1_0.text(-0.02, 0.5, 'Outgoing connections (Cell count)', va='center', rotation='vertical', fontsize=14)
figS1.text(-0.02, 0.5, 'Outgoing connections (Cell count)', va='center', rotation='vertical', fontsize=14)

frac_inside_shading = np.sum(np.abs(np.hstack(z_scored_data)) <=2) / np.hstack(z_scored_data).size

p_vals = []
for arr in z_scored_data:
    _, p = kstest(arr, 'norm')
    p_vals.append(p)

print(p_vals)
fig1_1, ax = plt.subplots(1, 2, figsize=(6, 3))

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
ax[0].plot(10**bin_ctrs, val, linewidth=3)
xx = np.linspace(-3.5, 3.5)
ax[0].plot(10**xx, norm_model.pdf(xx), linewidth=2, color='k', linestyle='--')
ax[0].set_xscale('log')
ax[0].set_xlabel('Cell count (z-scored)')
ax[0].set_ylabel('Probability')
ax[0].set_xscale('log')

# Q-Q plot of log-transformed data vs. fit normal distribution
quants = np.linspace(0, 1, 20)
for q in quants:
    th_pts = np.quantile(theory_distr, q, axis=1) # quantile value for each iteration
    ax[1].plot([10**np.quantile(data, q), 10**np.quantile(data, q)], [10**(np.mean(th_pts) - 2*np.std(th_pts)), 10**(np.mean(th_pts) + 2*np.std(th_pts))], 'k-')
    ax[1].plot(10**np.quantile(data, q), 10**np.mean(th_pts), 'ko')
ax[1].plot([10**-4, 10**4], [10**-4, 10**4], 'k--')
ax[1].set_xlabel('Data quantile')
ax[1].set_ylabel('Lognormal quantile')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ticks = [1e-4, 1e-2, 1, 1e2, 1e4]
ax[1].set_xticks(ticks)
ax[1].set_yticks(ticks)
ax[1].set_aspect('equal')

# %% Eg region traces and cross corrs
cmap = plt.get_cmap('Set3')
colors = cmap(np.arange(len(pull_regions))/len(pull_regions))
map_colors = np.tile([0.5, 0.5, 0.5, 0.2], (len(FC.rois), 1))
pull_inds = [np.where(np.array(FC.rois) == x)[0][0] for x in pull_regions]
map_colors[pull_inds, :] = colors
region_map = FC.getRegionMap(colors=map_colors)

zslices = [9, 21, 45]
fig1_2 = plt.figure(figsize=(2,6))
for z_ind, z in enumerate(zslices):
    ax = fig1_2.add_subplot(3, 1, z_ind+1)
    img = ax.imshow(np.swapaxes(region_map[:, :, z, :], 0, 1), rasterized=False)
    ax.set_axis_off()
    ax.set_aspect('equal')

# # TODO: put this df/f processing stuff in functional_connectivity
ind = 11
fs = 1.2 # Hz
cutoff = 0.01

x_start = 200
dt = 300 #datapts
timevec = np.arange(0, dt) / fs # sec

resp_fp = FC.response_filepaths[ind]

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


fig2_0, ax = plt.subplots(1, 2, figsize=(14, 7))
df = np.log10(AC.getConnectivityMatrix('CellCount', diag=np.nan)).replace([np.inf, -np.inf], 0)
sns.heatmap(df, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Connection strength (log10(Connecting cells))', 'shrink': .8}, cmap="cividis", rasterized=True)
ax[0].set_xlabel('Target');
ax[0].set_ylabel('Source');
ax[0].set_aspect('equal')

sns.heatmap(FC.CorrelationMatrix, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True)
ax[1].set_aspect('equal')


# Make adjacency matrices
# Log transform anatomical connectivity
anatomical_adjacency, keep_inds = AC.getAdjacency('CellCount', do_log=True)
functional_adjacency = FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]

r, p = pearsonr(anatomical_adjacency, functional_adjacency)
coef = np.polyfit(anatomical_adjacency, functional_adjacency, 1)
linfit = np.poly1d(coef)

fig2_1, ax = plt.subplots(1,1,figsize=(4.5, 4.5))
ax.scatter(10**anatomical_adjacency, functional_adjacency, color='k')
xx = np.linspace(anatomical_adjacency.min(), anatomical_adjacency.max(), 100)
ax.plot(10**xx, linfit(xx), color='k', linewidth=2, marker=None)
ax.set_xscale('log')
ax.set_xlabel('Anatomical adjacency (Connecting cells)')
ax.set_ylabel('Functional correlation (z)')
ax.annotate('r = {:.2f}'.format(r), xy=(1, 1.0));

r_vals = []
for c_ind in range(FC.cmats.shape[2]):
    cmat = FC.cmats[:, :, c_ind]
    functional_adjacency_new = cmat[FC.upper_inds][keep_inds]

    r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_new)
    r_vals.append(r_new)

fig2_2, ax = plt.subplots(1,1,figsize=(2,3))
fig2_2.tight_layout(pad=4)

sns.stripplot(x=np.ones_like(r_vals), y=r_vals, color='k')
sns.violinplot(y=r_vals)
ax.set_ylabel('Correlation coefficient (z)')
ax.set_xticks([])
ax.set_ylim([0, 1]);

# %%

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
ax[0].scatter(deg_anat, deg_fxn, color=[0.5, 0.5, 0.5, 0.5])
# for r_ind, r in enumerate(FC.rois):
#     ax[0].annotate(r, (deg_anat[r_ind]-0.1, deg_fxn[r_ind]), fontsize=8)
plotting.addLinearFit(ax[0], deg_anat, deg_fxn)
ax[0].set_xlabel('Structural')
ax[0].set_ylabel('Functional')

clust_fxn = np.array(list(nx.clustering(G_fxn, weight='weight').values()))
clust_anat = np.array(list(nx.clustering(G_anat, weight='weight').values()))
ax[1].scatter(clust_anat, clust_fxn, color=[0.5, 0.5, 0.5, 0.5])
# for r_ind, r in enumerate(FC.rois):
#     ax[1].annotate(r, (clust_anat[r_ind], clust_fxn[r_ind]), fontsize=8)
plotting.addLinearFit(ax[1], clust_anat, clust_fxn)
ax[1].set_xlabel('Structural')
ax[1].set_ylabel('Functional')


# %%

# Illustration schematic: node degree

G = nx.Graph()
G.add_edge(1, 2, weight=1)
G.add_edge(1, 3, weight=1)
G.add_edge(3, 2, weight=10)

fig3_1, ax = plt.subplots(1, 3, figsize=(9, 3))
[x.set_xlim([-1, 1]) for x in ax.ravel()]
[x.set_ylim([-1, 1]) for x in ax.ravel()]
# graph 1: low degree
position = nx.circular_layout(G, scale=0.75)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
deg = [val for (node, val) in G.degree(weight='weight')]
nx.draw(G, ax=ax[0], pos=position, width=np.array(weights)/2, node_color=['r', 'k', 'k'])
nx.draw_networkx_edge_labels(G, ax=ax[0], pos=position, edge_labels=nx.get_edge_attributes(G,'weight'))
ax[0].annotate(deg[0], position[1] + [-0.07, 0.12], fontsize=18, weight='bold')
# graph 2: mod degree
G.add_edge(1, 2, weight=0)
G.add_edge(1, 3, weight=10)
G.add_edge(3, 2, weight=10)
position = nx.circular_layout(G, scale=0.75)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
deg = [val for (node, val) in G.degree(weight='weight')]
nx.draw(G, ax=ax[1], pos=position, width=np.array(weights)/2, node_color=['r', 'k', 'k'])
nx.draw_networkx_edge_labels(G, ax=ax[1], pos=position, edge_labels=nx.get_edge_attributes(G,'weight'))
ax[1].annotate(deg[0], position[1] + [-0.07, 0.12], fontsize=18, weight='bold')

# graph 3: high degree
G.add_edge(1, 2, weight=10)
G.add_edge(1, 3, weight=10)
G.add_edge(3, 2, weight=1)
position = nx.circular_layout(G, scale=0.75)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
deg = [val for (node, val) in G.degree(weight='weight')]
nx.draw(G, ax=ax[2], pos=position, width=np.array(weights)/2, node_color=['r', 'k', 'k'])
nx.draw_networkx_edge_labels(G, ax=ax[2], pos=position, edge_labels=nx.get_edge_attributes(G,'weight'))
ax[2].annotate(deg[0], position[1] + [-0.07, 0.12], fontsize=18, weight='bold');
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


fig3_2, ax = plt.subplots(1, 3, figsize=(9, 3))
[x.set_xlim([-1, 1]) for x in ax.ravel()]
[x.set_ylim([-1, 1]) for x in ax.ravel()]
# graph 1: low clustering
position = nx.circular_layout(G, scale=0.75)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
clust = list(nx.clustering(G, weight='weight').values())
nx.draw(G, ax=ax[0], pos=position, width=np.array(weights)/2, node_color=['r', 'k', 'k', 'k'])
ax[0].annotate('{:.2f}'.format(clust[0]), position[1] + [-0.07, 0.12], fontsize=16, weight='bold')

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
nx.draw(G, ax=ax[1], pos=position, width=np.array(weights)/2, node_color=['r', 'k', 'k', 'k'])
ax[1].annotate('{:.2f}'.format(clust[0]), position[1] + [-0.07, 0.12], fontsize=16, weight='bold')
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
nx.draw(G, ax=ax[2], pos=position, width=np.array(weights)/2, node_color=['r', 'k', 'k', 'k'])
ax[2].annotate('{:.2f}'.format(clust[0]), position[1] + [-0.07, 0.12], fontsize=16, weight='bold');
# %%
# # # # # plot network graph with top x% of connections
take_top_pct = 0.2 # top fraction to include in network graphs
roilabels_to_skip = ['LAL(R)', 'CRE(R)', 'CRE(L)', 'EPA(R)','BU(R)']
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

ax_anat.view_init(-145, -95)
ax_anat.set_axis_off()
# ax_anat.set_title('Structural', fontweight='bold', fontsize=12)

ax_fxn.view_init(-145, -95)
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

    ctr = [15, 70, 60]
    dstep=10
    ax_anat.plot([ctr[0], ctr[0]+dstep], [ctr[1], ctr[1]], [ctr[2], ctr[2]], 'r') # x
    ax_anat.plot([ctr[0], ctr[0]], [ctr[1], ctr[1]-dstep], [ctr[2], ctr[2]], 'g') # y
    ax_anat.plot([ctr[0], ctr[0]], [ctr[1], ctr[1]], [ctr[2], ctr[2]-dstep], 'b') # z

    ax_fxn.plot([ctr[0], ctr[0]+dstep], [ctr[1], ctr[1]], [ctr[2], ctr[2]], 'r') # x
    ax_fxn.plot([ctr[0], ctr[0]], [ctr[1], ctr[1]-dstep], [ctr[2], ctr[2]], 'g') # y
    ax_fxn.plot([ctr[0], ctr[0]], [ctr[1], ctr[1]], [ctr[2], ctr[2]-dstep], 'b') # z


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
connectivity, keep_inds = AC.getAdjacency('CellCount', do_log=True)
commoninput, _ = AC.getAdjacency('CommonInputFraction')


X = np.vstack([connectivity,
               commoninput[keep_inds],
               FC.SizeMatrix.to_numpy()[FC.upper_inds][keep_inds],
               FC.DistanceMatrix.to_numpy()[FC.upper_inds][keep_inds],
               AC.CompletenessMatrix.to_numpy()[AC.upper_inds][keep_inds],
               FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]]).T

fig4_1, ax = plt.subplots(1, 2, figsize=(8, 4))
# linear regression model prediction:
regressor = LinearRegression()
regressor.fit(X[:, :-1], X[:, -1]);
pred = regressor.predict(X[:, :-1])
score = regressor.score(X[:, :-1], X[:, -1])
ax[0].plot(pred, X[:, -1], 'ko')
ax[0].plot([-0.2, 1.25], [-0.2, 1.25], 'k--')
ax[0].annotate('$r^2$={:.2f}'.format(score), (0, 1));
ax[0].set_xlabel('Predicted functional conectivity (z)')
ax[0].set_ylabel('Measured functional conectivity (z)');

fc_df = pd.DataFrame(data=X, columns=['Connectivity', 'Common Input', 'ROI size', 'ROI Distance', 'Completeness', 'fc'])
dominance_regression=Dominance(data=fc_df,target='fc',objective=1)

incr_variable_rsquare=dominance_regression.incremental_rsquare()
keys = np.array(list(incr_variable_rsquare.keys()))
vals = np.array(list(incr_variable_rsquare.values()))
s_inds = np.argsort(vals)[::-1]

sns.barplot(x=keys[s_inds], y=vals[s_inds], ax=ax[1], color=colors[0])
ax[1].set_ylabel('Incremental $r^2$')
for tick in ax[1].get_xticklabels():
    tick.set_rotation(90)

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

fig5_0, ax = plt.subplots(1, 1, figsize=(6,6))
lim = np.nanmax(np.abs(DifferenceMatrix.to_numpy().ravel()))
ax.scatter(A_zscore, F_zscore, alpha=1, c=diff, cmap="RdBu",  vmin=-lim, vmax=lim, edgecolors='k', linewidths=0.5)
ax.plot([-3, 4], [-3, 4], 'k-')
ax.set_xlabel('Anatomical connectivity (log10, zscore)')
ax.set_ylabel('Functional correlation (zscore)');

fig5_1, ax = plt.subplots(1, 1, figsize=(8,8))
sns.heatmap(sorted_diff, ax=ax, xticklabels=True, cbar_kws={'label': 'Anat - Fxnal connectivity','shrink': .75}, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
ax.set_aspect('equal')

diff_by_region = DifferenceMatrix.mean()
diff_brain = np.zeros(shape=FC.roi_mask[0].shape)
diff_brain[:] = np.nan
for r_ind, r in enumerate(FC.roi_mask):
    diff_brain[r] = diff_by_region[r_ind]

zslices = np.arange(5, 65, 12)
lim = np.nanmax(np.abs(diff_brain.ravel()))

fig5_2 = plt.figure(figsize=(15,3))
for z_ind, z in enumerate(zslices):
    ax = fig5_2.add_subplot(1, 5, z_ind+1)
    img = ax.imshow(diff_brain[:, :, z].T, cmap="RdBu", rasterized=False, vmin=-lim, vmax=lim)
    ax.set_axis_off()
    ax.set_aspect('equal')

cb = fig5_2.colorbar(img, ax=ax)
cb.set_label(label='Anat - Fxnal connectivity', weight='bold', color='k')
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
                fig4_1,
                fig5_0, fig5_1, fig5_2,
                figS1, figS2]
for f_ind, fh in enumerate(figs_to_save):
    fh.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig{}.svg'.format(f_ind)))

from neuprint import Client
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr, ttest_1samp
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from scipy.ndimage.measurements import center_of_mass
from scipy.stats import kstest, lognorm, norm, shapiro
from scipy.signal import correlate
from dominance_analysis import Dominance
from visanalysis import plot_tools
import networkx as nx

from matplotlib import rcParams
rcParams['svg.fonttype'] = 'none'
rcParams.update({'figure.autolayout': True})

from region_connectivity import RegionConnectivity

"""
References:
https://connectome-neuprint.github.io/neuprint-python/docs/index.html
https://github.com/connectome-neuprint/neuprint-python

"""

data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC'

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1heHdlbGxob2x0ZXR1cm5lckBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpMHJRX0M4akliX0ZrS2h2OU5DSElsWlpnRDY5YUMtVGdNLWVWM3lRP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzY2MTk1MzcwfQ.Q-57D4tX2sXMjWym2LFhHaUGHgHiUsIM_JI9xekxw_0')
mapping = RegionConnectivity.getRoiMapping()
rois = list(mapping.keys())
rois.sort()

roi_completeness = RegionConnectivity.getRoiCompleteness(neuprint_client, mapping)
CompletenessMatrix = pd.DataFrame(data=np.outer(roi_completeness['frac_post'], roi_completeness['frac_pre']), index=roi_completeness.index, columns=roi_completeness.index)

# %% LOAD FUNCTIONAL DATA, FILTER IT ACCORDING TO MAPPING, COMPUTE SOME GEOMETRY STUFF
"""
Functional connectivity and atlas data
    :CorrelationMatrix_Functional: Avg across animals, fischer z transformed correlation values
    :DistanceMatrix: distance between centers of mass for each pair of ROIs
    :SizeMatrix: geometric mean of the sizes for each pair of ROIs
"""
roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')
response_filepaths = glob.glob(os.path.join(data_dir, 'region_responses') + '/' + '*.pkl')
fs = 1.2 # Hz
cutoff = 0.01 # Hz

CorrelationMatrix_Functional, cmats = RegionConnectivity.getFunctionalConnectivity(response_filepaths, cutoff=cutoff, fs=fs)
roi_mask, roi_size = RegionConnectivity.loadAtlasData(atlas_path=atlas_path, roinames_path=roinames_path, mapping=mapping)

# indices for connectivity and correlation matrices
upper_inds = np.triu_indices(CorrelationMatrix_Functional.shape[0], k=1) # k=1 excludes main diagonal
lower_inds = np.tril_indices(CorrelationMatrix_Functional.shape[0], k=1) # k=1 excludes main diagonal

# find center of mass for each roi
coms = np.vstack([center_of_mass(x) for x in roi_mask])

# calulcate euclidean distance matrix between roi centers of mass
dist_mat = np.zeros_like(CorrelationMatrix_Functional)
dist_mat[upper_inds] = pdist(coms)
dist_mat += dist_mat.T # symmetrize to fill in below diagonal

DistanceMatrix = pd.DataFrame(data=dist_mat, index=CorrelationMatrix_Functional.index, columns=CorrelationMatrix_Functional.index)

# geometric mean of the sizes for each pair of ROIs
sz_mat = np.sqrt(np.outer(np.array(roi_size), np.array(roi_size)))

SizeMatrix = pd.DataFrame(data=sz_mat, index=CorrelationMatrix_Functional.index, columns=CorrelationMatrix_Functional.index)

# %% LOAD ANATOMICAL DATA AND MAKE ADJACENCY MATRICES

"""
Anatomical connectivity matrices and symmetrized versions of each
    :ConnectivityCount: Total number of cells with any inputs in source and outputs in target
    :ConnectivityWeight: sqrt(input PSDs in source x output tbars in target)
    :WeightedSynapseCount: output tbars (presynapses) in target * (input (post)synapses in source)/(total (post)synapses onto that cell)
    :CommonInputFraction: each fraction of total input cells to [row] that also project to region in [col]

        _Symmetrized: symmetrize each adjacency matrix by adding it to its
                      transpose and dividing by 2. Ignores directionality
"""

# 1) ConnectivityCount
WeakConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'WeakConnections_computed_20200806.pkl'))
MediumConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'MediumConnections_computed_20200806.pkl'))
StrongConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'StrongConnections_computed_20200806.pkl'))
conn_mat = WeakConnections + MediumConnections + StrongConnections
# set diag to nan
tmp_mat = conn_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)
ConnectivityCount_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=conn_mat.index, columns=conn_mat.index)
ConnectivityCount = pd.DataFrame(data=tmp_mat, index=conn_mat.index, columns=conn_mat.index)
# - - - - - - - - - - - - - - - - #
# 2) ConnectivityWeight
weight_mat = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'Connectivity_computed_20200806.pkl'))
# set diag to nannan
tmp_mat = weight_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)
ConnectivityWeight_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=weight_mat.index, columns=weight_mat.index)
ConnectivityWeight = pd.DataFrame(data=tmp_mat, index=weight_mat.index, columns=weight_mat.index)

# - - - - - - - - - - - - - - - - #
# 3) WeightedSynapseCount
syn_mat = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'WeightedSynapseNumber_computed_20200806.pkl'))
# set diag to nan
tmp_mat = syn_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)
WeightedSynapseCount_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=syn_mat.index, columns=syn_mat.index)
WeightedSynapseCount = pd.DataFrame(data=tmp_mat, index=syn_mat.index, columns=syn_mat.index)

# - - - - - - - - - - - - - - - - #
# 4) CommonInputFraction
CommonInputFraction = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'CommonInputFraction_computed_20200806.pkl'))
# set diag to nan
tmp_mat = CommonInputFraction.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)
CommonInputFraction_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=CommonInputFraction.index, columns=CommonInputFraction.index)
CommonInputFraction = pd.DataFrame(data=tmp_mat, index=CommonInputFraction.index, columns=CommonInputFraction.index)

# compute two-step count connectivity matrix
A = ConnectivityCount.to_numpy().copy()
two_steps = np.zeros_like(A)
for source in range(ConnectivityCount.shape[0]):
    for target in range(ConnectivityCount.shape[1]):
        if source != target:
            conns = [np.sqrt(A[source, x] * A[x, target]) for x in range(ConnectivityCount.shape[0]) if x not in (source, target)]
            two_steps[source, target] = np.nansum(conns)

TwoStep_Symmetrized = pd.DataFrame(data=(two_steps + two_steps.T)/2, index=ConnectivityCount.index, columns=ConnectivityCount.index)
TwoStep = pd.DataFrame(data=two_steps, index=ConnectivityCount.index, columns=ConnectivityCount.index)

# indices for log transforming and comparing to log-transformed mats
keep_inds = np.where(ConnectivityCount_Symmetrized.to_numpy()[upper_inds] > 0) # for log-transforming anatomical connectivity, toss zero values

#
# # Make adjacency matrices
# # Log transform anatomical connectivity
# anatomical_adjacency = np.log10(ConnectivityCount_Symmetrized.to_numpy().copy()[upper_inds][keep_inds])
# functional_adjacency = CorrelationMatrix_Functional.to_numpy().copy()[upper_inds][keep_inds]

# %% corr between weighted synapse count and cell count:
FigS3, ax = plt.subplots(1, 1, figsize=(5,5))
ax.plot(ConnectivityCount.to_numpy(), WeightedSynapseCount.to_numpy(), 'ko')
ax.set_xlabel('Cell count')
ax.set_ylabel('Weighted synapse count');

# %% corr between cell count and common input fraction
FigS3, ax = plt.subplots(1, 1, figsize=(5,5))
ax.plot(ConnectivityCount.to_numpy(), CommonInputFraction.to_numpy(), 'ko')
ax.set_xlabel('Direct connecting cells')
ax.set_ylabel('Common input');


# %% corr common input and functional connectivity


FigS4, ax = plt.subplots(1, 1, figsize=(5,5))
ax.plot(CommonInputFraction.to_numpy()[upper_inds], CorrelationMatrix_Functional.to_numpy()[upper_inds], 'ko')
ax.set_xlabel('Common input fraction');
ax.set_ylabel('Functional corr (z)')
r, p = pearsonr(CommonInputFraction.to_numpy()[upper_inds], CorrelationMatrix_Functional.to_numpy()[upper_inds])

# %%

fh, ax = plt.subplots(1, 3, figsize=(21, 7))

df = np.log10(ConnectivityCount).replace([np.inf, -np.inf], 0)
sns.heatmap(ConnectivityCount, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Connection strength (log10(Connecting cells))', 'shrink': .8}, cmap="cividis", rasterized=True)
ax[0].set_xlabel('Target');
ax[0].set_ylabel('Source');
ax[0].set_aspect('equal')

sns.heatmap(CommonInputFraction, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Common input fraction','shrink': .8}, cmap="cividis", rasterized=True)
ax[1].set_aspect('equal')

sns.heatmap(CorrelationMatrix_Functional, ax=ax[2], xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True)
ax[2].set_aspect('equal')

# %% ~Lognormal distribtution of connection strengths

pull_regions = ['AL(R)', 'CAN(R)', 'LH(R)', 'SPS(R)']

fig1_0, ax = plt.subplots(int(len(pull_regions)/2), 2, figsize=(12,6))
ax = ax.ravel()
fig1_0.tight_layout(w_pad=2, h_pad=8)


figS1, axS1 = plt.subplots(4, 9, figsize=(18,6))
axS1 = axS1.ravel()

z_scored_data = []
for p_ind, pr in enumerate(ConnectivityCount.index):
    outbound = ConnectivityCount.loc[pr,:]
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

# %%
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

colors

x, y, z = roi_mask[0].shape
region_map = np.zeros(shape=(x, y, z, 4))
region_map[:] = 0
for roi in roi_mask:
    gray = [0.5, 0.5, 0.5, 0.2]
    region_map[roi, :] = gray

for p, pr in enumerate(pull_regions):
    pull_ind = np.where(ConnectivityCount.index == pr)[0][0]
    new_mask = roi_mask[pull_ind]
    new_color = colors[p]
    region_map[new_mask, :] = new_color

zslices = [9, 21, 45]
# zslices = np.arange(5, 75, 4)
# fig1_2 = plt.figure(figsize=(15,12))
fig1_2 = plt.figure(figsize=(2,6))
for z_ind, z in enumerate(zslices):
    # ax = fig1_2.add_subplot(4, 5, z_ind+1)
    ax = fig1_2.add_subplot(3, 1, z_ind+1)
    img = ax.imshow(np.swapaxes(region_map[:, :, z, :], 0, 1), rasterized=False)
    ax.set_axis_off()
    ax.set_aspect('equal')
    # ax.set_title(z)


ind = 11
fs = 1.2 # Hz
cutoff = 0.01

x_start = 200
dt = 300 #datapts
timevec = np.arange(0, dt) / fs # sec

resp_fp = response_filepaths[ind]

file_id = resp_fp.split('/')[-1].replace('.pkl', '')
region_response = pd.read_pickle(resp_fp)
# convert to dF/F
dff = (region_response.to_numpy() - np.mean(region_response.to_numpy(), axis=1)[:, None]) / np.mean(region_response.to_numpy(), axis=1)[:, None]

# trim and filter
resp = RegionConnectivity.filterRegionResponse(dff, cutoff=cutoff, fs=fs)
resp = RegionConnectivity.trimRegionResponse(file_id, resp)
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

plot_tools.addScaleBars(ax[0], dT=5, dF=0.10, T_value=-2.5, F_value=-0.10)

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
            ax[ind_1, ind_2].axhline(0, color='k', alpha=0.5, LineStyle='-')
            ax[ind_1, ind_2].axvline(0, color='k', alpha=0.5, LineStyle='-')
            if ind_2==0:
                ax[ind_1, ind_2].set_ylabel(eg1)
            if ind_1==3:
                ax[ind_1, ind_2].set_xlabel(eg2)

plot_tools.addScaleBars(ax[3, 0], dT=30, dF=0.25, T_value=time[0], F_value=-0.15)
sns.despine(top=True, right=True, left=True, bottom=True)


# %%


fig2_0, ax = plt.subplots(1, 2, figsize=(14, 7))

df = np.log10(ConnectivityCount).replace([np.inf, -np.inf], 0)
sns.heatmap(df, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Connection strength (log10(Connecting cells))', 'shrink': .8}, cmap="cividis", rasterized=True)
ax[0].set_xlabel('Target');
ax[0].set_ylabel('Source');
ax[0].set_aspect('equal')

sns.heatmap(CorrelationMatrix_Functional, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True)
ax[1].set_aspect('equal')

r, p = pearsonr(anatomical_adjacency, functional_adjacency)
coef = np.polyfit(anatomical_adjacency, functional_adjacency, 1)
linfit = np.poly1d(coef)

fig2_1, ax = plt.subplots(1,1,figsize=(4.5, 4.5))
ax.scatter(10**anatomical_adjacency, functional_adjacency, color='k')
xx = np.linspace(anatomical_adjacency.min(), anatomical_adjacency.max(), 100)
ax.plot(10**xx, linfit(xx), color='k', LineWidth=2, marker=None)
ax.set_xscale('log')
ax.set_xlabel('Anatomical adjacency (Connecting cells)')
ax.set_ylabel('Functional correlation (z)')
ax.annotate('r = {:.2f}'.format(r), xy=(1, 1.0));

r_vals = []
for c_ind in range(cmats.shape[2]):
    cmat = cmats[:, :, c_ind]
    functional_adjacency_new = cmat[upper_inds][keep_inds]

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
for r in range(len(coms)):
    anat_position[r] = coms[r, :]

# # # # STRUCTURAL ADJACENCY MATRIX # # # #
adjacency_anat = ConnectivityCount_Symmetrized.to_numpy().copy()
np.fill_diagonal(adjacency_anat, 0)

# # # # FUNCTIONAL ADJACENCY MATRIX  # # # #
adjacency_fxn = CorrelationMatrix_Functional.to_numpy().copy()
np.fill_diagonal(adjacency_fxn, 0)

# significance test on fxnal cmat
num_comparisons = len(upper_inds[0])
p_cutoff = 0.01 / num_comparisons # bonferroni
t, p = ttest_1samp(cmats, 0, axis=2) # ttest against 0
np.fill_diagonal(p, 1) # replace nans in diag with p=1
adjacency_fxn[p>p_cutoff] = 0 # set nonsig regions to 0
print('Ttest included {} significant of {} total regions in fxnal connectivity matrix'.format((p<p_cutoff).sum(), p.size))

# Plot clustering and degree using full adjacency to make graphs
G_anat = nx.from_numpy_matrix(adjacency_anat/adjacency_anat.max())
G_fxn = nx.from_numpy_matrix(adjacency_fxn/adjacency_fxn.max())

fig3_0, ax = plt.subplots(1, 2, figsize=(8, 4))
clust_fxn = list(nx.clustering(G_fxn, weight='weight').values())
clust_anat = list(nx.clustering(G_anat, weight='weight').values())
r, p = spearmanr(clust_anat, clust_fxn)
ax[0].set_title('Clustering, $\\rho$ = {:.3f}'.format(r))
ax[0].plot(clust_anat, clust_fxn, 'ko')
ax[0].set_xlabel('Structural')
ax[0].set_ylabel('Functional')

deg_fxn = [val for (node, val) in G_fxn.degree(weight='weight')]
deg_anat = [val for (node, val) in G_anat.degree(weight='weight')]
r, p = spearmanr(deg_anat, deg_fxn)
ax[1].set_title('Degree, $\\rho$ = {:.3f}'.format(r))
ax[1].plot(deg_anat, deg_fxn, 'ko')
ax[1].set_xlabel('Structural')
ax[1].set_ylabel('Functional')

# # # # # plot network graph with top x% of connections
take_top_pct = 0.2 # top fraction to include in network graphs
roilabels_to_skip = ['LAL(R)', 'CRE(R)', 'CRE(L)', 'EPA(R)','BU(R)']
cmap = plt.get_cmap('Blues')

cutoff = np.quantile(adjacency_anat, 1-take_top_pct)
print('Threshold included {} of {} regions in anatomical connectivity matrix'.format((adjacency_anat>=cutoff).sum(), adjacency_anat.size))
temp_adj_anat = adjacency_anat.copy()
temp_adj_anat[temp_adj_anat<cutoff] = 0
G_anat = nx.from_numpy_matrix(temp_adj_anat/temp_adj_anat.max())

cutoff = np.quantile(adjacency_fxn[adjacency_fxn>0], 1-take_top_pct)
print('Threshold included {} of {} sig regions in functional connectivity matrix'.format((adjacency_fxn>=cutoff).sum(), (adjacency_fxn>0).sum()))
temp_adj_fxn = adjacency_fxn.copy()
temp_adj_fxn[temp_adj_fxn<cutoff] = 0
G_fxn = nx.from_numpy_matrix(temp_adj_fxn/temp_adj_fxn.max())

fig3_1 = plt.figure(figsize=(12,6))
ax_anat = fig3_1.add_subplot(1, 2, 1, projection='3d')
ax_fxn = fig3_1.add_subplot(1, 2, 2, projection='3d')

ax_anat.view_init(-145, -95)
ax_anat.set_axis_off()
ax_anat.set_title('Structural', fontweight='bold', fontsize=12)

ax_fxn.view_init(-145, -95)
ax_fxn.set_axis_off()
ax_fxn.set_title('Functional', fontweight='bold', fontsize=12)

for key, value in anat_position.items():
    xi = value[0]
    yi = value[1]
    zi = value[2]

    # Plot nodes
    ax_anat.scatter(xi, yi, zi, c='b', s=5+40*G_anat.degree(weight='weight')[key], edgecolors='k', alpha=0.25)
    ax_fxn.scatter(xi, yi, zi, c='b', s=5+20*G_fxn.degree(weight='weight')[key], edgecolors='k', alpha=0.25)
    if rois[key] not in roilabels_to_skip:
        ax_anat.text(xi, yi, zi+2, rois[key], zdir=(0,0,0), fontsize=8, fontweight='bold')
        ax_fxn.text(xi, yi, zi+2, rois[key], zdir=(0,0,0), fontsize=8, fontweight='bold')

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


# %% Other determinants of FC
# Corrs with size, distance etc

connectivity = np.log10(ConnectivityCount_Symmetrized.to_numpy()[upper_inds][keep_inds])
commoninput = CommonInputFraction.to_numpy()[upper_inds][keep_inds]
size = SizeMatrix.to_numpy()[upper_inds][keep_inds]
dist = DistanceMatrix.to_numpy()[upper_inds][keep_inds]
completeness = CompletenessMatrix.to_numpy()[upper_inds][keep_inds]

fc = CorrelationMatrix_Functional.to_numpy()[upper_inds][keep_inds]

fig4_0, ax = plt.subplots(3, 2, figsize=(8,12))
r, p = pearsonr(size, fc)
ax[0, 0].scatter(size, fc, color='k', alpha=0.5)
coef = np.polyfit(size, fc, 1)
linfit = np.poly1d(coef)
xx = np.linspace(size.min(), size.max(), 100)
ax[0, 0].plot(xx, linfit(xx), 'k-', LineWidth=3)
ax[0, 0].set_title('{:.3f}'.format(r))
ax[0, 0].set_ylabel('Functional correlation (z)');
ax[0, 0].set_xlabel('ROI pair size');

r, p = pearsonr(dist, fc)
ax[0, 1].scatter(dist, fc, color='k', alpha=0.5)
coef = np.polyfit(dist, fc, 1)
linfit = np.poly1d(coef)
xx = np.linspace(dist.min(), dist.max(), 100)
ax[0, 1].plot(xx, linfit(xx), 'k-', LineWidth=3)
ax[0, 1].set_title('{:.3f}'.format(r))
ax[0, 1].set_xlabel('Inter-ROI distance');



r, p = pearsonr(size, connectivity)
ax[1, 0].scatter(size, 10**connectivity, color='k', alpha=0.5)
coef = np.polyfit(size, connectivity, 1)
linfit = np.poly1d(coef)
xx = np.linspace(size.min(), size.max(), 100)
ax[1, 0].plot(xx, 10**linfit(xx), 'k-', LineWidth=3)
ax[1, 0].set_title('{:.3f}'.format(r))
ax[1, 0].set_ylabel('Anatomical adjacency (connecting cells)');
ax[1, 0].set_xlabel('ROI pair size');
ax[1, 0].set_yscale('log')

r, p = pearsonr(dist, connectivity)
ax[1, 1].scatter(dist, 10**connectivity, color='k', alpha=0.5)
coef = np.polyfit(dist, connectivity, 1)
linfit = np.poly1d(coef)
xx = np.linspace(dist.min(), dist.max(), 100)
ax[1, 1].plot(xx, 10**linfit(xx), 'k-', LineWidth=3)
ax[1, 1].set_title('{:.3f}'.format(r))
ax[1, 1].set_xlabel('Inter-ROI distance');
ax[1, 1].set_yscale('log')

r, p = pearsonr(size, commoninput)
ax[2, 0].scatter(size, commoninput, color='k', alpha=0.5)
coef = np.polyfit(size, commoninput, 1)
linfit = np.poly1d(coef)
xx = np.linspace(size.min(), size.max(), 100)
ax[2, 0].plot(xx, linfit(xx), 'k-', LineWidth=3)
ax[2, 0].set_title('{:.3f}'.format(r))
ax[2, 0].set_ylabel('Common input fraction');
ax[2, 0].set_xlabel('ROI pair size');

r, p = pearsonr(dist, commoninput)
ax[2, 1].scatter(dist, commoninput, color='k', alpha=0.5)
coef = np.polyfit(dist, commoninput, 1)
linfit = np.poly1d(coef)
xx = np.linspace(dist.min(), dist.max(), 100)
ax[2, 1].plot(xx, linfit(xx), 'k-', LineWidth=3)
ax[2, 1].set_title('{:.3f}'.format(r))
ax[2, 1].set_xlabel('Inter-ROI distance');

# %% Dominance analysis

fig4_1, ax = plt.subplots(1, 2, figsize=(8, 4))

X = np.vstack([connectivity, commoninput, size, dist, completeness, fc]).T


# linear regression model prediction:
regressor = LinearRegression()
regressor.fit(X[:, :-1], X[:, -1]);
pred = regressor.predict(X[:, :-1])
score = regressor.score(X[:, :-1], fc)
ax[0].plot(pred, fc, 'ko')
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
anatomical_mat = ConnectivityCount.to_numpy().copy()
np.fill_diagonal(anatomical_mat, 0)
functional_mat = CorrelationMatrix_Functional.to_numpy().copy()
np.fill_diagonal(functional_mat, 0)

# log transform anatomical connectivity values
keep_inds_diff = np.where(anatomical_mat > 0)
functional_adjacency_diff= functional_mat[keep_inds_diff]
anatomical_adjacency_diff = np.log10(anatomical_mat[keep_inds_diff])

F_zscore = zscore(functional_adjacency_diff)
A_zscore = zscore(anatomical_adjacency_diff)
diff = A_zscore - F_zscore


diff_m = np.zeros_like(ConnectivityCount.to_numpy())
diff_m[keep_inds_diff] = diff
DifferenceMatrix = pd.DataFrame(data=diff_m, index=ConnectivityCount.index, columns=ConnectivityCount.index)


# %% SUPP FIG: does completeness of reconstruction impact
#
#
comp_score = CompletenessMatrix.to_numpy().ravel()
diff_score = DifferenceMatrix.to_numpy().ravel()

include_inds = np.where(diff_score != 0)[0]
comp_score  = comp_score[include_inds]
diff_score = diff_score[include_inds]
anat_conn = ConnectivityCount.to_numpy().ravel()[include_inds]


r, p = pearsonr(comp_score, diff_score)

figS2, ax = plt.subplots(1, 2, figsize=(8,4))
ax[0].scatter(comp_score, diff_score, marker='o', color='k', alpha=0.5)

ax[0].set_xlabel('Completeness of reconstruction')
ax[0].set_ylabel('abs(Anat. - Fxnal (z-score))')

ax[1].scatter(comp_score, anat_conn, marker='o', color='k', alpha=0.5)

ax[1].set_xlabel('Completeness of reconstruction')
ax[1].set_ylabel('Anat connectivity')


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


# %%
diff_by_region = DifferenceMatrix.mean()
diff_brain = np.zeros(shape=roi_mask[0].shape)
diff_brain[:] = np.nan
for r_ind, r in enumerate(roi_mask):
    diff_brain[r] = diff_by_region[r_ind]

# %%

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

# Get region sizes from atlas data
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')
roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')
roi_mask, roi_size = RegionConnectivity.loadAtlasData(atlas_path, roinames_path, mapping=mapping)
bins = np.arange(np.floor(np.min(roi_size)), np.ceil(np.max(roi_size)))
values, base = np.histogram(roi_size, bins=bins, density=True)
cumulative = np.cumsum(values)

# Load precomputed subsampled Cmats for each brain
load_fn = os.path.join(data_dir, 'functional_connectivity', 'subsampled_cmats_20200626.npy')
(cmats_pop, CorrelationMatrix_Full, subsampled_sizes) = np.load(load_fn, allow_pickle=True)

# mean cmat over brains for each subsampledsize and iteration
cmats_popmean = np.mean(cmats_pop, axis=4) # roi x roi x iterations x sizes
scfc_r = np.zeros(shape=(cmats_popmean.shape[2], cmats_popmean.shape[3])) # iterations x sizes
for s_ind, sz in enumerate(subsampled_sizes):
    for it in range(cmats_popmean.shape[2]):
        functional_adjacency_tmp = cmats_popmean[:, :, it, s_ind][upper_inds][keep_inds]
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
                fig3_0, fig3_1,
                fig4_0, fig4_1,
                fig5_0, fig5_1, fig5_2,
                figS1, figS2]
for f_ind, fh in enumerate(figs_to_save):
    fh.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig{}.svg'.format(f_ind)))

from neuprint import Client
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, ARDRegression, BayesianRidge
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from scipy.ndimage.measurements import center_of_mass
from matplotlib.backends.backend_pdf import PdfPages
import datetime
from scipy.stats import kstest, lognorm, norm
from dominance_analysis import Dominance

from matplotlib import rcParams
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
    :ConnectivityCount_precomputed: from Janelia computation
    :ConnectivityWeight_precomputed: from Janelia computation

        _Symmetrized: symmetrize each adjacency matrix by adding it to its
                      transpose and dividing by 2. Ignores directionality
"""

# 1) ConnectivityCount
WeakConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'WeakConnections_computed_20200618.pkl'))
MediumConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'MediumConnections_computed_20200618.pkl'))
StrongConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'StrongConnections_computed_20200618.pkl'))
conn_mat = WeakConnections + MediumConnections + StrongConnections
# set diag to nan
tmp_mat = conn_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)
ConnectivityCount_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=conn_mat.index, columns=conn_mat.index)
ConnectivityCount = pd.DataFrame(data=tmp_mat, index=conn_mat.index, columns=conn_mat.index)
# - - - - - - - - - - - - - - - - #
# 2) ConnectivityWeight
weight_mat = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'Connectivity_computed_20200618.pkl'))
# set diag to nan
tmp_mat = weight_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)
ConnectivityWeight_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=weight_mat.index, columns=weight_mat.index)
ConnectivityWeight = pd.DataFrame(data=tmp_mat, index=weight_mat.index, columns=weight_mat.index)
# - - - - - - - - - - - - - - - - #
# 3) ConnectivityCount_precomputed
pccount_mat = RegionConnectivity.getPrecomputedConnectivityMatrix(neuprint_client, mapping, metric='count', diagonal='nan')
tmp_mat = pccount_mat.to_numpy().copy()
ConnectivityCount_precomputed_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=pccount_mat.index, columns=pccount_mat.index)
ConnectivityCount_precomputed = pd.DataFrame(data=tmp_mat, index=pccount_mat.index, columns=pccount_mat.index)
# - - - - - - - - - - - - - - - - #
# 4) ConnectivityWeight_precomputed
pcweight_mat = RegionConnectivity.getPrecomputedConnectivityMatrix(neuprint_client, mapping, metric='weight', diagonal='nan')
tmp_mat = pcweight_mat.to_numpy().copy()
ConnectivityWeight_precomputed_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=pcweight_mat.index, columns=pcweight_mat.index)
ConnectivityWeight_precomputed = pd.DataFrame(data=tmp_mat, index=pcweight_mat.index, columns=pcweight_mat.index)

# compute two-step count connectivity matrix
# TODO: remove AAC and ACC connections - check this
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

# Make adjacency matrices
# Log transform anatomical connectivity
anatomical_adjacency = np.log10(ConnectivityCount_Symmetrized.to_numpy().copy()[upper_inds][keep_inds])
functional_adjacency = CorrelationMatrix_Functional.to_numpy().copy()[upper_inds][keep_inds]

# %% Lognormal distribtution of connection strengths

# pull_regions = ['MBML(L)', 'MBML(R)', 'MBVL(R)', 'CRE(R)', 'CRE(L)', 'LAL(R)', 'AL(R)', 'FB', 'VES(R)']

pull_regions = ['AL(R)', 'CAN(R)', 'LH(R)', 'SPS(R)']

fig1_0, ax = plt.subplots(int(len(pull_regions)/2), 2, figsize=(12,6))
fig1_0.tight_layout(w_pad=2, h_pad=8)
ax = ax.ravel()

for p_ind, pr in enumerate(pull_regions):
    outbound = ConnectivityCount.loc[pull_regions[p_ind],:]
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
    err_down = 10**(np.mean(samples, axis=0) - np.std(samples, axis=0))
    err_up = 10**(np.mean(samples, axis=0) + np.std(samples, axis=0))

    ax[p_ind].plot(ct, 'bo')
    ax[p_ind].fill_between(list(range(len(mod_mean))), err_up, err_down, color='k', alpha=0.4)
    ax[p_ind].plot(mod_mean, 'k--')

    ax[p_ind].set_xticks(list(range(len(ct))))
    ax[p_ind].set_xticklabels(ct.index)
    ax[p_ind].set_yscale('log')
    for tick in ax[p_ind].get_xticklabels():
        tick.set_rotation(90)

    ax[p_ind].annotate('Source: {}'.format(pr), (12, mod_mean[0]))

fig1_0.text(-0.02, 0.5, 'Outgoing connections (Cell count)', va='center', rotation='vertical', fontsize=14)


fig1_1, ax = plt.subplots(1, 1, figsize=(5,4))
#  ConnectivityCount:
ct_mat = ConnectivityCount.to_numpy().copy()
np.fill_diagonal(ct_mat, 0) # replace diag nan with 0
counts = ct_mat.ravel()
keep_inds_count = np.where(counts > 0) # exclude 0 ct vals for log transform
ct = counts[keep_inds_count]
log_ct = np.log10(counts[keep_inds_count])

val, bin = np.histogram(log_ct, 20, density=True)
bin_ctrs = bin[:-1]
ax.plot(10**bin_ctrs, val, LineWidth=3)
xx = np.linspace(bin_ctrs.min(), bin_ctrs.max(), 100)
yy = norm(loc=np.mean(np.log10(ct)), scale=np.std(np.log10(ct))).pdf(xx)
ax.plot(10**xx, yy, 'k', alpha=1, LineWidth=3)

p_ct, _ = kstest(zscore(log_ct), 'norm')
ax.set_xlabel('Connectivity (Cell count)')
ax.set_ylabel('Probability')
ax.set_xscale('log')
print('KS test lognormal: Count p = {:.4f}'.format(p_ct))

# # ConnectivityWeight:
# wt_mat = ConnectivityWeight.to_numpy().copy()
# np.fill_diagonal(wt_mat, 0) # replace diag nan with 0
# weights = wt_mat.ravel()
# keep_inds_weight = np.where(weights > 0) # exclude 0 ct vals for log transform
# wt = weights[keep_inds_weight]
# log_wt = np.log10(weights[keep_inds_weight])
#
# val, bin = np.histogram(log_wt, 20, density=True)
# bin_ctrs = bin[:-1]
# ax[1].plot(10**bin_ctrs, val, LineWidth=2)
# xx = np.linspace(bin_ctrs.min(), bin_ctrs.max(), 100)
# yy = norm(loc=np.mean(np.log10(wt)), scale=np.std(np.log10(wt))).pdf(xx)
# ax[1].plot(10**xx, yy, 'k', alpha=1)
# p_wt, _ = kstest(zscore(log_wt), 'norm')
# ax[1].set_xlabel('Connection weight')
# ax[1].set_ylabel('Probability')
# ax[1].set_xscale('log')
# print('KS test lognormal: Weight p = {:.4f}'.format(p_wt))


# %% Eg region traces and corr scatter plots
cmap = plt.get_cmap('Set3')
colors = cmap(np.arange(len(pull_regions))/len(pull_regions))

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
fig1_2 = plt.figure(figsize=(3,9))
for z_ind, z in enumerate(zslices):
    # ax = fig1_2.add_subplot(4, 5, z_ind+1)
    ax = fig1_2.add_subplot(3, 1, z_ind+1)
    img = ax.imshow(np.swapaxes(region_map[:, :, z, :], 0, 1), rasterized=True)
    ax.set_axis_off()
    ax.set_aspect('equal')
    # ax.set_title(z)


ind = 7
fs = 1.2
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

fig1_3, ax = plt.subplots(4, 1, figsize=(12, 8))
fig1_3.tight_layout(pad=4)
ax = ax.ravel()
[x.set_axis_off() for x in ax]
[x.set_ylim([-0.2, 0.29]) for x in ax]
for p_ind, pr in enumerate(pull_regions):
    ax[p_ind].plot(timevec, region_dff.loc[pr, x_start:(x_start+dt-1)], color=colors[p_ind])
    ax[p_ind].annotate(pr, (-10, 0) , rotation=90)


fig1_4, ax = plt.subplots(1, 3, figsize=(15, 5))
fig1_4.tight_layout(h_pad=4, w_pad=4)
eg1 = 'AL(R)'
eg2 = 'LH(R)'
r, p = pearsonr(region_dff.loc[eg1, :], region_dff.loc[eg2, :])
ax[0].plot(region_dff.loc[eg1, :], region_dff.loc[eg2, :], 'ko')
ax[0].set_title('r={:.2f}'.format(r))
ax[0].set_xlabel('{} (dF/F)'.format(eg1))
ax[0].set_ylabel('{} (dF/F)'.format(eg2))

eg1 = 'AL(R)'
eg2 = 'CAN(R)'
r, p = pearsonr(region_dff.loc[eg1, :], region_dff.loc[eg2, :])
ax[1].plot(region_dff.loc[eg1, :], region_dff.loc[eg2, :], 'ko')
ax[1].set_title('r={:.2f}'.format(r))
ax[1].set_xlabel('{} (dF/F)'.format(eg1))
ax[1].set_ylabel('{} (dF/F)'.format(eg2))

eg1 = 'CAN(R)'
eg2 = 'SPS(R)'
r, p = pearsonr(region_dff.loc[eg1, :], region_dff.loc[eg2, :])
ax[2].plot(region_dff.loc[eg1, :], region_dff.loc[eg2, :], 'ko')
ax[2].set_title('r={:.2f}'.format(r))
ax[2].set_xlabel('{} (dF/F)'.format(eg1))
ax[2].set_ylabel('{} (dF/F)'.format(eg2))

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

fig2_1, ax = plt.subplots(1,1,figsize=(3.5, 3.5))
ax.scatter(10**anatomical_adjacency, functional_adjacency, color='k')
xx = np.linspace(anatomical_adjacency.min(), anatomical_adjacency.max(), 100)
ax.plot(10**xx, linfit(xx), 'k-')
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


fig2_2, ax = plt.subplots(1,1,figsize=(3,6))
fig2_2.tight_layout(pad=4)
sns.swarmplot(x=np.ones_like(r_vals), y=r_vals, color='k')
sns.violinplot(y=r_vals)
ax.set_ylabel('Correlation coefficient (z)')
ax.set_ylim([0, 1]);
# %% Other determinants of FC
# Corrs with size, distance etc

connectivity = np.log10(ConnectivityCount_Symmetrized.to_numpy()[upper_inds][keep_inds])
size = SizeMatrix.to_numpy()[upper_inds][keep_inds]
dist = DistanceMatrix.to_numpy()[upper_inds][keep_inds]
completeness = CompletenessMatrix.to_numpy()[upper_inds][keep_inds]

fc = CorrelationMatrix_Functional.to_numpy()[upper_inds][keep_inds]

X = np.vstack([connectivity, size, dist, completeness, fc]).T

fig3_0, ax = plt.subplots(2, 2, figsize=(8,8))
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

# %% Dominance analysis

fig3_1, ax = plt.subplots(1, 2, figsize=(8, 4))

# linear regression model prediction:
regressor = LinearRegression()
regressor.fit(X[:, :-1], X[:, -1]);
pred = regressor.predict(X[:, :-1])
score = regressor.score(X[:, :-1], fc)
ax[0].plot(pred, fc, 'ko')
ax[0].plot([0, 1.4], [0, 1.4], 'k--')
ax[0].annotate('$r^2$={:.2f}'.format(score), (0, 1));
ax[0].set_xlabel('Predicted functional conectivity (z)')
ax[0].set_ylabel('Measured functional conectivity (z)');


fc_df = pd.DataFrame(data=X, columns=['Connectivity', 'ROI size', 'ROI Distance', 'Completeness', 'fc'])
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
# comp_score = CompletenessMatrix.to_numpy().ravel()
# # diff_score = np.abs(DifferenceMatrix.to_numpy().ravel())
# diff_score = DifferenceMatrix.to_numpy().ravel()
#
# include_inds = np.where(diff_score != 0)[0]
# comp_score  = comp_score[include_inds]
# diff_score = diff_score[include_inds]
# anat_conn = ConnectivityCount.to_numpy().ravel()[include_inds]
#
#
# r, p = pearsonr(comp_score, diff_score)
#
# figS2, ax = plt.subplots(1, 2, figsize=(12,6))
# ax[0].scatter(comp_score, diff_score, marker='o', color='k', alpha=0.5)
#
# ax[0].set_xlabel('Completeness of reconstruction')
# ax[0].set_ylabel('abs(Anat. - Fxnal (z-score))')
#
# ax[1].scatter(comp_score, anat_conn, marker='o', color='k', alpha=0.5)
#
# ax[1].set_xlabel('Completeness of reconstruction')
# ax[1].set_ylabel('Anat connectivity')


# %% sort difference matrix by most to least different rois
diff_by_region = DifferenceMatrix.mean()
sort_inds = np.argsort(diff_by_region)
sort_keys = DifferenceMatrix.index[sort_inds]
sorted_diff = pd.DataFrame(data=np.zeros_like(DifferenceMatrix),columns=sort_keys, index=sort_keys)
for r_ind, r_key in enumerate(sort_keys):
    for c_ind, c_key in enumerate(sort_keys):
        sorted_diff.iloc[r_ind, c_ind]=DifferenceMatrix.loc[[r_key], [c_key]].to_numpy()

fig4_0, ax = plt.subplots(1, 1, figsize=(6,6))
lim = np.nanmax(np.abs(DifferenceMatrix.to_numpy().ravel()))
ax.scatter(A_zscore, F_zscore, alpha=1, c=diff, cmap="RdBu",  vmin=-lim, vmax=lim, edgecolors='k', linewidths=0.5)
ax.plot([-3, 3], [-3, 3], 'k-')
ax.set_xlabel('Anatomical connectivity (log10, zscore)')
ax.set_ylabel('Functional correlation (zscore)');

fig4_1, ax = plt.subplots(1, 1, figsize=(8,8))
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

fig4_2 = plt.figure(figsize=(15,3))
for z_ind, z in enumerate(zslices):
    ax = fig4_2.add_subplot(1, 5, z_ind+1)
    img = ax.imshow(diff_brain[:, :, z].T, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
    ax.set_axis_off()
    ax.set_aspect('equal')

cb = fig4_2.colorbar(img, ax=ax)
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
load_fn = os.path.join(data_dir, 'functional_connectivity', 'subsampled_cmats_20200619.npy')
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

figS1, ax1 = plt.subplots(1, 1, figsize=(5,5))
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

with PdfPages(os.path.join(analysis_dir, 'SC_FC_figs.pdf')) as pdf:
    pdf.savefig(fig1_0)
    pdf.savefig(fig1_1)
    pdf.savefig(fig1_2)
    pdf.savefig(fig1_3)
    pdf.savefig(fig1_4)

    pdf.savefig(fig2_0)
    pdf.savefig(fig2_1)
    pdf.savefig(fig2_2)

    pdf.savefig(fig3_0)
    pdf.savefig(fig3_1)

    pdf.savefig(fig4_0)
    pdf.savefig(fig4_1)
    pdf.savefig(fig4_2)

    pdf.savefig(figS1)

    d = pdf.infodict()
    d['Title'] = 'SC-FC early figs'
    d['Author'] = 'Max Turner'
    d['ModDate'] = datetime.datetime.today()

plt.close('all')

from neuprint import Client
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import os
from scipy.stats import pearsonr
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from scipy import stats
from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
import pickle

import nibabel as nib

import networkx as nx
from region_connectivity import RegionConnectivity

"""
https://connectome-neuprint.github.io/neuprint-python/docs/index.html
https://github.com/connectome-neuprint/neuprint-python

TODO:
-Activity states, how does fxnal map change?
-Check correlation with size of ROI

-Network connectivity metrics, and compare to other brains?
-Can we see network structure in anatomy that isn't there in fxn?

"""
analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/hemibrain_analysis/roi_connectivity'
atlas_dir = '/home/mhturner/GitHub/DrosAdultBRAINdomains'

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.0.1', token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1heHdlbGxob2x0ZXR1cm5lckBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpMHJRX0M4akliX0ZrS2h2OU5DSElsWlpnRDY5YUMtVGdNLWVWM3lRP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzY2MTk1MzcwfQ.Q-57D4tX2sXMjWym2LFhHaUGHgHiUsIM_JI9xekxw_0')


mapping = RegionConnectivity.getRoiMapping(neuprint_client)
rois = list(mapping.keys())
rois.sort()
roi_completeness = RegionConnectivity.getRoiCompleteness(neuprint_client, mapping)





# %%






# %%
fn_atlas = 'JFRCtempate2010.mask130819_crop.nii'
fn_atlas_index = 'Original_Index_panda.csv'

# load atlas data
pull_inds = np.array([ 3,  4, 5,  6,  7,  8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23,
   25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49])
atlas_index = pd.read_csv(os.path.join(analysis_dir, 'data', fn_atlas_index)).iloc[pull_inds, :]
# match to fxn roi names
new_rois = atlas_index.name
new_rois = [x.split('_R')[0] for x in new_rois]
new_rois = [x.replace('_', '') for x in new_rois]
atlas_index.name = new_rois

# load mask brain
mask_brain = np.asarray(np.squeeze(nib.load(os.path.join(analysis_dir, 'from_kevin', fn_atlas)).get_fdata()), 'uint8')

# get individual roi masks and compute size of each roi (in voxels)
roi_mask = []
roi_size = []
for r in rois_fxn:
    new_roi_mask = np.zeros_like(mask_brain)
    target_index = atlas_index[atlas_index.name==r].num.to_numpy()[0]
    new_roi_mask = mask_brain == target_index
    roi_mask.append(new_roi_mask)
    roi_size.append(np.sum(new_roi_mask))

# find center of mass for each roi
coms = np.vstack([center_of_mass(x) for x in roi_mask])

# calulcate euclidean distance matrix between roi centers of mass
dist_mat = np.zeros_like(CM_fxn)
upper_inds = np.triu_indices(dist_mat.shape[0], k=1)
dist_mat[upper_inds] = pdist(coms)
dist_mat += dist_mat.T # symmetrize

distance_mat = pd.DataFrame(data=dist_mat, index=rois_fxn, columns=rois_fxn)

# geometric mean of the sizes for each pair of ROIs
sz_mat = np.sqrt(np.outer(np.array(roi_size), np.array(roi_size)))
size_mat = pd.DataFrame(data=sz_mat, index=rois_fxn, columns=rois_fxn)

# %% get pre-computed connectivity matrices for rois of interest

ConnectivityMatrix_ct = RegionConnectivity.getPrecomputedConnectivityMatrix(neuprint_client, mapping, metric='count', diagonal='nan')
ConnectivityMatrix_wt = RegionConnectivity.getPrecomputedConnectivityMatrix(neuprint_client, mapping, metric='weight', diagonal='nan')

# rename roi names to drop the (R)
roi_names = ConnectivityMatrix_ct.index
roi_names = [x.split('(')[0] for x in rois]

ConnectivityMatrix_ct.index = roi_names
ConnectivityMatrix_ct.columns = roi_names

ConnectivityMatrix_wt.index = roi_names
ConnectivityMatrix_wt.columns = roi_names


# %% load my own pre-computed connectivity matrix

ConnectivityMatrix_comp = pd.read_pickle(os.path.join(analysis_dir, 'ConnectivityMatrix_computed_20200422.pkl'))
SynapseCount_comp = pd.read_pickle(os.path.join(analysis_dir, 'SynapseCount_computed_20200422.pkl'))
with open(os.path.join(analysis_dir, 'NeuronCount_computed_20200422.pkl'), 'rb') as f:
    NeuronCount_comp = pickle.load(f)

roi_names = ConnectivityMatrix_comp.index
roi_names = [x.split('(')[0] for x in rois]

# normalize each target (column) by total number of input synapses
tmp_mat = NeuronCount_comp[0].to_numpy().copy()
frac_pre = SynapseCount_comp['assigned_synapses'].to_numpy() / SynapseCount_comp['total_synapses'].to_numpy()
frac_post = roi_completeness['frac_post']
tmp_mat = tmp_mat / frac_pre[None, :]
tmp_mat = tmp_mat / frac_post[:, None]


# set diag to nan
np.fill_diagonal(tmp_mat, np.nan)
ConnectivityMatrix_comp = pd.DataFrame(data=tmp_mat, index=roi_names, columns=roi_names)


ConnectivityMatrix_comp.loc['MBPED', 'MBCA']
ConnectivityMatrix_ct.loc['MBPED', 'MBCA']

ConnectivityMatrix_comp.loc['WED', 'SPS']
ConnectivityMatrix_ct.loc['WED', 'SPS']

# %%

fh, ax = plt.subplots(1, 3, figsize=(18,6))
sns.heatmap(ConnectivityMatrix_wt, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Weight'}, cmap="viridis", rasterized=True)
ax[0].set_aspect('equal')

sns.heatmap(ConnectivityMatrix_ct, ax=ax[1], xticklabels=True, cbar_kws={'label': 'ct'}, cmap="viridis", rasterized=True)
ax[1].set_aspect('equal')

sns.heatmap(ConnectivityMatrix_comp, ax=ax[2], xticklabels=True, cbar_kws={'label': 'computed'}, cmap="viridis", rasterized=True)
ax[2].set_aspect('equal')


# %%

fh1, ax = plt.subplots(1, 2, figsize=(16,8))
sns.heatmap(ConnectivityMatrix_ct, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Cell count', 'shrink': .8}, cmap="cividis", rasterized=True)
ax[0].set_xlabel('Target');
ax[0].set_ylabel('Source');
ax[0].set_aspect('equal')

sns.heatmap(CorrMat_fxnal, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Correlation','shrink': .8}, cmap="cividis", rasterized=True)
ax=ax[1].set_aspect('equal')

# %% check symmetry of anatomical matrices
upper_inds = np.triu_indices(ConnectivityMatrix_ct.shape[0], k=1) #k=1 excludes main diagonal

to_ct = ConnectivityMatrix_ct.to_numpy()[upper_inds]
from_ct = ConnectivityMatrix_ct.to_numpy().T[upper_inds]

to_wt = ConnectivityMatrix_wt.to_numpy()[upper_inds]
from_wt = ConnectivityMatrix_wt.to_numpy().T[upper_inds]

to_comp = ConnectivityMatrix_comp.to_numpy()[upper_inds]
from_comp = ConnectivityMatrix_comp.to_numpy().T[upper_inds]

ct_ind = 0
to_ct_comp = NeuronCount_comp[ct_ind].to_numpy()[upper_inds]
from_ct_comp = NeuronCount_comp[ct_ind].to_numpy().T[upper_inds]

fh2, ax = plt.subplots(1, 4, figsize=(16,3))
ax[0].plot(to_ct, from_ct, 'ko')
ax[0].plot([0, np.nanmax(to_ct)], [0, np.nanmax(to_ct)], 'k--')
ax[0].set_title('Neuron count - loaded')
ax[0].set_ylabel('B=>A')

ax[1].plot(to_wt, from_wt, 'ko')
ax[1].plot([0, np.nanmax(to_wt)], [0, np.nanmax(to_wt)], 'k--')
ax[1].set_title('Weight - loaded')
ax[1].set_xlabel('A=>B')

ax[2].plot(to_ct_comp, from_ct_comp, 'ko')
ax[2].plot([0, np.nanmax(to_ct_comp)], [0, np.nanmax(to_ct_comp)], 'k--')
ax[2].set_title('Count - computed')

ax[3].plot(to_comp, from_comp, 'ko')
ax[3].plot([0, np.nanmax(to_comp)], [0, np.nanmax(to_comp)], 'k--')
ax[3].set_title('Weight - computed');

# %%
# save the three matrices
ConnectivityMatrix_wt.to_pickle(os.path.join(analysis_dir, 'ConnectivityMatrix_wt.pkl'))
ConnectivityMatrix_ct.to_pickle(os.path.join(analysis_dir, 'ConnectivityMatrix_ct.pkl'))
CorrelationMatrix_Functional.to_pickle(os.path.join(analysis_dir, 'CorrelationMatrix_Functional.pkl'))

# %% hist of completeness of included rois and correlation between completeness and count, weight

fh2, ax = plt.subplots(1, 2, figsize=(8,4))
ax[0].hist(roi_completeness['frac_pre']);
ax[0].set_title('Frac. output synapses assigned')
ax[0].set_ylabel('Num. ROIs')
ax[1].hist(roi_completeness['frac_post']);
ax[1].set_title('Frac. input synapses assigned');

# define "completeness" between two rois as the product of pre and post completeness
completeness_matrix = np.array(roi_completeness['frac_pre']).reshape(-1, 1) @ np.array(roi_completeness['frac_pre']).reshape(-1, 1).T
upper_inds = np.triu_indices(completeness_matrix.shape[0], k=1) #k=1 excludes main diagonal
ct = (ConnectivityMatrix_ct.to_numpy() + ConnectivityMatrix_ct.to_numpy().T)[upper_inds]
wt = (ConnectivityMatrix_wt.to_numpy() + ConnectivityMatrix_wt.to_numpy().T)[upper_inds]
ct_comp = (NeuronCount_comp[ct_ind].to_numpy() + NeuronCount_comp[ct_ind].to_numpy().T)[upper_inds]
computed = (ConnectivityMatrix_comp.to_numpy() + ConnectivityMatrix_comp.to_numpy().T)[upper_inds]
compl = completeness_matrix[upper_inds]

fh3, ax = plt.subplots(1, 5, figsize=(15,3))
r, p = pearsonr(compl, wt)
coef = np.polyfit(compl, wt,1)
poly1d_fn = np.poly1d(coef)
xx = [compl.min(), compl.max()]
ax[0].plot(compl, wt, 'ko')
ax[0].plot(xx, poly1d_fn(xx), 'k-')
ax[0].set_title(r)
ax[0].set_xlabel('completeness')
ax[0].set_ylabel('weight')

r, p = pearsonr(compl, ct)
coef = np.polyfit(compl, ct,1)
poly1d_fn = np.poly1d(coef)
xx = [compl.min(), compl.max()]
ax[1].plot(compl, ct, 'ko')
ax[1].plot(xx, poly1d_fn(xx), 'k-')
ax[1].set_title(r)
ax[1].set_xlabel('completeness')
ax[1].set_ylabel('count')

r, p = pearsonr(ct, wt)
coef = np.polyfit(ct, wt,1)
poly1d_fn = np.poly1d(coef)
xx = [ct.min(), ct.max()]
ax[2].plot(ct, wt, 'ko')
ax[2].plot(xx, poly1d_fn(xx), 'k-')
ax[2].set_title(r)
ax[2].set_xlabel('count')
ax[2].set_ylabel('weight');

r, p = pearsonr(ct, ct_comp)
coef = np.polyfit(ct, ct_comp,1)
poly1d_fn = np.poly1d(coef)
xx = [ct.min(), ct.max()]
ax[3].plot(ct, ct_comp, 'ko')
ax[3].plot(xx, poly1d_fn(xx), 'k-')
ax[3].set_title(r)
ax[3].set_xlabel('count')
ax[3].set_ylabel('count_computed');

r, p = pearsonr(computed, wt)
coef = np.polyfit(computed, wt,1)
poly1d_fn = np.poly1d(coef)
xx = [computed.min(), computed.max()]
ax[4].plot(computed, wt, 'ko')
ax[4].plot(xx, poly1d_fn(xx), 'k-')
ax[4].set_title(r)
ax[4].set_xlabel('computed wt')
ax[4].set_ylabel('wt');


# %% correlation between functional correlation matrix and symmetrized connectivity matrix

adj_mat = ConnectivityMatrix_ct.to_numpy().copy()
# symmetrize adjacency matrix by just adding it to its transpose. Ignores directionality
adj_mat_sym = adj_mat + adj_mat.T

upper_inds = np.triu_indices(adj_mat_sym.shape[0], k=1) #k=1 excludes main diagonal
functional_adjacency = CorrMat_fxnal.to_numpy()[upper_inds]
em_adjacency = adj_mat_sym[upper_inds]

fh4, ax = plt.subplots(1, 3, figsize=(12,4))

r, p = pearsonr(em_adjacency, functional_adjacency)
coef = np.polyfit(em_adjacency, functional_adjacency,1)
poly1d_fn = np.poly1d(coef)
xx = [em_adjacency.min(), em_adjacency.max()]
ax[0].plot(em_adjacency, functional_adjacency, 'ko', alpha=0.6)
ax[0].plot(xx, poly1d_fn(xx), 'k-')
ax[0].set_title('r = {:.3f}'.format(r));
ax[0].set_xlabel('Anatomical connectivity count')
ax[0].set_ylabel('Functional correlation');

keep_inds = np.where(em_adjacency > 0)[0]
em_adjacency_no0 = em_adjacency[keep_inds]
functional_adjacency_no0 = functional_adjacency[keep_inds]

r, p = pearsonr(em_adjacency_no0, functional_adjacency_no0)
coef = np.polyfit(em_adjacency_no0, functional_adjacency_no0,1)
poly1d_fn = np.poly1d(coef)
xx = [em_adjacency_no0.min(), em_adjacency_no0.max()]
ax[1].plot(em_adjacency_no0, functional_adjacency_no0, 'ko', alpha=0.6)
ax[1].plot(xx, poly1d_fn(xx), 'k-')
ax[1].set_title('r = {:.3f} (only nonzero connections)'.format(r));
ax[1].set_xlabel('Anatomical connectivity count')
ax[1].set_ylabel('Functional correlation');


em_adjacency_log = np.log10(em_adjacency_no0)
functional_adjacency_log = functional_adjacency_no0

r, p = pearsonr(em_adjacency_log, functional_adjacency_log)
coef = np.polyfit(em_adjacency_log, functional_adjacency_log,1)
poly1d_fn = np.poly1d(coef)
xx = [em_adjacency_log.min(), em_adjacency_log.max()]
ax[2].plot(em_adjacency_log, functional_adjacency_log, 'ko', alpha=0.6)
ax[2].plot(xx, poly1d_fn(xx), 'k-')
ax[2].set_title('r = {:.3f} (log transformed)'.format(r));
ax[2].set_xlabel('Log10 anatomical connectivity count')
ax[2].set_ylabel('Functional correlation');


plt.hist(em_adjacency_log, 20)

# %% dependence of fxnal correlation on roi distance
upper_inds = np.triu_indices(distance_mat.to_numpy().shape[0], k=1) #k=1 excludes main diagonal

fh5, ax = plt.subplots(1, 2, figsize=(12,6))
fc = zscore(CorrMat_fxnal.to_numpy()[upper_inds])
dist = zscore(distance_mat.to_numpy()[upper_inds])

r, p = pearsonr(dist, fc)
coef = np.polyfit(dist, fc, 1)
poly1d_fn = np.poly1d(coef)
xx = [dist.min(), dist.max()]
ax[0].scatter(dist, fc, color='k', alpha=0.5)
ax[0].plot(xx, poly1d_fn(xx), 'k-')
ax[0].set_xlabel('Distance between ROIs (zscore)')
ax[0].set_ylabel('Functional correlation (zscore)')
ax[0].set_title('r = {}'.format(r))


residuals = fc - dist
anat = ConnectivityMatrix_ct.to_numpy()[upper_inds]

keep_inds = np.where(anat > 0)[0]
residuals = residuals[keep_inds]
anat = np.log10(anat[keep_inds])

r, p = pearsonr(anat, residuals)
coef = np.polyfit(anat, residuals, 1)
poly1d_fn = np.poly1d(coef)
xx = [anat.min(), anat.max()]
ax[1].scatter(anat, residuals, color='k', alpha=0.5)
ax[1].plot(xx, poly1d_fn(xx), 'k-')
ax[1].set_xlabel('Anatomical connectivity (log10)')
ax[1].set_ylabel('Residual (zscore)')
ax[1].set_title('r = {}'.format(r));

# %%


do_log = True

adj_mat = ConnectivityMatrix_ct.to_numpy().copy()
adj_mat_sym = adj_mat + adj_mat.T  # symmetrize

upper_inds = np.triu_indices(adj_mat_sym.shape[0], k=1)  # k=1 excludes main diagonal
functional_adjacency = CorrMat_fxnal.to_numpy()[upper_inds]
em_adjacency = adj_mat_sym[upper_inds]

if do_log:
    # cut out zeros
    keep_inds = np.where(em_adjacency > 0)[0]
    em_adjacency_no0 = em_adjacency[keep_inds]
    functional_adjacency_no0 = functional_adjacency[keep_inds]
    em_adjacency_log = np.log10(em_adjacency_no0)

    F_zscore = zscore(functional_adjacency_no0)
    A_zscore = zscore(em_adjacency_log)

    difference_matrix = np.zeros_like(CorrMat_fxnal.to_numpy())
    diff = A_zscore - F_zscore
    difference_matrix[upper_inds[0][keep_inds], upper_inds[1][keep_inds]] = diff
    np.fill_diagonal(difference_matrix, np.nan)
    difference_matrix += difference_matrix.T
    difference_df = pd.DataFrame(data=difference_matrix, index=roi_names, columns=roi_names)
else:
    F_zscore = zscore(functional_adjacency)
    A_zscore = zscore(em_adjacency)

    difference_matrix = np.zeros_like(CorrMat_fxnal.to_numpy())
    diff = A_zscore - F_zscore
    difference_matrix[upper_inds] = diff
    np.fill_diagonal(difference_matrix, np.nan)
    difference_matrix += difference_matrix.T
    difference_df = pd.DataFrame(data=difference_matrix, index=roi_names, columns=roi_names)



lim = np.nanmax(np.abs(difference_df.to_numpy().ravel()))
fh6, ax = plt.subplots(1, 1, figsize=(6,6))
col = diff
ax.scatter(A_zscore, F_zscore, alpha=1, c=col, cmap="RdBu",  vmin=-lim, vmax=lim, edgecolors='k', linewidths=0.5)
ax.plot([-3, 3], [-3, 3], 'k-')
ax.set_xlabel('Anatomical connectivity count (log10, zscore)')
ax.set_ylabel('Functional correlation (zscore)');


fh7, ax = plt.subplots(1, 2, figsize=(18,9))
sns.heatmap(difference_df, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Anat - Fxnal connectivity','shrink': .75}, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
ax[0].set_aspect('equal')

sns.heatmap(distance_mat, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Physical distance','shrink': .75}, rasterized=True, cmap='cividis')
ax[1].set_aspect('equal')
#
# sns.heatmap(size_mat, ax=ax[2], xticklabels=True, cbar_kws={'label': 'G. Mean sizes','shrink': .8}, rasterized=True, cmap='cividis')
# ax[2].set_aspect('equal')

# %% examine effect of completeness on difference & fxnal corr
diff_by_region = difference_df.mean()
roi_completeness['frac_post']

fh, ax = plt.subplots(1, 2, figsize=(10,5))
r, p = pearsonr(roi_completeness['frac_post'], diff_by_region)
coef = np.polyfit(roi_completeness['frac_post'], diff_by_region, 1)
poly1d_fn = np.poly1d(coef)
ax[0].scatter(roi_completeness['frac_post'], diff_by_region)
ax[0].set_title('r={}'.format(r))
ax[0].set_xlabel('frac_post')
ax[0].set_ylabel('Anat-Fxn')

r, p = pearsonr(roi_completeness['frac_pre'], diff_by_region)
coef = np.polyfit(roi_completeness['frac_pre'], diff_by_region, 1)
poly1d_fn = np.poly1d(coef)
ax[1].scatter(roi_completeness['frac_pre'], diff_by_region)
ax[1].set_title('r={}'.format(r));
ax[1].set_xlabel('frac_pre')
ax[1].set_ylabel('Anat-Fxn')

corr_by_region = CorrMat_fxnal.mean()
fh, ax = plt.subplots(1, 2, figsize=(10,5))
r, p = pearsonr(roi_completeness['frac_post'], corr_by_region)
coef = np.polyfit(roi_completeness['frac_post'], corr_by_region, 1)
poly1d_fn = np.poly1d(coef)
ax[0].scatter(roi_completeness['frac_post'], corr_by_region)
ax[0].set_title('r={}'.format(r))
ax[0].set_xlabel('frac_post')
ax[0].set_ylabel('Mean Fxnal corr by region')

r, p = pearsonr(roi_completeness['frac_pre'], corr_by_region)
coef = np.polyfit(roi_completeness['frac_pre'], corr_by_region, 1)
poly1d_fn = np.poly1d(coef)
ax[1].scatter(roi_completeness['frac_pre'], corr_by_region)
ax[1].set_title('r={}'.format(r))
ax[1].set_xlabel('frac_pre')
ax[1].set_ylabel('Mean Fxnal corr by region')

# %%


# %% linear regression
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# sz_mat = np.sqrt(np.outer(np.array(roi_size), np.array(roi_size)))

pre_pre_mat = np.sqrt(np.outer(np.array(roi_completeness['frac_pre']), np.array(roi_completeness['frac_pre'])))
pre_post_mat = np.sqrt(np.outer(np.array(roi_completeness['frac_pre']), np.array(roi_completeness['frac_post'])))
post_pre_mat = np.sqrt(np.outer(np.array(roi_completeness['frac_post']), np.array(roi_completeness['frac_pre'])))
post_post_mat = np.sqrt(np.outer(np.array(roi_completeness['frac_post']), np.array(roi_completeness['frac_post'])))

upper_inds = np.triu_indices(ConnectivityMatrix_ct.shape[0], k=1)  # k=1 excludes main diagonal

pre_pre = pre_pre_mat[upper_inds[0][keep_inds], upper_inds[1][keep_inds]]
pre_post = pre_post_mat[upper_inds[0][keep_inds], upper_inds[1][keep_inds]]
post_pre = post_pre_mat[upper_inds[0][keep_inds], upper_inds[1][keep_inds]]
post_post = post_post_mat[upper_inds[0][keep_inds], upper_inds[1][keep_inds]]
sz = size_mat.to_numpy()[upper_inds[0][keep_inds], upper_inds[1][keep_inds]]
dist = distance_mat.to_numpy()[upper_inds[0][keep_inds], upper_inds[1][keep_inds]]

# %%

r, p = pearsonr(dist, F_zscore)
print('r = {}; p = {}'.format(r, p))

# %%

X = np.vstack([A_zscore, sz, dist, pre_pre, pre_post, post_pre, post_post]).T

# X = np.vstack([A_zscore, sz, dist]).T
# X = np.vstack([pre_pre, pre_post, post_pre, post_post]).T
# X = np.vstack([sz, dist]).T

X_train, X_test, y_train, y_test = train_test_split(X, F_zscore, test_size=0.1, random_state=1)

est = sm.OLS(y_train, X_train)
est2 = est.fit()
print(est2.summary())

pred_y = est2.predict(X_test)


plt.plot(y_test, pred_y, 'ko')
plt.plot([-2, 2], [-2, 2], 'k--')

# %%

# X = np.vstack([A_zscore, sz, pre_pre, pre_post, post_pre, post_post]).T # samples x features
X = np.vstack([A_zscore, sz, pre_pre, pre_post, post_pre, post_post]).T # samples x features
regressor = LinearRegression()
regressor.fit(X, F_zscore);
regressor.intercept_
print(regressor.coef_)
regressor.score(X, F_zscore)
r, p = pearsonr(sz, F_zscore)


# xx = np.array([-3, 3])
# lim = np.nanmax(np.abs(difference_df.to_numpy().ravel()))
# fh6, ax = plt.subplots(1, 1, figsize=(6,6))
# col = diff
# ax.scatter(A_zscore, F_zscore, alpha=1, c=col, cmap="RdBu",  vmin=-lim, vmax=lim, edgecolors='k', linewidths=0.5)
# ax.plot(xx, regressor.predict(xx[:, None]), 'k-')
# ax.set_xlabel('Anatomical connectivity count (log10, zscore)')
# ax.set_ylabel('Functional correlation (zscore)');

# %%
#build brain map of regions color coded by difference between fxn and anatomical
diff_brain = np.zeros(shape=mask_brain.shape)
diff_brain[:] = np.nan


for r_ind, r in enumerate(roi_mask):
    diff_brain[r] = diff_by_region[r_ind]

# %%
fh, ax = plt.subplots(1, 1, figsize=(8,8))
ax.imshow(np.nanmean(diff_brain, axis=2).T, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)

# %%
diff_brain.shape
zslices = np.arange(30, 200, 20)
lim = np.nanmax(np.abs(difference_df.to_numpy().ravel()))


fh8 = plt.figure(figsize=(12,12), facecolor='black')
for z_ind, z in enumerate(zslices):
    ax = fh8.add_subplot(3, 3, z_ind+1)
    ax.imshow(diff_brain[:, :, z].T, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
    ax.set_axis_off()
# %%
# whereas fxnal connectivity is normally distributed,
# count connectivity is much closer to lognormal than normal
# may help justify log transform before z-scoring


# stats.kstest(F_zscore, 'norm')
# stats.kstest(A_zscore, 'norm')

fh9, ax = plt.subplots(1, 3, figsize=(14,4))
ax[0].hist(zscore(functional_adjacency), 20);
ax[0].set_title('fxnal (Z)')
ax[1].hist(zscore(em_adjacency_no0), 20);
ax[1].set_title('connectivity (Z)')
ax[2].hist(zscore(em_adjacency_log), 20);
ax[2].set_title('connectivity (log, Z)');

# %%
"""
Cluster based on anatomy and sort functional matrix by those clusters
"""

n_clusters = 5


# get anatomical matrix and symmetrize it. Set diagonal nans to 0
anat_mat = ConnectivityMatrix_ct.to_numpy().copy()
anat_mat = anat_mat + anat_mat.T
np.fill_diagonal(anat_mat, 0)
anat_DF = pd.DataFrame(data=anat_mat, columns=roi_names, index=roi_names)

# get fxnal matrix and set diagonal nans to 0, already symmetric
fxn_mat = CorrMat_fxnal.to_numpy().copy()
# thresh = 0.4
# fxn_mat[fxn_mat<thresh] = 0
np.fill_diagonal(fxn_mat, 0)
fxn_DF = pd.DataFrame(data=fxn_mat, columns=roi_names, index=roi_names)

# cluster anatomical matrix
sc_anat = SpectralClustering(n_clusters, affinity='precomputed', n_init=100)
sc_anat.fit(anat_DF.to_numpy());
sort_inds = np.argsort(sc_anat.labels_)
sort_keys = list(np.array(roi_names)[sort_inds])
cluster_lines = np.where(np.diff(np.sort(sc_anat.labels_))==1)[0]+1
sorted_anat = pd.DataFrame(data=np.zeros_like(anat_mat),columns=sort_keys, index=sort_keys)
sorted_fxn = pd.DataFrame(data=np.zeros_like(fxn_mat),columns=sort_keys, index=sort_keys)
sorted_diff = pd.DataFrame(data=np.zeros_like(difference_df),columns=sort_keys, index=sort_keys)
for r_ind, r_key in enumerate(sort_keys):
    for c_ind, c_key in enumerate(sort_keys):
        sorted_anat.iloc[r_ind, c_ind]=anat_DF.loc[[r_key], [c_key]].to_numpy()
        sorted_fxn.iloc[r_ind, c_ind]=fxn_DF.loc[[r_key], [c_key]].to_numpy()
        sorted_diff.iloc[r_ind, c_ind]=difference_df.loc[[r_key], [c_key]].to_numpy()

fh10, ax = plt.subplots(1, 3, figsize=(24,8))
sns.heatmap(sorted_anat, ax=ax[0], xticklabels=True, cbar_kws={'label': 'count (symmetrized)', 'shrink': .75}, cmap="cividis", rasterized=True)
for cl in cluster_lines:
    ax[0].vlines(cl, 0, 40, color='w')
    ax[0].hlines(cl, 0, 40, color='w')

ax[0].set_aspect('equal')
ax[0].set_title('Anatomical prediction');

sns.heatmap(sorted_fxn, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Correlation', 'shrink': .75}, cmap="cividis", rasterized=True)
for cl in cluster_lines:
    ax[1].vlines(cl, 0, 40, color='w')
    ax[1].hlines(cl, 0, 40, color='w')

ax[1].set_aspect('equal')
ax[1].set_title('Functional correlation');

sns.heatmap(sorted_diff, ax=ax[2], xticklabels=True, cbar_kws={'label': 'Anat - Fxnal connectivity','shrink': .75}, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
for cl in cluster_lines:
    ax[2].vlines(cl, 0, 40, color='k')
    ax[2].hlines(cl, 0, 40, color='k')

ax[2].set_aspect('equal')
ax[2].set_title('Difference');
# %%
# within cluster Correlation
fh11, ax = plt.subplots(n_clusters, n_clusters, figsize=(8,8))
[x.set_axis_off() for x in ax.ravel()]
corr_mat = np.zeros((n_clusters, n_clusters))
corr_mat[:] = np.nan
for row in range(n_clusters):
    for col in range(n_clusters):
        if row >= col:
            cluster_mat_anat = anat_DF.iloc[np.where(sc_anat.labels_==row)[0], np.where(sc_anat.labels_==col)[0]].to_numpy()
            cluster_mat_fxn = fxn_DF.iloc[np.where(sc_anat.labels_==row)[0], np.where(sc_anat.labels_==col)[0]].to_numpy()

            if row == col: # on the diagonal. only take the upper triangle since these are symmetric
                upper_inds = np.triu_indices(cluster_mat_anat.shape[0], k=1, m=cluster_mat_anat.shape[1]) #k=1 excludes main diagonal
                functional_adjacency = cluster_mat_fxn[upper_inds]
                em_adjacency = cluster_mat_anat[upper_inds]
            else:
                functional_adjacency = cluster_mat_fxn.ravel()
                em_adjacency = cluster_mat_anat.ravel()

            r, p = pearsonr(em_adjacency, functional_adjacency)
            coef = np.polyfit(em_adjacency, functional_adjacency,1)
            poly1d_fn = np.poly1d(coef)
            if row >= col:
                corr_mat[row, col] = r

            xx = [em_adjacency.min(), em_adjacency.max()]
            ax[row, col].plot(em_adjacency, functional_adjacency, 'ko', alpha=0.6)
            ax[row, col].plot(xx, poly1d_fn(xx), 'k-')
            ax[row, col].set_axis_off()


fh12, ax = plt.subplots(1, 1, figsize=(6,6))
sns.heatmap(corr_mat, ax=ax, xticklabels=True, cbar_kws={'label': 'corr(anat, fxn)', 'shrink': .8}, cmap="viridis", rasterized=True, vmax=1)
ax.set_xlabel('Cluster_r')
ax.set_ylabel('Cluster_c')
ax.set_aspect('equal')
ax.set_title('Correlation for each cluster');

# %% network of collapsed clusters

labels = {}
clust_connection_matrix = np.zeros((n_clusters, n_clusters))
for source_cluster in range(n_clusters):
    source_rois = np.array(roi_names)[np.where(sc_anat.labels_ == source_cluster)[0]]
    if 'EB' in source_rois:
        labels[source_cluster] = 'CC'
    elif 'MBPED' in source_rois:
        labels[source_cluster] = 'MB'
    else:
        labels[source_cluster] = str(source_cluster)

    for target_cluster in range(n_clusters):
        target_rois = np.array(roi_names)[np.where(sc_anat.labels_ == target_cluster)[0]]

        conns = ConnectivityMatrix_ct.loc[source_rois, target_rois]

        clust_connection_matrix[source_cluster, target_cluster] = conns.mean().mean()

DG = nx.from_numpy_matrix(clust_connection_matrix, parallel_edges=False, create_using=nx.DiGraph)

edges, weights = zip(*nx.get_edge_attributes(DG,'weight').items())
weights = np.array(weights)

widths = 1 + 40 * weights / np.max(weights)

draw_options = {
    'cmap': 'tab10',
    'edge_cmap': plt.get_cmap('Blues'),
    'edge_vmin': weights.min(),
    'edge_vmax': weights.max(),
    'alpha': 1.0,
    'vmin': 0,
    'vmax': n_clusters,
    'font_weight': 'bold',
    'node_size': 100,
    'width': widths,
    'arrows': True,
    'arrowstyle': '-|>',
    'arrowsize': 12,
    'labels': labels,
}

fh, ax = plt.subplots(1, 1, figsize=(9,9))
nx.draw_circular(DG, edgelist=edges, edge_color=weights, ax=ax, **draw_options)
# %% plot clusters in geometric space

for c in range(n_clusters):
    clust_inds = np.where(sc_anat.labels_ == c)[0]
    clust_rois = np.array(rois)[clust_inds]

clust_rois

roi_name = 'AL(R)'

neuprint_client.fetch_roi_mesh(roi_name, export_path=os.path.join(analysis_dir, 'meshes', '{}.obj'.format(roi_name)));

# %%

fxn_DF

Z = linkage(fxn_DF.to_numpy(), method='ward')
C = fcluster(Z, t=5, criterion='maxclust')
dendrogram(Z, p=8, truncate_mode='lastp');

# %%

fxn_mat = fxn_DF.to_numpy()
thresh_fxn = 0.3*np.max(fxn_mat)
fxn_mat = np.round(fxn_mat > thresh_fxn)


anat_mat = anat_DF.to_numpy()
thresh_anat = 0.3*np.max(anat_mat)
fxn_mat = np.round(fxn_mat > thresh_fxn)


DG_anat = nx.from_numpy_matrix(anat_mat, parallel_edges=False, create_using=nx.DiGraph)
DG_fxn = nx.from_numpy_matrix(fxn_mat, parallel_edges=False, create_using=nx.Graph)

# %%

draw_options = {
    'node_color': sc_anat.labels_,
    'cmap': 'tab10',
    'alpha': 1.0,
    'vmin': 0,
    'vmax': n_clusters,
    'font_weight': 'bold',
    'node_size': 100,
    'width': 1,
    'arrows': True,
    'arrowstyle': '-|>',
    'arrowsize': 12,
}

fh, ax = plt.subplots(1, 2, figsize=(12,6))
nx.draw_spring(DG_anat, ax=ax[0], **draw_options)

nx.draw_spring(DG_fxn, ax=ax[1], **draw_options)



# %% cluster adjacency matrix and plot digraph
# make digraph

thresh = 10
con_mat = ConnectivityMatrix_ct.to_numpy().copy()

# con_mat[con_mat < thresh] = 0
DG = nx.from_numpy_matrix(con_mat, parallel_edges=False, create_using=nx.DiGraph)

labels = {}
for i, lab in enumerate(list(ConnectivityMatrix_ct.index)):
    labels[i] = lab

edges, weights = zip(*nx.get_edge_attributes(DG,'weight').items())
weights = np.log(np.array(weights))
# weights = np.array(weights)

draw_options = {
    'node_color': sc_anat.labels_,
    'cmap': 'tab10',
    'edge_cmap': plt.get_cmap('viridis'),
    'edge_vmin': weights.min(),
    'edge_vmax': weights.max(),
    'alpha': 1.0,
    'vmin': 0,
    'vmax': n_clusters,
    'font_weight': 'bold',
    'node_size': 100,
    'width': 1,
    'arrows': True,
    'arrowstyle': '-|>',
    'arrowsize': 12,
    'labels': labels,
}

fh4, ax = plt.subplots(1, 1, figsize=(12,12))
ax.set_aspect('equal')
# nx.draw_circular(DG, edgelist=edges, edge_color=weights, ax=ax, **draw_options)
nx.draw_spring(DG, edgelist=edges, edge_color=weights, ax=ax, **draw_options)

# %%


with PdfPages(os.path.join(analysis_dir, 'roi_connectivity_comparison_v3.pdf')) as pdf:
    pdf.savefig(fh1)
    pdf.savefig(fh2)
    pdf.savefig(fh3)
    pdf.savefig(fh4)
    pdf.savefig(fh5)
    pdf.savefig(fh6)
    pdf.savefig(fh7)
    pdf.savefig(fh8)
    pdf.savefig(fh9)
    pdf.savefig(fh10)
    pdf.savefig(fh11)
    pdf.savefig(fh12)

    d = pdf.infodict()
    d['Title'] = 'Connectivity'
    d['Author'] = 'Max Turner'
    d['ModDate'] = datetime.datetime.today()

plt.close('all')

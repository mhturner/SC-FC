from neuprint import Client
import pandas as pd
import numpy as np
import nibabel as nib
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from scipy.ndimage.measurements import center_of_mass
from sklearn.cluster import SpectralClustering
from matplotlib.backends.backend_pdf import PdfPages
import datetime

from region_connectivity import RegionConnectivity


"""
References:
https://connectome-neuprint.github.io/neuprint-python/docs/index.html
https://github.com/connectome-neuprint/neuprint-python

"""


analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/hemibrain_analysis/roi_connectivity'
atlas_dir = '/home/mhturner/GitHub/DrosAdultBRAINdomains'

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.0.1', token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1heHdlbGxob2x0ZXR1cm5lckBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpMHJRX0M4akliX0ZrS2h2OU5DSElsWlpnRDY5YUMtVGdNLWVWM3lRP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzY2MTk1MzcwfQ.Q-57D4tX2sXMjWym2LFhHaUGHgHiUsIM_JI9xekxw_0')
mapping = RegionConnectivity.getRoiMapping()
rois = list(mapping.keys())
rois.sort()

roi_completeness = RegionConnectivity.getRoiCompleteness(neuprint_client, mapping)


# %% LOAD DATA
"""
Functional connectivity matrix
    Avg across animals, fischer z transformed correlation values

    :CorrelationMatrix_Functional
"""
CorrelationMatrix_Full = RegionConnectivity.prepFullCorrrelationMatrix()

CorrelationMatrix_Functional = RegionConnectivity.filterCorrelationMatrix(CorrelationMatrix_Full, mapping)
rois_fxn = CorrelationMatrix_Functional.index

"""
Atlas data

    :roi_mask
    :DistanceMatrix
    :SizeMatrix
"""
#
# pull_inds = np.array([ 3,  4, 5,  6,  7,  8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23,
#    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49])
# atlas_index = pd.read_csv(os.path.join(analysis_dir, 'data', 'Original_Index_panda.csv')).iloc[pull_inds, :]
#
# # match to fxn roi names
# new_rois = atlas_index.name
# new_rois = [x.split('_R')[0] for x in new_rois]
# new_rois = [x.replace('_', '') for x in new_rois]
# atlas_index.name = new_rois
#
# # load mask brain
# mask_brain = np.asarray(np.squeeze(nib.load(os.path.join(analysis_dir, 'data', 'JFRCtempate2010.mask130819_crop.nii')).get_fdata()), 'uint8')
#
# # get individual roi masks and compute size of each roi (in voxels)
# roi_mask = []
# roi_size = []
# for r in rois_fxn:
#     new_roi_mask = np.zeros_like(mask_brain)
#     target_index = atlas_index[atlas_index.name==r].num.to_numpy()[0]
#     new_roi_mask = mask_brain == target_index
#     roi_mask.append(new_roi_mask)
#     roi_size.append(np.sum(new_roi_mask))
#
# # find center of mass for each roi
# coms = np.vstack([center_of_mass(x) for x in roi_mask])
#
# # calulcate euclidean distance matrix between roi centers of mass
# dist_mat = np.zeros_like(CorrelationMatrix_Functional)
# upper_inds = np.triu_indices(dist_mat.shape[0], k=1)
# dist_mat[upper_inds] = pdist(coms)
# dist_mat += dist_mat.T # symmetrize to fill in below diagonal
#
# DistanceMatrix = pd.DataFrame(data=dist_mat, index=rois_fxn, columns=rois_fxn)
#
# # geometric mean of the sizes for each pair of ROIs
# sz_mat = np.sqrt(np.outer(np.array(roi_size), np.array(roi_size)))
#
# SizeMatrix = pd.DataFrame(data=sz_mat, index=rois_fxn, columns=rois_fxn)

# %%

"""
Anatomical connectivity matrix

    :ConnectivityMatrix
    :ConnectivityMatrix_Symmetrized
"""
usemat = 'count_all'
correct_for_completeness = False

WeakConnections = pd.read_pickle(os.path.join(analysis_dir,'data', 'WeakConnections_computed_20200507.pkl'))
MediumConnections = pd.read_pickle(os.path.join(analysis_dir,'data', 'MediumConnections_computed_20200507.pkl'))
StrongConnections = pd.read_pickle(os.path.join(analysis_dir,'data', 'StrongConnections_computed_20200507.pkl'))

if usemat == 'count_all':
    conn_mat = WeakConnections + MediumConnections + StrongConnections

elif usemat == 'weight_computed':
    conn_mat = pd.read_pickle(os.path.join(analysis_dir,'data', 'Connectivity_computed_20200507.pkl'))

elif usemat == 'ct_precomputed':
    conn_mat = RegionConnectivity.getPrecomputedConnectivityMatrix(neuprint_client, mapping, metric='count', diagonal='nan')

if correct_for_completeness:
    conn_mat = conn_mat / roi_completeness['completeness'][:, None]


# drop '(R)' from roi names
roi_names = conn_mat.index
# roi_names = [x.split('(')[0] for x in rois]
# set diag to nan
tmp_mat = conn_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)

ConnectivityMatrix = pd.DataFrame(data=tmp_mat, index=roi_names, columns=roi_names)
# symmetrize anatomical adjacency matrix by just adding it to its transpose and dividing by 2. Ignores directionality
ConnectivityMatrix_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=roi_names, columns=roi_names)


# %% FIGURE 1: Correlation between anatomical and functional connectivty matrices

fig1_0, ax = plt.subplots(1, 2, figsize=(16,8))

ConnectivityMatrix.index
CorrelationMatrix_Functional.index

sns.heatmap(ConnectivityMatrix, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Connection strength (cells)', 'shrink': .8}, cmap="cividis", rasterized=True)
ax[0].set_xlabel('Target');
ax[0].set_ylabel('Source');
ax[0].set_aspect('equal')
ax[0].set_title('Anatomical');

sns.heatmap(CorrelationMatrix_Functional, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True)
ax[1].set_aspect('equal')
ax[1].set_title('Functional');


upper_inds = np.triu_indices(ConnectivityMatrix.shape[0], k=1) # k=1 excludes main diagonal

anatomical_adjacency = ConnectivityMatrix_Symmetrized.to_numpy().copy()[upper_inds]
functional_adjacency = CorrelationMatrix_Functional.to_numpy().copy()[upper_inds]

adj_DF = pd.DataFrame(data = np.vstack([anatomical_adjacency, functional_adjacency]).T, columns = ['Anatomical', 'Functional'])

r, p = pearsonr(anatomical_adjacency, functional_adjacency)
coef = np.polyfit(anatomical_adjacency, functional_adjacency, 1)
linfit = np.poly1d(coef)


# fig1_1 = plt.figure(figsize=(8,8));
g = sns.JointGrid(x="Anatomical", y="Functional", data=adj_DF)
g = g.set_axis_labels('Connection strength (cells)', 'Functional correlation (z)')
g = g.plot_joint(sns.scatterplot, color='k', alpha=0.5)
g = g.plot_marginals(sns.distplot, kde=True, color='k')
g.ax_marg_y.set_axis_off()
g.ax_marg_x.set_axis_off()
xx = [adj_DF['Anatomical'].min(), adj_DF['Anatomical'].max()]
g.ax_joint.plot(xx, linfit(xx), 'k-')
g.ax_joint.annotate('r = {:.3f}'.format(r), xy=(2000, 1.5))

fig1_1 = plt.gcf()

# TODO: re-do anatomy roi names
#
# # log transform anatomical connectivity values
# keep_inds = np.where(anatomical_adjacency > 0)[0] # toss zero connection values
# functional_adjacency_no0= functional_adjacency[keep_inds]
# anatomical_adjacency_log = np.log10(anatomical_adjacency[keep_inds])
#
# adj_log_DF = pd.DataFrame(data = np.vstack([anatomical_adjacency_log, functional_adjacency_no0]).T, columns = ['Anatomical', 'Functional'])
#
# r_log, p_log = pearsonr(anatomical_adjacency_log, functional_adjacency_no0)
# coef = np.polyfit(anatomical_adjacency_log, functional_adjacency_no0, 1)
# linfit_log = np.poly1d(coef)

# fig1_1, ax = plt.subplots(1, 2, figsize=(12, 6))

# ax[0].plot(adj_DF['Anatomical'], adj_DF['Functional'], 'ko', alpha=0.6)
# xx = [adj_DF['Anatomical'].min(), adj_DF['Anatomical'].max()]
# ax[0].plot(xx, linfit(xx), 'k-')
# ax[0].set_title('r = {:.3f}'.format(r));
# ax[0].set_ylabel('Functional correlation (z)');
# ax[0].set_xlabel('Connection strength (cells)')

# ax[1].plot(10**adj_log_DF['Anatomical'], adj_log_DF['Functional'], 'ko', alpha=0.5)
# ax[1].set_xscale('log')
# xx = np.linspace(adj_log_DF['Anatomical'].min(), adj_log_DF['Anatomical'].max(), 20)
# ax[1].plot(10**xx, linfit_log(xx), LineWidth=3, color='k', marker=None)
# ax[1].set_title('r = {:.3f}'.format(r_log));
# ax[1].set_ylabel('Functional correlation (z)');
# ax[1].set_xlabel('Connection strength (cells)');

# %% FIGURE 2


upper_inds = np.triu_indices(DistanceMatrix.to_numpy().shape[0], k=1) #k=1 excludes main diagonal

fig2_0, ax = plt.subplots(1, 3, figsize=(18,6))

dist = DistanceMatrix.to_numpy()[upper_inds]
anat = ConnectivityMatrix_Symmetrized.to_numpy()[upper_inds]

# keep_inds = np.where(anat > 0)[0]
# dist = dist[keep_inds]
# anat = np.log10(anat[keep_inds])

r, p = pearsonr(dist, anat)
coef = np.polyfit(dist, anat, 1)
poly1d_fn = np.poly1d(coef)
xx = [dist.min(), dist.max()]
ax[0].scatter(dist, anat, color='k', alpha=0.5)
ax[0].plot(xx, poly1d_fn(xx), LineWidth=2, color='k', marker=None)
ax[0].set_xlabel('Distance between ROIs')
ax[0].set_ylabel('Anatomical connectivity')
ax[0].set_title('r = {:.3f}'.format(r))


fc = CorrelationMatrix_Functional.to_numpy()[upper_inds]
dist = DistanceMatrix.to_numpy()[upper_inds]

r, p = pearsonr(dist, fc)
coef = np.polyfit(dist, fc, 1)
poly1d_fn = np.poly1d(coef)
xx = [dist.min(), dist.max()]
ax[1].scatter(dist, fc, color='k', alpha=0.5)
ax[1].plot(xx, poly1d_fn(xx), LineWidth=2, color='k', marker=None)
ax[1].set_xlabel('Distance between ROIs')
ax[1].set_ylabel('Functional correlation (z)')
ax[1].set_title('r = {:.3f}'.format(r))


residuals = zscore(fc) - zscore(dist)
anat = ConnectivityMatrix_Symmetrized.to_numpy()[upper_inds]
# keep_inds = np.where(anat > 0)[0]
# residuals = residuals[keep_inds]
# anat = np.log10(anat[keep_inds])

r, p = pearsonr(anat, residuals)
coef = np.polyfit(anat, residuals, 1)
poly1d_fn = np.poly1d(coef)
xx = np.linspace(anat.min(), anat.max(), 20)
# ax[2].scatter(10**anat, residuals, color='k', alpha=0.5)
# ax[2].plot(10**xx, poly1d_fn(xx), LineWidth=2, color='k', marker=None)
# ax[2].set_xscale('log')

CompletenessMatrix = pd.DataFrame(data=np.outer(roi_completeness['frac_post'], roi_completeness['frac_pre']), index=ConnectivityMatrix.index, columns=ConnectivityMatrix.index)
comp_score = CompletenessMatrix.to_numpy()[upper_inds]

sc = ax[2].scatter(anat, residuals, alpha=1.0, c=comp_score, cmap="Blues",  vmin=comp_score.min(), vmax=comp_score.max())
ax[2].plot(xx, poly1d_fn(xx), LineWidth=2, marker=None)
ax[2].set_xlabel('Anatomical connectivity')
ax[2].set_ylabel('Residual (zscore)')
ax[2].set_title('r = {:.3f}'.format(r));

cb = fig2_0.colorbar(sc, ax=ax[2])


# %% FIGURE 3

# compute difference matrix using original, asymmetric anatomical connectivity matrix
anatomical_mat = ConnectivityMatrix.to_numpy().copy()
np.fill_diagonal(anatomical_mat, 0)
functional_mat = CorrelationMatrix_Functional.to_numpy().copy()
np.fill_diagonal(functional_mat, 0)

# log transform anatomical connectivity values
keep_inds = np.where(anatomical_mat > 0) # toss zero connection values
functional_adjacency_no0= functional_mat[keep_inds]
anatomical_adjacency_log = np.log10(anatomical_mat[keep_inds])

F_zscore = zscore(functional_adjacency_no0)
A_zscore = zscore(anatomical_adjacency_log)

diff_m = np.zeros_like(ConnectivityMatrix.to_numpy())
diff = A_zscore - F_zscore
diff_m[keep_inds] = diff
DifferenceMatrix = pd.DataFrame(data=diff_m, index=ConnectivityMatrix.index, columns=ConnectivityMatrix.index)

# %% SUPP FIG: does completeness of reconstruction impact

CompletenessMatrix = pd.DataFrame(data=np.outer(roi_completeness['frac_post'], roi_completeness['frac_pre']), index=ConnectivityMatrix.index, columns=ConnectivityMatrix.index)


comp_score = CompletenessMatrix.to_numpy().ravel()
# diff_score = np.abs(DifferenceMatrix.to_numpy().ravel())
diff_score = DifferenceMatrix.to_numpy().ravel()

include_inds = np.where(diff_score != 0)[0]
comp_score  = comp_score[include_inds]
diff_score = diff_score[include_inds]
anat_conn = ConnectivityMatrix.to_numpy().ravel()[include_inds]


r, p = pearsonr(comp_score, diff_score)
r
p

figS1, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].scatter(comp_score, diff_score, marker='o', color='k', alpha=0.5)

ax[0].set_xlabel('Completeness of reconstruction')
ax[0].set_ylabel('abs(Anat. - Fxnal (z-score))')

ax[1].scatter(comp_score, anat_conn, marker='o', color='k', alpha=0.5)

ax[1].set_xlabel('Completeness of reconstruction')
ax[1].set_ylabel('Anat connectivity')

# %% compare clustering of anat and fxnal matrices

# spectral clustering of anatomical matrix
n_clusters = 6

roi_names = ConnectivityMatrix.index

# get anatomical matrix and set diagonal nans to 0
anat_mat = ConnectivityMatrix_Symmetrized.to_numpy().copy()
np.fill_diagonal(anat_mat, 0)
anatomical = pd.DataFrame(data=anat_mat, columns=roi_names, index=roi_names)

# get fxnal matrix and set diagonal nans to 0
fxn_mat = CorrelationMatrix_Functional.to_numpy().copy()
np.fill_diagonal(fxn_mat, 0)
functional = pd.DataFrame(data=fxn_mat, columns=roi_names, index=roi_names)

# cluster anatomical matrix
sc_anatomical = SpectralClustering(n_clusters, affinity='precomputed', n_init=100)
sc_anatomical.fit(anatomical.to_numpy());

# cluster fxnal matrix
sc_functional = SpectralClustering(n_clusters, affinity='precomputed', n_init=100)
sc_functional.fit(functional.to_numpy());


cluster_brain_anat = np.zeros(shape=mask_brain.shape)
cluster_brain_anat[:] = np.nan
cluster_brain_fxn = np.zeros(shape=mask_brain.shape)
cluster_brain_fxn[:] = np.nan
for r_ind, r in enumerate(roi_mask):
    cluster_brain_anat[r] = sc_anatomical.labels_[r_ind]
    cluster_brain_fxn[r] = sc_functional.labels_[r_ind]

# %%
zslices = np.arange(30, 200, 40)
lim = sc.labels_.max()



fh, axes = plt.subplots(5, 2, figsize=(4,8))
for z_ind, z in enumerate(zslices):
    ax = axes[z_ind, 0]
    img = ax.imshow(cluster_brain_anat[:, :, z].T, cmap="tab20", rasterized=True, vmin=0, vmax=lim)
    ax.set_axis_off()
    ax.set_aspect('equal')

    ax = axes[z_ind, 1]
    img = ax.imshow(cluster_brain_fxn[:, :, z].T, cmap="tab20", rasterized=True, vmin=0, vmax=lim)
    ax.set_axis_off()
    ax.set_aspect('equal')

# %%
# spectral clustering of anatomical matrix
n_clusters = 8

roi_names = ConnectivityMatrix.index

# get anatomical matrix and set diagonal nans to 0
anat_mat = ConnectivityMatrix_Symmetrized.to_numpy().copy()
np.fill_diagonal(anat_mat, 0)
anatomical = pd.DataFrame(data=anat_mat, columns=roi_names, index=roi_names)

# get fxnal matrix and set diagonal nans to 0
fxn_mat = CorrelationMatrix_Functional.to_numpy().copy()
np.fill_diagonal(fxn_mat, 0)
functional = pd.DataFrame(data=fxn_mat, columns=roi_names, index=roi_names)

# cluster anatomical matrix
sc = SpectralClustering(n_clusters, affinity='precomputed', n_init=100)
sc.fit(anatomical.to_numpy());
sort_inds = np.argsort(sc.labels_)
sort_keys = list(np.array(roi_names)[sort_inds])
cluster_lines = np.where(np.diff(np.sort(sc.labels_))==1)[0]+1
sorted_anat = pd.DataFrame(data=np.zeros_like(anatomical),columns=sort_keys, index=sort_keys)
sorted_fxn = pd.DataFrame(data=np.zeros_like(functional),columns=sort_keys, index=sort_keys)
sorted_diff = pd.DataFrame(data=np.zeros_like(DifferenceMatrix),columns=sort_keys, index=sort_keys)
for r_ind, r_key in enumerate(sort_keys):
    for c_ind, c_key in enumerate(sort_keys):
        sorted_anat.iloc[r_ind, c_ind]=anatomical.loc[[r_key], [c_key]].to_numpy()
        sorted_fxn.iloc[r_ind, c_ind]=functional.loc[[r_key], [c_key]].to_numpy()
        sorted_diff.iloc[r_ind, c_ind]=DifferenceMatrix.loc[[r_key], [c_key]].to_numpy()


fig3_0, ax = plt.subplots(2, 2, figsize=(16,16))
sns.heatmap(sorted_anat, ax=ax[0, 0], xticklabels=True, cbar_kws={'label': 'Connection strength (cells)', 'shrink': .75}, cmap="cividis", rasterized=True)
for cl in cluster_lines:
    ax[0, 0].vlines(cl, 0, 40, color='w')
    ax[0, 0].hlines(cl, 0, 40, color='w')
ax[0, 0].set_aspect('equal')
ax[0, 0].set_title('Anatomical');

sns.heatmap(sorted_fxn, ax=ax[0, 1], xticklabels=True, cbar_kws={'label': 'Functional correlation (z)', 'shrink': .75}, cmap="cividis", rasterized=True)
for cl in cluster_lines:
    ax[0, 1].vlines(cl, 0, 40, color='w')
    ax[0, 1].hlines(cl, 0, 40, color='w')
ax[0, 1].set_aspect('equal')
ax[0, 1].set_title('Functional');

lim = np.nanmax(np.abs(DifferenceMatrix.to_numpy().ravel()))
ax[1, 0].scatter(A_zscore, F_zscore, alpha=1, c=diff, cmap="RdBu",  vmin=-lim, vmax=lim, edgecolors='k', linewidths=0.5)
ax[1, 0].plot([-3, 3], [-3, 3], 'k-')
ax[1, 0].set_xlabel('Anatomical connectivity (log10, zscore)')
ax[1, 0].set_ylabel('Functional correlation (zscore)');

sns.heatmap(sorted_diff, ax=ax[1, 1], xticklabels=True, cbar_kws={'label': 'Anat - Fxnal connectivity','shrink': .75}, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
for cl in cluster_lines:
    ax[1, 1].vlines(cl, 0, 40, color='k')
    ax[1, 1].hlines(cl, 0, 40, color='k')
ax[1, 1].set_aspect('equal')
ax[1, 1].set_title('Difference');

# %% make brain region map based on clustering results

cmaps = [plt.get_cmap('Greys'), plt.get_cmap('Purples'), plt.get_cmap('Blues'), plt.get_cmap('Greens'), plt.get_cmap('Reds'), plt.get_cmap('Oranges')]
cmaps = cmaps[:n_clusters]
cluster_brain = np.zeros(shape=(mask_brain.shape[0], mask_brain.shape[1], mask_brain.shape[2], 4))
cluster_brain[:] = np.nan
for r_ind, r in enumerate(roi_mask):
    current_cluster = sc.labels_[r_ind]
    place_in_cluster = np.where(np.where(sc.labels_ == current_cluster)[0]==r_ind)[0][0]
    cluster_len = np.sum(sc.labels_ == current_cluster)
    cluster_cmap = cmaps[current_cluster]
    color = cluster_cmap((place_in_cluster+1)/(cluster_len+1))
    cluster_brain[r, :] = color

# %%
zslices = np.arange(30, 200, 20)

fig4_0 = plt.figure(figsize=(12,12))
for z_ind, z in enumerate(zslices):
    ax = fig4_0.add_subplot(3, 3, z_ind+1)
    np.moveaxis(cluster_brain[:, :, z, :], [0, 1, 2], [1, 0, 2])

    img = ax.imshow(np.moveaxis(cluster_brain[:, :, z, :], [0, 1, 2], [1, 0, 2]), rasterized=True)
    ax.set_axis_off()
    ax.set_aspect('equal')


# %%
diff_by_region = DifferenceMatrix.mean()
diff_brain = np.zeros(shape=mask_brain.shape)
diff_brain[:] = np.nan


for r_ind, r in enumerate(roi_mask):
    diff_brain[r] = diff_by_region[r_ind]

# %%
lim = np.nanmax(np.abs(diff_brain))
fh, ax = plt.subplots(1, 1, figsize=(8,8))
ax.imshow(np.nanmean(diff_brain, axis=2).T, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
ax.set_axis_off()
ax.set_aspect('equal')

# %%

zslices = np.arange(30, 200, 20)
lim = np.nanmax(np.abs(diff_brain.ravel()))

fig4_0 = plt.figure(figsize=(12,12))
for z_ind, z in enumerate(zslices):
    ax = fig4_0.add_subplot(3, 3, z_ind+1)
    img = ax.imshow(diff_brain[:, :, z].T, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
    ax.set_axis_off()
    ax.set_aspect('equal')

cb = fig4_0.colorbar(img, ax=ax)
cb.set_label(label='Anat - Fxnal connectivity', weight='bold', color='k')
cb.ax.tick_params(labelsize=12, color='k')
# for l in cb.ax.yaxis.get_ticklabels():
#     l.set_weight("bold")
#     l.set_color("white")
#     l.set_fontsize(12)



# %%

with PdfPages(os.path.join(analysis_dir, 'Lab_mtg_figs_v1.pdf')) as pdf:
    pdf.savefig(fig1_0)
    pdf.savefig(fig1_1)
    pdf.savefig(fig2_0)
    pdf.savefig(fig3_0)
    pdf.savefig(fig4_0)

    d = pdf.infodict()
    d['Title'] = 'SC-FC early figs'
    d['Author'] = 'Max Turner'
    d['ModDate'] = datetime.datetime.today()

plt.close('all')

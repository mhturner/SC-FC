import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, zscore, ttest_rel, kstest, wilcoxon
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns


from region_connectivity import RegionConnectivity



analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC'
data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'

motion_filepaths = glob.glob(os.path.join(data_dir, 'behavior_data', 'motion_events') + '*')

# Load anatomical stuff:
WeakConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'WeakConnections_computed_20200626.pkl'))
MediumConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'MediumConnections_computed_20200626.pkl'))
StrongConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'StrongConnections_computed_20200626.pkl'))
conn_mat = WeakConnections + MediumConnections + StrongConnections
roi_names = conn_mat.index
# set diag to nan
tmp_mat = conn_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)
# symmetrize anatomical adjacency matrix by just adding it to its transpose and dividing by 2. Ignores directionality
ConnectivityCount = pd.DataFrame(data=tmp_mat, index=conn_mat.index, columns=conn_mat.index)
ConnectivityMatrix_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=roi_names, columns=roi_names)
upper_inds = np.triu_indices(ConnectivityMatrix_Symmetrized.shape[0], k=1) # k=1 excludes main diagonal
keep_inds = np.where(ConnectivityMatrix_Symmetrized.to_numpy()[upper_inds] > 0) # for log-transforming anatomical connectivity, toss zero values
anatomical_adjacency = np.log10(ConnectivityMatrix_Symmetrized.to_numpy().copy()[upper_inds][keep_inds])


# %%

# TODO: how to best control for time spent in each state?
shifted_control = False

fs = 1.2 # Hz
cutoff = 0.01 # Hz

eg_ind = 0

fh, ax = plt.subplots(2, 1, figsize=(12, 4))

cmats_full = []
r_full = []

cmats_behaving = []
r_behaving = []

cmats_nonbehaving = []
r_nonbehaving = []

times_behaving = []

for ind, motion_fp in enumerate(motion_filepaths):
    suffix = motion_fp.split('motion_events_')[-1].split('.')[0]
    # load region responses for this fly
    resp_fp = os.path.join(data_dir, 'region_responses', suffix + '.pkl')
    region_responses = RegionConnectivity.getProcessedRegionResponse(resp_fp, cutoff=cutoff, fs=fs)

    # get behavior binary
    is_behaving = RegionConnectivity.getBehavingBinary(motion_fp)
    # filter behaving binary
    is_behaving = RegionConnectivity.trimRegionResponse(suffix, is_behaving)

    if shifted_control:
        is_behaving = np.roll(is_behaving, int(len(is_behaving)/2))

    time_behaving = np.sum(is_behaving)
    times_behaving.append(time_behaving)

    behaving_responses = region_responses.iloc[:, np.where(is_behaving)[0]]
    nonbehaving_responses = region_responses.iloc[:, np.where(np.logical_not(is_behaving))[0]]


    # cmat while behaving
    cmat_behaving = np.corrcoef(behaving_responses)
    np.fill_diagonal(cmat_behaving, 0)
    cmat_behaving = np.arctanh(cmat_behaving)
    functional_adjacency_behaving = cmat_behaving.copy()[upper_inds][keep_inds]
    r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_behaving)
    r_behaving.append(r_new)
    cmats_behaving.append(cmat_behaving)

    # cmat while nonbehaving
    cmat_nonbehaving = np.corrcoef(nonbehaving_responses)
    np.fill_diagonal(cmat_nonbehaving, 0)
    cmat_nonbehaving = np.arctanh(cmat_nonbehaving)
    functional_adjacency_nonbehaving = cmat_nonbehaving.copy()[upper_inds][keep_inds]
    r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_nonbehaving)
    r_nonbehaving.append(r_new)
    cmats_nonbehaving.append(cmat_nonbehaving)

    # plot resp / behavior
    if ind == eg_ind:
        ax[0].plot(is_behaving)
        ax[1].plot(region_responses.T)


_, p = ttest_rel(r_nonbehaving, r_behaving)
fh, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(r_nonbehaving, r_behaving, 'ko')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('Nonbehaving')
ax.set_ylabel('Behaving')
ax.set_title('p = {:.3f}'.format(p))

print((np.array(r_behaving) - np.array(r_nonbehaving)))


# %%

cmat_behaving = pd.DataFrame(data=np.mean(np.stack(cmats_behaving, axis=2), axis=2), index=roi_names, columns=roi_names)
cmat_nonbehaving = pd.DataFrame(data=np.mean(np.stack(cmats_nonbehaving, axis=2), axis=2), index=roi_names, columns=roi_names)
vmin = np.min((cmat_behaving.to_numpy().min(), cmat_nonbehaving.to_numpy().min()))
vmax = np.max((cmat_behaving.to_numpy().max(), cmat_nonbehaving.to_numpy().max()))

fh, ax = plt.subplots(1, 2, figsize=(16, 8))
sns.heatmap(cmat_behaving, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True, vmin=vmin, vmax=vmax)
ax[0].set_aspect('equal')
ax[0].set_title('Behaving')

sns.heatmap(cmat_nonbehaving, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True, vmin=vmin, vmax=vmax)
ax[1].set_aspect('equal')
ax[1].set_title('NonBehaving')


# %%
functional_adjacency_behaving = cmat_behaving.to_numpy().copy()[upper_inds][keep_inds]
functional_adjacency_nonbehaving = cmat_nonbehaving.to_numpy().copy()[upper_inds][keep_inds]

fh, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(anatomical_adjacency, functional_adjacency_behaving, 'ko')
r, _ = pearsonr(anatomical_adjacency, functional_adjacency_behaving)
ax[0].set_title('Behaving, r = {:.3f}'.format(r))
ax[0].set_ylabel('Correlation (z)')
ax[0].set_xlabel('Anatomical connectivity (log10)')

ax[1].plot(anatomical_adjacency, functional_adjacency_nonbehaving, 'ko')
r, _ = pearsonr(anatomical_adjacency, functional_adjacency_nonbehaving)
ax[1].set_title('Nonbehaving, r = {:.3f}'.format(r))
ax[1].set_xlabel('Anatomical connectivity (log10)')

# %%

fh, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(functional_adjacency_behaving, functional_adjacency_nonbehaving, marker='.', color='b', alpha=1.0, LineStyle='None')
ax.plot([-0.2, 1], [-0.2, 1], 'k-')
ax.set_xlabel('Behaving')
ax.set_ylabel('Nonbehaving')


# %%

# %%
mapping = RegionConnectivity.getRoiMapping()
roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')
roi_mask, roi_size = RegionConnectivity.loadAtlasData(atlas_path=atlas_path, roinames_path=roinames_path, mapping=mapping)

DifferenceMatrix = cmat_behaving - cmat_nonbehaving


diff_by_region = DifferenceMatrix.mean()
sort_inds = np.argsort(diff_by_region)
sort_keys = DifferenceMatrix.index[sort_inds]
sorted_diff = pd.DataFrame(data=np.zeros_like(DifferenceMatrix),columns=sort_keys, index=sort_keys)
for r_ind, r_key in enumerate(sort_keys):
    for c_ind, c_key in enumerate(sort_keys):
        sorted_diff.iloc[r_ind, c_ind]=DifferenceMatrix.loc[[r_key], [c_key]].to_numpy()

lim = np.nanmax(np.abs(DifferenceMatrix.to_numpy().ravel()))
fh, ax = plt.subplots(1, 1, figsize=(8,8))
sns.heatmap(sorted_diff, ax=ax, xticklabels=True, cbar_kws={'label': 'Behaving - Nonbehaving','shrink': .75}, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
ax.set_aspect('equal')

diff_by_region = DifferenceMatrix.mean()
diff_brain = np.zeros(shape=roi_mask[0].shape)
diff_brain[:] = np.nan
for r_ind, r in enumerate(roi_mask):
    diff_brain[r] = diff_by_region[r_ind]


zslices = np.arange(5, 65, 12)
lim = np.nanmax(np.abs(diff_brain.ravel()))

fh2 = plt.figure(figsize=(15,3))
for z_ind, z in enumerate(zslices):
    ax = fh2.add_subplot(1, 5, z_ind+1)
    img = ax.imshow(diff_brain[:, :, z].T, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
    ax.set_axis_off()
    ax.set_aspect('equal')

cb = fh2.colorbar(img, ax=ax)
cb.set_label(label='Behaving - Nonbehaving', weight='bold', color='k')
cb.ax.tick_params(labelsize=12, color='k')
# # %%
# fs = 1.2
# cutoff = 0.01
#
# ind = 2
#
# dt = 5
# ws = 100
#
# resp_fp = response_filepaths[ind]
#
# fh, ax = plt.subplots(2, 1, figsize=(16, 8))
#
#
# region_responses_processed = RegionConnectivity.getProcessedRegionResponse(resp_fp, cutoff=cutoff, fs=fs)
# ax[0].plot(region_responses_processed.T)
#
# window_centers = np.arange(ws/2, region_responses_processed.shape[1]-ws/2, dt)
# r = []
# act = []
# cmats = []
# corr = []
# for t in window_centers:
#     correlation_matrix = np.corrcoef(region_responses_processed.to_numpy()[:, int(t-ws/2):int(t+ws/2)])
#     cmats.append(correlation_matrix[upper_inds])
#
#     corr.append(np.mean(correlation_matrix[upper_inds]))
#
#     r_new, _ = pearsonr(anatomical_adjacency, correlation_matrix[upper_inds])
#     r.append(r_new)
#
#     activity = np.var(region_responses_processed.to_numpy()[:, int(t-ws/2):int(t+ws/2)], axis=1) / np.mean(region_responses_processed.to_numpy()[:, int(t-ws/2):int(t+ws/2)], axis=1)
#     act.append(np.mean(activity))
#
# rr, _ = pearsonr(corr, r)
#
# ax[1].plot(window_centers, r, LineStyle='-', c='k', label='SC-FC corr')
# ax[1].plot(window_centers, corr, LineStyle='-', c='r', label='Mean FC')
# # ax[1].set_ylim([0, 1])
# ax[1].set_title('r={:.2f}'.format(rr))
# # ax2 = plt.twinx(ax[1])
# # ax2.plot(window_centers, act, LineStyle='--', label='activity_{}'.format(ws), color='b')
#
#
# cmats = np.vstack(cmats)
#
# # fh.savefig(os.path.join(analysis_dir, 'SCFC_MeanCorr_TimeEvolved.png'))
#
# # %%
# fh, ax = plt.subplots(1,1, figsize=(6,6))
# ax.scatter(corr, r, c='k')
# rr, _ = pearsonr(corr, r)
# ax.set_xlabel('FC corr')
# ax.set_ylabel('SC-FC correlation')
# ax.set_title('r = {}'.format(rr))
#
#
#
# # %%
#
# n_clusters = 3
# kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(cmats)
#
# FC_corr = []
# SC_FC = []
# fh, ax = plt.subplots(1,1, figsize=(6,6))
# for c in range(n_clusters):
#     mean_x = np.mean(np.array(corr)[np.where(kmeans.labels_==c)])
#     err_x = np.std(np.array(corr)[np.where(kmeans.labels_==c)]) / np.sqrt(np.sum(kmeans.labels_==c))
#
#     mean_y = np.mean(np.array(r)[np.where(kmeans.labels_==c)])
#     err_y = np.std(np.array(r)[np.where(kmeans.labels_==c)]) / np.sqrt(np.sum(kmeans.labels_==c))
#     ax.errorbar(mean_x, mean_y, xerr=err_x, yerr=err_y, color='k')
#
# ax.set_xlabel('Mean FC corr')
# ax.set_ylabel('SC-FC corr')

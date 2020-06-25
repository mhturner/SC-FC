import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, zscore
from sklearn.cluster import KMeans
import pandas as pd


from region_connectivity import RegionConnectivity



analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC'
data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'

motion_filepaths = glob.glob(os.path.join(data_dir, 'behavior_data', 'motion_events') + '*')

# Load anatomical stuff:
WeakConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'WeakConnections_computed_20200618.pkl'))
MediumConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'MediumConnections_computed_20200618.pkl'))
StrongConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'StrongConnections_computed_20200618.pkl'))
conn_mat = WeakConnections + MediumConnections + StrongConnections
roi_names = conn_mat.index
# set diag to nan
tmp_mat = conn_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)
# symmetrize anatomical adjacency matrix by just adding it to its transpose and dividing by 2. Ignores directionality
ConnectivityMatrix_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=roi_names, columns=roi_names)
upper_inds = np.triu_indices(ConnectivityMatrix_Symmetrized.shape[0], k=1) # k=1 excludes main diagonal
keep_inds = np.where(ConnectivityMatrix_Symmetrized.to_numpy()[upper_inds] > 0) # for log-transforming anatomical connectivity, toss zero values
anatomical_adjacency = np.log10(ConnectivityMatrix_Symmetrized.to_numpy().copy()[upper_inds][keep_inds])


# %%

fs = 1.2 # Hz
cutoff = 0.01 # Hz

fh, ax = plt.subplots(2, 6, figsize=(24, 8))
ax = ax.ravel()
[x.set_axis_off() for x in ax]

cmats_full = []
r_full = []

cmats_behaving = []
r_behaving = []

cmats_nonbehaving = []
r_nonbehaving = []

for ind, motion_fp in enumerate(motion_filepaths):
    suffix = motion_fp.split('motion_events_')[-1].split('.')[0]

    # get behavior binary
    is_behaving = RegionConnectivity.getBehavingBinary(motion_fp)
    # filter behaving binary
    is_behaving = RegionConnectivity.trimRegionResponse(suffix, is_behaving)

    # load region responses for this fly
    resp_fp = os.path.join(data_dir, 'region_responses', suffix + '.pkl')
    region_responses = RegionConnectivity.getProcessedRegionResponse(resp_fp, cutoff=cutoff, fs=fs)

    # full response trace
    cmat_full = np.corrcoef(region_responses)
    functional_adjacency_full = cmat_full.to_numpy().copy()[upper_inds][keep_inds]
    r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_full)
    r_full.append(r_new)
    cmats_full.append(cmat_full)

    # responses while behaving
    behaving_responses = region_responses[:, np.where(is_behaving)]
    cmat_behaving = np.corrcoef(behaving_responses)
    functional_adjacency_behaving = cmat_behaving.to_numpy().copy()[upper_inds][keep_inds]
    r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_behaving)
    r_behaving.append(r_new)
    cmats_behaving.append(cmat_behaving)

    # responses while nonbehaving
    nonbehaving_responses = region_responses[:, np.where(np.logical_not(is_behaving))]
    cmat_nonbehaving = np.corrcoef(nonbehaving_responses)
    functional_adjacency_nonbehaving = cmat_nonbehaving.to_numpy().copy()[upper_inds][keep_inds]
    r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_nonbehaving)
    r_nonbehaving.append(r_new)
    cmats_nonbehaving.append(cmat_behaving)

    # plot resp / behavior
    ax[0, ind].plot(is_behaving)
    ax[1, ind].plot(region_responses)


fh, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(r_nonbehaving, r_behaving, 'ko')
ax.plot([0, 1], [0, 1], 'k--')

#
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

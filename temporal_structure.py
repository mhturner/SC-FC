import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, zscore
import pandas as pd

from region_connectivity import RegionConnectivity



analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC'
data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'

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
anatomical_adjacency = ConnectivityMatrix_Symmetrized.to_numpy().copy()[upper_inds]

# %%

fs = 1.2
cutoff = 0.01
t_start = 100
t_end = None

response_filepaths = glob.glob(os.path.join(data_dir, 'region_responses') + '/' + '*.pkl')
fh, ax = plt.subplots(5, 3, figsize=(24, 8))
ax = ax.ravel()
[x.set_axis_off() for x in ax]

for ind in range(len(response_filepaths)):


    region_response = pd.read_pickle(response_filepaths[ind])
    region_response = RegionConnectivity.filterRegionResponse(region_response, cutoff=cutoff, fs=fs, t_start=t_start, t_end=t_end)

    correlation_matrix = np.corrcoef(region_response)


    ax[ind].plot(region_response.T);

    r, _ = pearsonr(anatomical_adjacency, correlation_matrix[upper_inds])
    ax[ind].set_title('r={:.2f}'.format(r))



fh.savefig(os.path.join(analysis_dir, 'HPfiltering_{}.png'.format(fs)))
# %%
fs = 2
cutoff = 0.025
t_start = 50
t_end = None

ind = 9
dt = 5
ws = 50

resp_fp = response_filepaths[ind]

fh, ax = plt.subplots(2, 1, figsize=(16, 8))


region_responses = pd.read_pickle(resp_fp)
region_responses = RegionConnectivity.filterRegionResponse(region_responses, cutoff=cutoff, fs=fs, t_start=t_start, t_end=t_end)
ax[0].plot(region_responses.T)

window_centers = np.arange(ws/2, region_responses.shape[1]-ws/2, dt)
r = []
act = []
cmats = []
corr = []
for t in window_centers:
    correlation_matrix = np.corrcoef(region_responses.to_numpy()[:, int(t-ws/2):int(t+ws/2)])
    cmats.append(correlation_matrix[upper_inds])

    corr.append(np.mean(correlation_matrix[upper_inds]))

    r_new, _ = pearsonr(anatomical_adjacency, correlation_matrix[upper_inds])
    r.append(r_new)

    activity = np.var(region_responses.to_numpy()[:, int(t-ws/2):int(t+ws/2)], axis=1) / np.mean(region_responses.to_numpy()[:, int(t-ws/2):int(t+ws/2)], axis=1)
    act.append(np.mean(activity))

rr, _ = pearsonr(corr, r)

ax[1].plot(window_centers, r, LineStyle='-', c='k', label='SC-FC corr')
ax[1].plot(window_centers, corr, LineStyle='-', c='r', label='Mean FC')
ax[1].set_ylim([0, 1])
ax[1].set_title('r={:.2f}'.format(rr))
ax2 = plt.twinx(ax[1])
ax2.plot(window_centers, act, LineStyle='--', label='activity_{}'.format(ws), color='b')


cmats = np.vstack(cmats)

# fh.savefig(os.path.join(analysis_dir, 'SCFC_MeanCorr_TimeEvolved.png'))

# %%
fh, ax = plt.subplots(1,1, figsize=(6,6))
ax.scatter(corr, r, c='k')
rr, _ = pearsonr(corr, r)
ax.set_xlabel('FC corr')
ax.set_ylabel('SC-FC correlation')
ax.set_title('r = {}'.format(rr))



# %%

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(cmats)

FC_corr = []
SC_FC = []
fh, ax = plt.subplots(1,1, figsize=(6,6))
for c in range(n_clusters):
    mean_x = np.mean(np.array(corr)[np.where(kmeans.labels_==c)])
    err_x = np.std(np.array(corr)[np.where(kmeans.labels_==c)]) / np.sqrt(np.sum(kmeans.labels_==c))

    mean_y = np.mean(np.array(r)[np.where(kmeans.labels_==c)])
    err_y = np.std(np.array(r)[np.where(kmeans.labels_==c)]) / np.sqrt(np.sum(kmeans.labels_==c))
    ax.errorbar(mean_x, mean_y, xerr=err_x, yerr=err_y, color='k')

ax.set_xlabel('Mean FC corr')
ax.set_ylabel('SC-FC corr')

# %%
reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, metric='correlation')
embedding = reducer.fit_transform(cmats)
embedding.shape
plt.scatter(embedding[:, 0], embedding[:, 1], c=kmeans.labels_)

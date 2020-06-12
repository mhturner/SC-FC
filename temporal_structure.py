import os
import glob
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr
# from region_connectivity import RegionConnectivity
# import pandas as pd
# from matplotlib.backends.backend_pdf import PdfPages
# import datetime
# import time
# import socket
# import umap
# from sklearn.cluster import KMeans

analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC'
data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'

# Load anatomical stuff:
WeakConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'WeakConnections_computed_20200507.pkl'))
MediumConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'MediumConnections_computed_20200507.pkl'))
StrongConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'StrongConnections_computed_20200507.pkl'))
conn_mat = WeakConnections + MediumConnections + StrongConnections
roi_names = conn_mat.index
# set diag to nan
tmp_mat = conn_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)
# symmetrize anatomical adjacency matrix by just adding it to its transpose and dividing by 2. Ignores directionality
ConnectivityMatrix_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=roi_names, columns=roi_names)
upper_inds = np.triu_indices(ConnectivityMatrix_Symmetrized.shape[0], k=1) # k=1 excludes main diagonal
anatomical_adjacency = ConnectivityMatrix_Symmetrized.to_numpy().copy()[upper_inds]


response_filepaths = glob.glob(os.path.join(data_dir, 'region_responses') + '/' + '*.pkl')
fp = response_filepaths[0]
# for fp in response_filepaths:

region_responses = pd.read_pickle(fp)

# %%

dt = 5
window_sizes = [100]
fh, ax = plt.subplots(1,1, figsize=(16,8))
ax2 = plt.twinx(ax)
for ws in window_sizes:
    window_centers = np.arange(ws/2, region_responses.shape[1]-ws/2, dt)
    r = []
    act = []
    cmats = []
    corr = []
    for t in window_centers:
        correlation_matrix = np.corrcoef(region_responses[:, int(t-ws/2):int(t+ws/2)])
        cmats.append(correlation_matrix[upper_inds])

        corr.append(np.mean(correlation_matrix[upper_inds]))

        r_new, _ = pearsonr(anatomical_adjacency, correlation_matrix[upper_inds])
        r.append(r_new)

        activity = np.var(region_responses[:, int(t-ws/2):int(t+ws/2)], axis=1) / np.mean(region_responses[:, int(t-ws/2):int(t+ws/2)], axis=1)
        act.append(np.mean(activity))

    ax.plot(window_centers, r, LineStyle='-', c='k', label='SC-FC corr')
    ax2.plot(window_centers, corr, LineStyle='-', c='r', label='Mean FC')
    # ax2.plot(window_centers, act, LineStyle='--', label='activity_{}'.format(ws), color='b')

ax.set_ylabel('SC-FC correlation')
ax2.set_ylabel('Mean FC')
fh.legend()
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

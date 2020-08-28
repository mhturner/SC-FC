import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import os
from scipy.stats import zscore
import pandas as pd
import seaborn as sns

from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
import matplotlib
from matplotlib import rcParams
rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})

data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC'

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token=bridge.getNeuprintToken())

# Get FunctionalConnectivity object
FC = functional_connectivity.FunctionalConnectivity(data_dir=data_dir, fs=1.2, cutoff=0.01, mapping=bridge.getRoiMapping())

# Get AnatomicalConnectivity object
AC = anatomical_connectivity.AnatomicalConnectivity(data_dir=data_dir, neuprint_client=neuprint_client, mapping=bridge.getRoiMapping())

plot_colors = plt.get_cmap('tab10')(np.arange(8)/8)

# %%

adjacency_anat = AC.getConnectivityMatrix('CellCount', symmetrize=True, diag=0)

# %%
from sklearn.decomposition import PCA
from scipy import linalg
import pandas as pd


U, s, Vh = linalg.svd(adjacency_anat.T, full_matrices=False)

U.shape
s.shape
Vh.shape


fh, ax = plt.subplots(3, 1, figsize=(9, 6))
ax[0].plot(s, 'k-o')
for m in range(2):
    ax[1].plot(U[:, m])


modes = pd.DataFrame(data=U, columns=AC.rois)

# %%

import seaborn as sns

fh, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.heatmap(modes, ax=ax, yticklabels=True, xticklabels=True, cmap="cividis", rasterized=True)


# %%
import umap

reducer = umap.UMAP(
                    n_neighbors=10,
                    min_dist=0.1,
                    n_components=3,
                    metric='correlation'
                    )

embedding = reducer.fit_transform(adjacency_anat)

# %%
from scipy.spatial.distance import pdist

fc = FC.CorrelationMatrix.to_numpy()[FC.upper_inds]

c = np.nanmean(FC.CorrelationMatrix.to_numpy(), axis=0)

fh, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=c)
for r_ind, r in enumerate(AC.rois):
    ax.annotate(r, (embedding[r_ind, 0], embedding[r_ind, 1]))



# %%
D = pdist(embedding)
plt.plot(D, fc, 'ko')
#%%
import networkx as nx
from node2vec import Node2Vec
import umap
from scipy.spatial.distance import pdist

fh, ax = plt.subplots(1, 2, figsize=(12, 6))
for i in range(2):
    if i == 0:
        adj = AC.getConnectivityMatrix('CellCount', symmetrize=True, diag=0).to_numpy()
        G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)

    elif i == 1:
        adj = FC.CorrelationMatrix.to_numpy()
        adj[adj<0] = 0
        np.fill_diagonal(adj, 0)
        G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)

    n2v = Node2Vec(graph=G, walk_length=20, num_walks=100)

    w2v = n2v.fit()

    embedding_w2v = np.vstack([np.array(w2v[str(u)]) for u in sorted(G.nodes)])

    reducer = umap.UMAP(
                        n_neighbors=10,
                        min_dist=0.1,
                        n_components=2,
                        metric='correlation'
                        )

    embedding_umap = reducer.fit_transform(embedding_w2v)


    fc = FC.CorrelationMatrix.to_numpy()[FC.upper_inds]

    c = np.nanmean(FC.CorrelationMatrix.to_numpy(), axis=0)

    ax[i].scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=c)
    for r_ind, r in enumerate(AC.rois):
        ax[i].annotate(r, (embedding_umap[r_ind, 0], embedding_umap[r_ind, 1]))


# %%
# ANAT
adj = AC.getConnectivityMatrix('CellCount', symmetrize=True, diag=0).to_numpy()
G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
n2v = Node2Vec(graph=G, walk_length=36, num_walks=500)
w2v = n2v.fit()
embedding_w2v = np.vstack([np.array(w2v[str(u)]) for u in sorted(G.nodes)])
reducer = umap.UMAP(
                    n_neighbors=10,
                    min_dist=0.1,
                    n_components=2,
                    metric='correlation'
                    )
embedding_anat = reducer.fit_transform(embedding_w2v)

# FXN
adj = FC.CorrelationMatrix.to_numpy()
adj[adj<0] = 0
np.fill_diagonal(adj, 0)
G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
n2v = Node2Vec(graph=G, walk_length=36, num_walks=500)
w2v = n2v.fit()
embedding_w2v = np.vstack([np.array(w2v[str(u)]) for u in sorted(G.nodes)])
reducer = umap.UMAP(
                    n_neighbors=10,
                    min_dist=0.1,
                    n_components=2,
                    metric='correlation'
                    )
embedding_fxn = reducer.fit_transform(embedding_w2v)

# %%

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
diff_by_region = DifferenceMatrix.mean()
# %%

# # #
c = diff_by_region
fh, ax = plt.subplots(1, 1, figsize=(8,8))
lim = np.nanmax(np.abs(c))
ax.scatter(embedding_anat[:, 0], embedding_anat[:, 1], c=c, cmap="RdBu",  vmin=-lim, vmax=lim,)
for r_ind, r in enumerate(AC.rois):
    ax.annotate(r, (embedding_anat[r_ind, 0], embedding_anat[r_ind, 1]))

# %%
embedding_fxn.shape

fh, ax = plt.subplots(1, 1, figsize=(8,8))
for r_ind, r in enumerate(AC.rois):
    ax.plot([embedding_fxn[r_ind, 0], embedding_anat[r_ind, 0]], [embedding_fxn[r_ind, 1], embedding_anat[r_ind, 1]], 'k-', alpha=0.5)
    ax.plot(embedding_fxn[r_ind, 0], embedding_fxn[r_ind, 1], 'ro')
    ax.plot(embedding_anat[r_ind, 0], embedding_anat[r_ind, 1], 'ko')
    ax.annotate(r, (embedding_anat[r_ind, 0], embedding_anat[r_ind, 1]))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score

analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/hemibrain_analysis/roi_connectivity'

# Conn_df = pd.read_pickle(os.path.join(analysis_dir, 'ConnectivityMatrix_computed_20200415.pkl'))
Conn_df = pd.read_pickle(os.path.join(analysis_dir, 'NeuronCount_computed_20200415.pkl'))

Corr_df = pd.read_pickle(os.path.join(analysis_dir, 'CorrMat_fxnal.pkl'))

dt = 0.01

C = Conn_df.to_numpy()
np.fill_diagonal(C, 0)
C = C / C.ravel().max()
nodes = C.shape[0]
timevector = np.arange(0, 1000, dt)

c_scale = np.sum(C)

# alpha = 1.05

alphas = np.arange(0.045, 0.055, 0.005)
scales = np.arange(0, 1, 0.1)
corrs = pd.DataFrame(np.zeros(shape=(len(scales), len(alphas))), index=scales, columns=alphas)
corrs[:] = np.nan
for alpha in alphas:
    for scale in scales:
        U = np.zeros(shape=(nodes, len(timevector)))
        for t in range(1, len(timevector)):
            A = (1-alpha*c_scale*dt)*np.eye(nodes) + dt * C
            U[:, t] = A @ U[:, t-1] +  c_scale*np.random.normal(loc=0.0, scale=scale, size=nodes)


        # fh1, ax = plt.subplots(1, 1, figsize=(12, 4))
        # ax.plot(timevector, U.T);

        # correlation matrix of model nodes
        if np.abs(U).max() > 10000:
            continue

        pred_corr = np.corrcoef(U)

        if np.any(np.isnan(pred_corr)):
            continue


        np.fill_diagonal(pred_corr, 0)
        # apply fischer transform
        pred_corr = np.arctanh(pred_corr)

        pred_corr_df = pd.DataFrame(data=pred_corr, index=Corr_df.index, columns=Corr_df.columns)

        # compare measured functional and predicted functional correlation matrices
        upper_inds = np.triu_indices(Corr_df.shape[0], k=1) #k=1 excludes main diagonal
        functional_adjacency = Corr_df.to_numpy()[upper_inds]
        pred_functional_adjacency = pred_corr_df.to_numpy()[upper_inds]

        # r, p = pearsonr(pred_functional_adjacency, functional_adjacency)
        r2 = explained_variance_score(functional_adjacency, pred_functional_adjacency)
        corrs.loc[scale, alpha] = r2

# fh2, ax = plt.subplots(1, 2, figsize=(12, 6))
# sns.heatmap(pred_corr_df, ax=ax[0], xticklabels=True, cbar_kws={'label': 'computed'}, cmap="viridis", rasterized=True)
# ax[0].set_aspect('equal')
#
# sns.heatmap(Corr_df, ax=ax[1], xticklabels=True, cbar_kws={'label': 'computed'}, cmap="viridis", rasterized=True)
# ax[1].set_aspect('equal')

# %%
corrs

fh, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.plot(scales, np.nanmean(corrs.to_numpy(), axis=1))
# ax.set_ylim([0, 0.5])

 #%%
fh2, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.heatmap(corrs, ax=ax, xticklabels=True, cbar_kws={'label': 'correlation'}, cmap="viridis", rasterized=True, vmin=0)
ax.set_aspect('equal')
ax.set_xlabel('alpha')
ax.set_ylabel('noise scale');



# %%

peak_alpha = 0.052
scale = 1

# Conn_df = pd.read_pickle(os.path.join(analysis_dir, 'ConnectivityMatrix_computed_20200415.pkl'))
Conn_df = pd.read_pickle(os.path.join(analysis_dir, 'NeuronCount_computed_20200415.pkl'))
Corr_df = pd.read_pickle(os.path.join(analysis_dir, 'CorrMat_fxnal.pkl'))

dt = 0.01

C = Conn_df.to_numpy()
np.fill_diagonal(C, 0)
C = C / C.ravel().max()
nodes = C.shape[0]
timevector = np.arange(0, 1000, dt)

c_scale = np.sum(C)

U = np.zeros(shape=(nodes, len(timevector)))
for t in range(1, len(timevector)):
    A = (1-peak_alpha*c_scale*dt)*np.eye(nodes) + dt * C
    U[:, t] = A @ U[:, t-1] +  c_scale*np.random.normal(loc=0.0, scale=scale, size=nodes)

fh1, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(timevector, U.T);

# correlation matrix of model nodes
pred_corr = np.corrcoef(U)
np.fill_diagonal(pred_corr, 0)
# apply fischer transform
pred_corr = np.arctanh(pred_corr)

pred_corr_df = pd.DataFrame(data=pred_corr, index=Corr_df.index, columns=Corr_df.columns)

# compare measured functional and predicted functional correlation matrices
upper_inds = np.triu_indices(Corr_df.shape[0], k=1) #k=1 excludes main diagonal
functional_adjacency = Corr_df.to_numpy()[upper_inds]
pred_functional_adjacency = pred_corr_df.to_numpy()[upper_inds]

# r, p = pearsonr(pred_functional_adjacency, functional_adjacency)
r2 = explained_variance_score(functional_adjacency, pred_functional_adjacency)

fh2, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(pred_corr_df, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Predicted correlation'}, cmap="viridis", rasterized=True)
ax[0].set_aspect('equal')
ax[0].set_title(r2)

sns.heatmap(Corr_df, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Measured correlation'}, cmap="viridis", rasterized=True)
ax[1].set_aspect('equal')


# %%
from scipy.stats import zscore

adj_mat = Conn_df.to_numpy().copy()
adj_mat_sym = adj_mat + adj_mat.T
upper_inds = np.triu_indices(adj_mat_sym.shape[0], k=1)  # k=1 excludes main diagonal
em_adjacency = adj_mat_sym[upper_inds]
functional_adjacency = Corr_df.to_numpy()[upper_inds]
functional_adjacency_pred = pred_corr_df.to_numpy()[upper_inds]

# cut out zeros
keep_inds = np.where(em_adjacency > 0)[0]
em_adjacency_log = np.log10(em_adjacency[keep_inds])

functional_adjacency = functional_adjacency[keep_inds]
functional_adjacency_pred = functional_adjacency_pred[keep_inds]

functional_adjacency.shape
functional_adjacency_pred.shape

A_zscore = zscore(em_adjacency_log)
F_zscore = zscore(functional_adjacency)
F_zscore_pred = zscore(functional_adjacency_pred)

diff = A_zscore - F_zscore
diff_pred = A_zscore - F_zscore_pred


lim = np.nanmax(np.abs(diff))

fh3, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(A_zscore, F_zscore, alpha=1, c=diff, cmap="RdBu",  vmin=-lim, vmax=lim, edgecolors='k', linewidths=0.5)
ax[0].plot([-3, 3], [-3, 3], 'k-')
ax[0].set_xlabel('Anatomical connectivity count (log10, zscore)')
ax[0].set_ylabel('Functional correlation (zscore)');
ax[0].set_title('Measured correlation')

ax[1].scatter(A_zscore, F_zscore_pred, alpha=1, c=diff_pred, cmap="RdBu",  vmin=-lim, vmax=lim, edgecolors='k', linewidths=0.5)
ax[1].plot([-3, 3], [-3, 3], 'k-')
ax[1].set_xlabel('Anatomical connectivity count (log10, zscore)')
ax[1].set_ylabel('Functional correlation (zscore)');
ax[1].set_title('Model predicted correlation')


# %%

r2 = explained_variance_score(functional_adjacency, functional_adjacency_pred)
fh, ax = plt.subplots(1, 1, figsize=(6,6))
ax.scatter(functional_adjacency, functional_adjacency_pred, alpha=1, color='k')
ax.plot([0, 1.4], [0, 1.4], 'k-')
ax.set_xlabel('Measured correlation')
ax.set_ylabel('Predicted correlation');
ax.set_title(r2);

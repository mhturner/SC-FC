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
from scipy.stats import kstest, lognorm, norm

from region_connectivity import RegionConnectivity


"""
References:
https://connectome-neuprint.github.io/neuprint-python/docs/index.html
https://github.com/connectome-neuprint/neuprint-python

"""

data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/hemibrain_analysis/roi_connectivity'

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.0.1', token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1heHdlbGxob2x0ZXR1cm5lckBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpMHJRX0M4akliX0ZrS2h2OU5DSElsWlpnRDY5YUMtVGdNLWVWM3lRP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzY2MTk1MzcwfQ.Q-57D4tX2sXMjWym2LFhHaUGHgHiUsIM_JI9xekxw_0')
mapping = RegionConnectivity.getRoiMapping()
rois = list(mapping.keys())
rois.sort()

roi_completeness = RegionConnectivity.getRoiCompleteness(neuprint_client, mapping)

# %% LOAD FUNCTIONAL DATA, FILTER IT ACCORDING TO MAPPING, COMPUTE SOME GEOMETRY STUFF
"""
Functional connectivity and atlas data
    :CorrelationMatrix_Functional: Avg across animals, fischer z transformed correlation values
    :DistanceMatrix: distance between centers of mass for each pair of ROIs
    :SizeMatrix: geometric mean of the sizes for each pair of ROIs
"""
roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')
cmat_path = os.path.join(data_dir, 'functional_connectivity', 'full_cmat.txt')
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')

# CorrelationMatrix_Functional = RegionConnectivity.loadFunctionalData(cmat_path=cmat_path, roinames_path=roinames_path, mapping=mapping)
CorrelationMatrix_Functional = pd.read_pickle(os.path.join(data_dir, 'functional_connectivity', 'CorrelationMatrix_Functional_{}.pkl'.format('20200612')))
roi_mask, roi_size = RegionConnectivity.loadAtlasData(atlas_path=atlas_path, roinames_path=roinames_path, mapping=mapping)

# find center of mass for each roi
coms = np.vstack([center_of_mass(x) for x in roi_mask])

# calulcate euclidean distance matrix between roi centers of mass
dist_mat = np.zeros_like(CorrelationMatrix_Functional)
upper_inds = np.triu_indices(dist_mat.shape[0], k=1)
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
WeakConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'WeakConnections_computed_20200507.pkl'))
MediumConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'MediumConnections_computed_20200507.pkl'))
StrongConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'StrongConnections_computed_20200507.pkl'))
conn_mat = WeakConnections + MediumConnections + StrongConnections
# set diag to nan
tmp_mat = conn_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)
ConnectivityCount_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=conn_mat.index, columns=conn_mat.index)
ConnectivityCount = pd.DataFrame(data=tmp_mat, index=conn_mat.index, columns=conn_mat.index)
# - - - - - - - - - - - - - - - - #
# 2) ConnectivityWeight
weight_mat = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'Connectivity_computed_20200507.pkl'))
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
pcweight_mat = RegionConnectivity.getPrecomputedConnectivityMatrix(neuprint_client, mapping, metric='count', diagonal='nan')
tmp_mat = pcweight_mat.to_numpy().copy()
PrecomputedCount_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=pcweight_mat.index, columns=pcweight_mat.index)
PrecomputedCount = pd.DataFrame(data=tmp_mat, index=pcweight_mat.index, columns=pcweight_mat.index)

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

# %%

plt.plot(TwoStep_Symmetrized.to_numpy().ravel(), ConnectivityCount.to_numpy().ravel(),'ko')

# %% Fig1: Distribtution of connection strengths and weight/count relationship
upper_inds = np.triu_indices(ConnectivityCount.shape[0], k=1) # k=1 excludes main diagonal

fig1_0, ax = plt.subplots(1, 2, figsize=(16,4))

 # Toss zero connection values for log transform
keep_inds = np.where(ConnectivityCount_Symmetrized.to_numpy()[upper_inds] > 0)

#  ConnectivityCount:
ct = ConnectivityCount_Symmetrized.to_numpy().copy()[upper_inds][keep_inds]
log_ct = np.log10(ConnectivityCount_Symmetrized.to_numpy().copy()[upper_inds][keep_inds])

val, bin = np.histogram(log_ct, 20, density=True)
ax[0].plot(10**bin[:-1], val, LineWidth=2)
xx = np.linspace(-1, 4, 100)
yy = norm(loc=np.mean(np.log10(ct)), scale=np.std(np.log10(ct))).pdf(xx)
ax[0].plot(10**xx, yy, 'k', alpha=1)

p_ct, _ = kstest(zscore(log_ct), 'norm')
ax[0].set_xlabel('Cell count')
ax[0].set_ylabel('Prob.')
ax[0].set_xscale('log')

# ConnectivityWeight:
wt = ConnectivityWeight_Symmetrized.to_numpy().copy()[upper_inds][keep_inds]
log_wt = np.log10(ConnectivityWeight_Symmetrized.to_numpy().copy()[upper_inds][keep_inds])
val, bin = np.histogram(wt, 30, density=True)

val, bin = np.histogram(log_wt, 30, density=True)
ax[1].plot(10**bin[:-1], val, LineWidth=2)
xx = np.linspace(-1, 7, 100)
yy = norm(loc=np.mean(np.log10(wt)), scale=np.std(np.log10(wt))).pdf(xx)
ax[1].plot(10**xx, yy, 'k', alpha=1)
p_wt, _ = kstest(zscore(log_wt), 'norm')
ax[1].set_xlabel('Connection weight')
ax[1].set_ylabel('Prob.')
ax[1].set_xscale('log')

print('KS test lognormal: Count p = {:.4f}; weight p = {:.4f}'.format(p_ct, p_wt))


fig1_1, ax = plt.subplots(1, figsize=(4,4))
r, p = pearsonr(ConnectivityCount_Symmetrized.to_numpy()[upper_inds], ConnectivityWeight_Symmetrized.to_numpy()[upper_inds])
ax.plot(ConnectivityCount_Symmetrized.to_numpy()[upper_inds], ConnectivityWeight_Symmetrized.to_numpy()[upper_inds], 'ko')
ax.set_title('r = {:.3f}'.format(r));
ax.set_xlabel('Cell count')
ax.set_ylabel('Connection weight')

# %% FIGURE 2: Connectivity matrices and SC-FC correlation
# TODO: add e.g. region traces
fig2_1, ax = plt.subplots(1, 2, figsize=(16,8))

df = np.log10(ConnectivityCount).replace([np.inf, -np.inf], 0)


sns.heatmap(df, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Connection strength (cells)', 'shrink': .8}, cmap="cividis", rasterized=True)
ax[0].set_xlabel('Target');
ax[0].set_ylabel('Source');
ax[0].set_aspect('equal')
ax[0].set_title('Anatomical');

sns.heatmap(CorrelationMatrix_Functional, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True)
ax[1].set_aspect('equal')
ax[1].set_title('Functional');


do_log_transform = True
if do_log_transform:
    keep_inds = np.where(ConnectivityCount_Symmetrized.to_numpy()[upper_inds] > 0) # toss zero connection values

    anatomical_adjacency = np.log10(ConnectivityCount_Symmetrized.to_numpy().copy()[upper_inds][keep_inds])
    functional_adjacency = CorrelationMatrix_Functional.to_numpy().copy()[upper_inds][keep_inds]
else:
    anatomical_adjacency = ConnectivityMatrix_Symmetrized.to_numpy().copy()[upper_inds]
    functional_adjacency = CorrelationMatrix_Functional.to_numpy().copy()[upper_inds]

r, p = pearsonr(anatomical_adjacency, functional_adjacency)
coef = np.polyfit(anatomical_adjacency, functional_adjacency, 1)
linfit = np.poly1d(coef)

fig2_2, ax = plt.subplots(1,1,figsize=(6,6))
ax.scatter(10**anatomical_adjacency, functional_adjacency, color='k')
xx = np.linspace(anatomical_adjacency.min(), anatomical_adjacency.max(), 100)
ax.plot(10**xx, linfit(xx), 'k-')
ax.set_xscale('log')
ax.set_xlabel('Connection strength (cells)')
ax.set_ylabel('Functional correlation (z)')
ax.annotate('r = {:.3f}'.format(r), xy=(1, 1.3));
# %% Other determinants of FC
# TODO: corrs with size, distance etc

# Multiple linear regression model:
from sklearn.linear_model import LinearRegression, Ridge, ARDRegression, BayesianRidge

upper_inds = np.triu_indices(ConnectivityCount_Symmetrized.to_numpy().shape[0], k=1) #k=1 excludes main fill_diagonal
keep_inds = np.where(ConnectivityCount_Symmetrized.to_numpy()[upper_inds] > 0) # toss zero connection values

# regressors:
count = zscore(np.log10(ConnectivityCount_Symmetrized.to_numpy()[upper_inds][keep_inds]))
weight = zscore(np.log10(ConnectivityWeight_Symmetrized.to_numpy()[upper_inds][keep_inds]))
twostep = zscore(np.log10(TwoStep_Symmetrized.to_numpy()[upper_inds][keep_inds]))
size = zscore(SizeMatrix.to_numpy()[upper_inds][keep_inds])
dist = zscore(DistanceMatrix.to_numpy()[upper_inds][keep_inds])

X = np.vstack([count, size, dist]).T

# to predict:
fc = zscore(CorrelationMatrix_Functional.to_numpy()[upper_inds][keep_inds])
# fc = CorrelationMatrix_Functional.to_numpy()[upper_inds][keep_inds]

# linear correlation of each variable with fc
corr_vals = np.array([np.corrcoef(X[:,c],fc)[0,1] for c in range(X.shape[1])])

fig3_0, ax = plt.subplots(1, 1, figsize=(6,6))
regressor = LinearRegression()
regressor.fit(X, fc);
pred = regressor.predict(X)
frac_var = corr_vals * regressor.coef_
print('Count:{:.2f}; size:{:.2f}; Distance:{:.2f}'.format(*frac_var))

ax.plot(pred, fc, 'ko')
ax.plot([-2, 2], [-2, 2], 'k--')
ax.set_title(regressor.score(X, fc));
ax.set_xlabel('Predicted FC')
ax.set_ylabel('Measured FC (z-score)');



# %% Difference matrix

# compute difference matrix using original, asymmetric anatomical connectivity matrix
anatomical_mat = ConnectivityCount.to_numpy().copy()
np.fill_diagonal(anatomical_mat, 0)
functional_mat = CorrelationMatrix_Functional.to_numpy().copy()
np.fill_diagonal(functional_mat, 0)

# log transform anatomical connectivity values
keep_inds = np.where(anatomical_mat > 0) # toss zero connection values
functional_adjacency_no0= functional_mat[keep_inds]
anatomical_adjacency_log = np.log10(anatomical_mat[keep_inds])

F_zscore = zscore(functional_adjacency_no0)
A_zscore = zscore(anatomical_adjacency_log)

diff_m = np.zeros_like(ConnectivityCount.to_numpy())
diff = A_zscore - F_zscore
diff_m[keep_inds] = diff
DifferenceMatrix = pd.DataFrame(data=diff_m, index=ConnectivityCount.index, columns=ConnectivityCount.index)

# %% SUPP FIG: does completeness of reconstruction impact

CompletenessMatrix = pd.DataFrame(data=np.outer(roi_completeness['frac_post'], roi_completeness['frac_pre']), index=ConnectivityCount.index, columns=ConnectivityCount.index)

comp_score = CompletenessMatrix.to_numpy().ravel()
# diff_score = np.abs(DifferenceMatrix.to_numpy().ravel())
diff_score = DifferenceMatrix.to_numpy().ravel()

include_inds = np.where(diff_score != 0)[0]
comp_score  = comp_score[include_inds]
diff_score = diff_score[include_inds]
anat_conn = ConnectivityCount.to_numpy().ravel()[include_inds]


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


# %% sort difference matrix by most to least different rois
diff_by_region = DifferenceMatrix.mean()
sort_inds = np.argsort(diff_by_region)
sort_keys = DifferenceMatrix.index[sort_inds]
sorted_diff = pd.DataFrame(data=np.zeros_like(DifferenceMatrix),columns=sort_keys, index=sort_keys)
for r_ind, r_key in enumerate(sort_keys):
    for c_ind, c_key in enumerate(sort_keys):
        sorted_diff.iloc[r_ind, c_ind]=DifferenceMatrix.loc[[r_key], [c_key]].to_numpy()

fig3_0, ax = plt.subplots(1, 2, figsize=(16,8))
lim = np.nanmax(np.abs(DifferenceMatrix.to_numpy().ravel()))
ax[0].scatter(A_zscore, F_zscore, alpha=1, c=diff, cmap="RdBu",  vmin=-lim, vmax=lim, edgecolors='k', linewidths=0.5)
ax[0].plot([-3, 3], [-3, 3], 'k-')
ax[0].set_xlabel('Anatomical connectivity (log10, zscore)')
ax[0].set_ylabel('Functional correlation (zscore)');

sns.heatmap(sorted_diff, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Anat - Fxnal connectivity','shrink': .75}, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
ax[1].set_aspect('equal')
ax[1].set_title('Difference');


# %%
diff_by_region = DifferenceMatrix.mean()
diff_brain = np.zeros(shape=roi_mask[0].shape)
diff_brain[:] = np.nan
for r_ind, r in enumerate(roi_mask):
    diff_brain[r] = diff_by_region[r_ind]

# %%

zslices = np.arange(5, 65, 12)
lim = np.nanmax(np.abs(diff_brain.ravel()))

fig4_0 = plt.figure(figsize=(15,3))
for z_ind, z in enumerate(zslices):
    ax = fig4_0.add_subplot(1, 5, z_ind+1)
    img = ax.imshow(diff_brain[:, :, z].T, cmap="RdBu", rasterized=True, vmin=-lim, vmax=lim)
    ax.set_axis_off()
    ax.set_aspect('equal')

cb = fig4_0.colorbar(img, ax=ax)
cb.set_label(label='Anat - Fxnal connectivity', weight='bold', color='k')
cb.ax.tick_params(labelsize=12, color='k')




# %%

with PdfPages(os.path.join(analysis_dir, 'SC_FC_figs.pdf')) as pdf:
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

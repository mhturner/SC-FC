import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, zscore, ttest_rel, kstest, wilcoxon
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
rcParams['svg.fonttype'] = 'none'

from scfc import functional_connectivity


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

eg_ind = 3

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
    region_responses = functional_connectivity.getProcessedRegionResponse(resp_fp, cutoff=cutoff, fs=fs)

    # get behavior binary
    is_behaving = functional_connectivity.getBehavingBinary(motion_fp)
    # filter behaving binary
    is_behaving = functional_connectivity.trimRegionResponse(suffix, is_behaving)

    if shifted_control:
        is_behaving = np.roll(is_behaving, int(len(is_behaving)/2))

    time_behaving = np.sum(is_behaving)
    times_behaving.append(time_behaving)

    behaving_responses = region_responses.iloc[:, np.where(is_behaving)[0]]
    nonbehaving_responses = region_responses.iloc[:, np.where(np.logical_not(is_behaving))[0]]


    # cmat while behaving
    cmat_behaving = np.corrcoef(behaving_responses)
    np.fill_diagonal(cmat_behaving, np.nan)
    cmat_behaving = np.arctanh(cmat_behaving)
    functional_adjacency_behaving = cmat_behaving.copy()[upper_inds][keep_inds]
    r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_behaving)
    r_behaving.append(r_new)
    cmats_behaving.append(cmat_behaving)

    # cmat while nonbehaving
    cmat_nonbehaving = np.corrcoef(nonbehaving_responses)
    np.fill_diagonal(cmat_nonbehaving, np.nan)
    cmat_nonbehaving = np.arctanh(cmat_nonbehaving)
    functional_adjacency_nonbehaving = cmat_nonbehaving.copy()[upper_inds][keep_inds]
    r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_nonbehaving)
    r_nonbehaving.append(r_new)
    cmats_nonbehaving.append(cmat_nonbehaving)

    # plot resp / behavior
    if ind == eg_ind:
        fig1, ax = plt.subplots(1, 1, figsize=(8, 4))
        file_id = resp_fp.split('/')[-1].replace('.pkl', '')
        region_response = pd.read_pickle(resp_fp)
        # convert to dF/F
        dff = (region_response.to_numpy() - np.mean(region_response.to_numpy(), axis=1)[:, None]) / np.mean(region_response.to_numpy(), axis=1)[:, None]

        # trim and filter
        resp = functional_connectivity.filterRegionResponse(dff, cutoff=cutoff, fs=fs)
        resp = functional_connectivity.trimRegionResponse(file_id, resp)
        region_dff = pd.DataFrame(data=resp, index=region_response.index)

        st = 200
        eg_show = 600
        time_vec = np.arange(0, eg_show) / fs
        yarr = np.vstack((is_behaving[st:(st+eg_show)],))

        ax.imshow(yarr, extent=(min(time_vec), max(time_vec), -0.3, 0.5), cmap='binary', clim=[0, 2], interpolation='nearest')
        ax.plot(time_vec, region_dff.iloc[:, st:(st+eg_show)].T, alpha=0.4)
        ax.set_aspect(100)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('dF/F')

_, p = ttest_rel(r_nonbehaving, r_behaving)
fig2, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(r_nonbehaving, r_behaving, 'ko')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('Nonbehaving')
ax.set_ylabel('Behaving')
ax.set_title('p = {:.3f}'.format(p))

print((np.array(r_behaving) - np.array(r_nonbehaving)))


# %%

cmat_behaving = pd.DataFrame(data=np.mean(np.stack(cmats_behaving, axis=2), axis=2), index=roi_names, columns=roi_names)
cmat_nonbehaving = pd.DataFrame(data=np.mean(np.stack(cmats_nonbehaving, axis=2), axis=2), index=roi_names, columns=roi_names)
vmin = np.nanmin((np.nanmin(cmat_behaving.to_numpy()), np.nanmin(cmat_nonbehaving.to_numpy())))
vmax = np.nanmax((np.nanmax(cmat_behaving.to_numpy()), np.nanmax(cmat_nonbehaving.to_numpy())))
cmat_behaving
fig3, ax = plt.subplots(1, 2, figsize=(16, 8))
sns.heatmap(cmat_behaving, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True, vmin=vmin, vmax=vmax)
ax[0].set_aspect('equal')
ax[0].set_title('Behaving')

sns.heatmap(cmat_nonbehaving, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True, vmin=vmin, vmax=vmax)
ax[1].set_aspect('equal')
ax[1].set_title('NonBehaving')


# %%
functional_adjacency_behaving = cmat_behaving.to_numpy().copy()[upper_inds][keep_inds]
functional_adjacency_nonbehaving = cmat_nonbehaving.to_numpy().copy()[upper_inds][keep_inds]

fig4, ax = plt.subplots(1, 2, figsize=(8, 4))
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

fig5, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(functional_adjacency_behaving, functional_adjacency_nonbehaving, marker='.', color='b', alpha=1.0, LineStyle='None')
ax.plot([-0.2, 1], [-0.2, 1], 'k-')
ax.set_xlabel('Behaving')
ax.set_ylabel('Nonbehaving')
functional_adjacency_behaving.shape

# %% save figs

figs_to_save = [fig1, fig2, fig3, fig4, fig5]
for f_ind, fh in enumerate(figs_to_save):
    fh.savefig(os.path.join(analysis_dir, 'figpanels', 'BehaviorFig{}.svg'.format(f_ind)))

# %%
mapping = functional_connectivity.getRoiMapping()
roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')
roi_mask, roi_size = functional_connectivity.loadAtlasData(atlas_path=atlas_path, roinames_path=roinames_path, mapping=mapping)

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

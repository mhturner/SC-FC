import os
import nibabel as nib
import numpy as np
import ants
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from region_connectivity import RegionConnectivity
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import time
import socket


analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/hemibrain_analysis/roi_connectivity'

if socket.gethostname() == 'max-laptop':
    analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/hemibrain_analysis/roi_connectivity'
    data_dir = '/home/mhturner/CurrentData/resting_state'
elif 'sh' in socket.gethostname():
    analysis_dir = '/oak/stanford/groups/trc/data/Max/Analysis/Hemibrain'
    data_dir = '/oak/stanford/groups/trc/data/Max/Analysis/resting_state'

fly_id = 'fly2'

atlas_brain = np.asarray(ants.image_read(os.path.join(data_dir, fly_id, 'vfb_68_Original.nii.gz')).numpy(), 'uint8')
functional_brain = ants.image_read(os.path.join(data_dir, fly_id, 'func_volreg_trim.nii.gz'))

# %% filter to select rois of interest from mapping
mapping = RegionConnectivity.getRoiMapping()
CorrelationMatrix_Full = RegionConnectivity.loadFunctionalData(data_dir=data_dir)
roi_mask, roi_size = RegionConnectivity.loadAtlasData(data_dir=data_dir)

_, pull_inds = RegionConnectivity.filterFunctionalData(CorrelationMatrix_Full, mapping)
roi_mask = [roi_mask[x] for x in pull_inds]
roi_size = [roi_size[x] for x in pull_inds]

# %%
"""
Fig1: Correlation with full fxnal correlation matrix as a function of restricted region size

"""
t0 = time.time()

# get full cmat
full_responses = []
for r_ind, mask in enumerate(roi_mask):
    full_responses.append(np.mean(functional_brain[mask, :], axis=0))
cmat_full = np.corrcoef(np.vstack(full_responses))

# cycle through subsampled region sizes
voxel_numbers = np.logspace(1, 4.4, 10)
C_mats = []
for num_vox in voxel_numbers:
    roi_resp = []
    for r_ind, mask in enumerate(roi_mask):
        mask_size = np.sum(mask)
        if mask_size >= num_vox:
            pull_inds = np.random.choice(np.arange(roi_size[r_ind]), size=int(num_vox), replace=False)
            roi_resp.append(np.mean(functional_brain[mask, :][pull_inds,:], axis=0))
        else: # not enough voxels in mask, so just take the whole region
            roi_resp.append(np.mean(functional_brain[mask, :], axis=0))

    C = np.corrcoef(np.vstack(roi_resp))
    C_mats.append(C)

# compute correlation of each subsampled cmat with full cmat
upper_inds = np.triu_indices(C_mats[-1].shape[0], k=1) #k=1 excludes main diagonal
corr = []
for ind, num_vox in enumerate(voxel_numbers):
    r, p = pearsonr(C_mats[ind][upper_inds], cmat_full[upper_inds])
    corr.append(r)

# plot
fh1, ax1 = plt.subplots(1, 1, figsize=(6,6))
ax1.plot(voxel_numbers, corr, 'k-o')
ax1.set_xlabel('Region size (voxels)')
ax1.set_ylabel('Correlation with full-region functional connectivity'.format(voxel_numbers[-1]))
ax1.set_ylim([0, 1.05])
ax1.set_xscale('log')

bins = np.arange(np.floor(np.min(roi_size)), np.ceil(np.max(roi_size)))
values, base = np.histogram(roi_size, bins=bins, density=True)
cumulative = np.cumsum(values)

ax2 = ax1.twinx()
ax2.plot(base[:-1], cumulative)
ax2.set_ylabel('Cumulative fraction')
ax2.set_ylim([0, 1.05]);

print('Finished F1 (time = {:.1f} sec)'.format(time.time()-t0))

# %%
"""
Fig 2: subsample set fractions of all regions and compute correlation to full region response

"""
t0 = time.time()

# get full region response
full_responses = []
for r_ind, mask in enumerate(roi_mask):
    full_responses.append(np.mean(functional_brain[mask, :], axis=0))
full_responses = np.vstack(full_responses)

# get subsampled region responses and compute correlation with full response
test_fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
region_corrs = []
for r_ind, mask in enumerate(roi_mask):
    test_sizes = [np.int(np.ceil(roi_size[r_ind] * x)) for x in test_fractions]
    new_rs = []
    for ts in test_sizes:
        pull_inds = np.random.choice(np.arange(roi_size[r_ind]), size=ts, replace=False)
        sub_resp = np.mean(functional_brain[mask, :][pull_inds,:], axis=0)
        r, _ = pearsonr(sub_resp, full_responses[r_ind,:])
        new_rs.append(r)
    region_corrs.append(new_rs)

region_corrs = np.vstack(region_corrs)

# plot
fh2, ax = plt.subplots(1, 1, figsize=(5,5))
ax.plot(test_fractions, region_corrs.T, LineStyle='-', Marker='.', alpha=0.2)
ax.plot(test_fractions, np.mean(region_corrs, axis=0), LineStyle='-', Marker='o', Color='k', LineWidth=2)
ax.set_ylim([0, 1.1])
ax.set_xlabel('Fraction of voxels used')
ax.set_ylabel('Corr. with full region response');

print('Finished F2 (time = {:.1f} sec)'.format(time.time()-t0))



# %% calculate corr with anatomy for subsampled regions
"""
Fig 3: calculate SC-FC correlation for subsampled regions, bootstrap over some iterations
"""
t0 = time.time()

# first get anatomical matrix
WeakConnections = pd.read_pickle(os.path.join(analysis_dir,'data', 'WeakConnections_computed_20200507.pkl'))
MediumConnections = pd.read_pickle(os.path.join(analysis_dir,'data', 'MediumConnections_computed_20200507.pkl'))
StrongConnections = pd.read_pickle(os.path.join(analysis_dir,'data', 'StrongConnections_computed_20200507.pkl'))
conn_mat = WeakConnections + MediumConnections + StrongConnections
roi_names = conn_mat.index
# set diag to nan
tmp_mat = conn_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)
# symmetrize anatomical adjacency matrix by just adding it to its transpose and dividing by 2. Ignores directionality
ConnectivityMatrix_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=roi_names, columns=roi_names)
upper_inds = np.triu_indices(ConnectivityMatrix_Symmetrized.shape[0], k=1) # k=1 excludes main diagonal
anatomical_adjacency = ConnectivityMatrix_Symmetrized.to_numpy().copy()[upper_inds]

# get full region corr matrix
full_responses = []
for r_ind, mask in enumerate(roi_mask):
    full_responses.append(np.mean(functional_brain[mask, :], axis=0))
full_responses = np.vstack(full_responses)
CorrelationMatrix_Full = np.corrcoef(np.vstack(full_responses))
functional_adjacency_full = CorrelationMatrix_Full[upper_inds]
r_full, _ = pearsonr(anatomical_adjacency, functional_adjacency_full)

# now for subsampled regions: compute SC-FC corr
subsampled_sizes = np.logspace(1, 4.4, 3) # voxels
n_iter = 1 # num iterations for randomly subsampling regions

r_subsampled = []
for subsampled_size in subsampled_sizes:
    r_iter = []
    for n in range(n_iter):
        roi_resp = []
        for r_ind, mask in enumerate(roi_mask):
            mask_size = np.sum(mask)
            if mask_size >= subsampled_size:
                pull_inds = np.random.choice(np.arange(roi_size[r_ind]), size=int(subsampled_size), replace=False)
                roi_resp.append(np.mean(functional_brain[mask, :][pull_inds,:], axis=0))
            else: # not enough voxels in mask, so just take the whole region
                roi_resp.append(np.mean(functional_brain[mask, :], axis=0))

        CorrelationMatrix_Subsampled = np.corrcoef(np.vstack(roi_resp))
        functional_adjacency_Subsampled = CorrelationMatrix_Subsampled[upper_inds]
        r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_Subsampled)
        r_iter.append(r_new)

    r_subsampled.append(r_iter)
r_subsampled = np.vstack(r_subsampled)

# plot mean+/-SEM results on top of region size cumulative histogram
err_y = np.std(r_subsampled, axis=1) / np.sqrt(r_subsampled.shape[1])
mean_y = np.mean(r_subsampled, axis=1)

fh3, ax1 = plt.subplots(1, 1, figsize=(6,6))
# ax1.plot(subsampled_sizes, mean_y, 'k-o')
ax1.errorbar(subsampled_sizes, mean_y, yerr=err_y, color='k')
ax1.hlines(r_full, subsampled_sizes.min(), subsampled_sizes.max(), color='k', linestyle='--')
ax1.set_xlabel('Region size (voxels)')
ax1.set_ylabel('Correlation with anatomical connectivity'.format(voxel_numbers[-1]))
ax1.set_xscale('log')
ax2 = ax1.twinx()
ax2.plot(base[:-1], cumulative)
ax2.set_ylabel('Cumulative fraction')
ax2.set_ylim([0, 1.05])

print('Finished F3 (time = {:.1f} sec)'.format(time.time()-t0))


# %% save all figs to collated PDF
with PdfPages(os.path.join(analysis_dir, 'SC_FC_subsample_{}.pdf'.format(fly_id))) as pdf:
    pdf.savefig(fh1)
    pdf.savefig(fh2)
    pdf.savefig(fh3)

    d = pdf.infodict()
    d['Title'] = 'SC-FC size ctl'
    d['Author'] = 'Max Turner'
    d['ModDate'] = datetime.datetime.today()

plt.close('all')

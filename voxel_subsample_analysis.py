import os
import nibabel as nib
import numpy as np
import ants
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from region_connectivity import RegionConnectivity

data_dir = '/home/mhturner/CurrentData/resting_state'
analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/hemibrain_analysis/roi_connectivity'


fly_id = 'fly2'

atlas_brain = np.asarray(ants.image_read(os.path.join(data_dir, fly_id, 'vfb_68_Original.nii.gz')).numpy(), 'uint8')
functional_brain = ants.image_read(os.path.join(data_dir, fly_id, 'func_volreg_trim.nii.gz'))

# %%
mapping = RegionConnectivity.getRoiMapping()
CorrelationMatrix_Full = RegionConnectivity.loadFunctionalData()
roi_mask, roi_size = RegionConnectivity.loadAtlasData()

_, pull_inds = RegionConnectivity.filterFunctionalData(CorrelationMatrix_Full, mapping)
roi_mask = [roi_mask[x] for x in pull_inds]
roi_size = [roi_size[x] for x in pull_inds]


print(np.sort(roi_size))
# %%
voxel_numbers = [5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

at_least = voxel_numbers[-1]
C_mats = []
for num_vox in voxel_numbers:
    roi_resp = []
    for r_ind, mask in enumerate(roi_mask):
        mask_size = np.sum(mask)
        if mask_size >= at_least:
            pull_inds = np.random.choice(np.arange(roi_size[r_ind]), size=num_vox, replace=False)
            roi_resp.append(np.mean(functional_brain[mask, :][pull_inds,:], axis=0))

    C = np.corrcoef(np.vstack(roi_resp))
    C_mats.append(C)

#

# %% correlation with final C as fxn of voxel population
upper_inds = np.triu_indices(C_mats[-1].shape[0], k=1) #k=1 excludes main diagonal
corr = []
for ind, num_vox in enumerate(voxel_numbers):
    r, p = pearsonr(C_mats[ind][upper_inds], C_mats[-1][upper_inds])
    corr.append(r)

# %%
fh1, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].plot(voxel_numbers[:-1], corr[:-1], 'k-o')
ax[0].plot(voxel_numbers, np.ones_like(voxel_numbers), 'k--')
ax[0].set_xlabel('Number of voxels included')
ax[0].set_ylabel('Correlation with {}-voxel functional connectivity'.format(voxel_numbers[-1]))
ax[0].set_ylim([0, 1.1])

bins = np.arange(np.floor(np.min(roi_size)), np.ceil(np.max(roi_size)))
values, base = np.histogram(roi_size, bins=bins, density=True)
cumulative = np.cumsum(values)

ax[1].plot(base[:-1], cumulative)
ax[1].plot([1000, 1000], [0, 1], 'k--')
ax[1].set_ylabel('Cumulative fraction')
ax[1].set_xlabel('Region size')

# %%
"""
For each region: subsample set fractions of all voxels and compute correlation to full region response

"""
full_responses = []
for r_ind, mask in enumerate(roi_mask):
    full_responses.append(np.mean(functional_brain[mask, :], axis=0))
full_responses = np.vstack(full_responses)
full_responses.shape
# %%

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
# %%
fh2, ax = plt.subplots(1, 1, figsize=(5,5))
ax.plot(test_fractions, region_corrs.T, LineStyle='-', Marker='.', alpha=0.2)
ax.plot(test_fractions, np.mean(region_corrs, axis=0), LineStyle='-', Marker='o', Color='k', LineWidth=2)
ax.set_ylim([0, 1.1])
ax.set_xlabel('Fraction of voxels used')
ax.set_ylabel('Corr. with full region response');

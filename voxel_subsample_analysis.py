"""
calculate SC-FC correlation for subsampled regions, bootstrap over some iterations
"""

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import time
import glob

from region_connectivity import RegionConnectivity

t_total_0 = time.time()

analysis_dir = '/oak/stanford/groups/trc/data/Max/flynet/analysis'
data_dir = '/oak/stanford/groups/trc/data/Max/flynet/data'

brain_filepaths = glob.glob(os.path.join(data_dir, '5d_atlas', 'func_volreg') + '*')

# get ROI names and mapping info
mapping = RegionConnectivity.getRoiMapping()
rois = list(mapping.keys())
rois.sort()

# Get anatomical connectivity
WeakConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'WeakConnections_computed_20200507.pkl'))
MediumConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'MediumConnections_computed_20200507.pkl'))
StrongConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'StrongConnections_computed_20200507.pkl'))
conn_mat = WeakConnections + MediumConnections + StrongConnections
# set diag to nan
tmp_mat = conn_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)
# symmetrize anatomical adjacency matrix by just adding it to its transpose and dividing by 2. Ignores directionality
ConnectivityMatrix_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=rois, columns=rois)
upper_inds = np.triu_indices(ConnectivityMatrix_Symmetrized.shape[0], k=1) # k=1 excludes main diagonal
anatomical_adjacency = ConnectivityMatrix_Symmetrized.to_numpy().copy()[upper_inds]
print('Loaded anatomical connectivity')

brain_filepaths = brain_filepaths[0:1] # TEST

subsampled_sizes = np.logspace(1, 4.4, 3) # voxels
n_iter = 2 # num iterations for randomly subsampling regions


# Get region sizes from atlas data
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')
roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')
roi_mask, roi_size = RegionConnectivity.loadAtlasData(atlas_path, roinames_path, mapping=mapping)
bins = np.arange(np.floor(np.min(roi_size)), np.ceil(np.max(roi_size)))
values, base = np.histogram(roi_size, bins=bins, density=True)
cumulative = np.cumsum(values)

# Get full region SC-FC correlation, avg across flies
cmats = []
for brain_fp in brain_filepaths:
    suffix = brain_fp.split('func_volreg_')[-1]
    atlas_fp = os.path.join(data_dir, 'vfb_68_' + suffix)
    roi_mask, _ = RegionConnectivity.loadAtlasData(atlas_fp, roinames_path, mapping=mapping)

    # Load functional brain
    functional_brain = np.asanyarray(nib.load(brain_fp).dataobj).astype('uint16')

    # get region responses and SC-FC corr
    region_responses_full = RegionConnectivity.computeRegionResponses(functional_brain, roi_mask)
    correlation_matrix = np.corrcoef(full_region_responses)
    # set diag to 0
    np.fill_diagonal(correlation_matrix, 0)
    # fischer z transform (arctanh) and append
    cmats.append(np.arctanh(correlation_matrix))

cmats = np.stack(cmats, axis=2) # population cmats, z transformed

CorrelationMatrix_Full = np.mean(cmats, axis=2)

functional_adjacency_full = CorrelationMatrix_Full[upper_inds]
sf_corr_full, _ = pearsonr(anatomical_adjacency, functional_adjacency_full)


# Loop over subsampled sizes
sf_corr_subsampled = [] #sizes x iterations, r values for SC-FC
for subsampled_size in subsampled_sizes:
    print('Starting size {}'.format(subsampled_size))
    t0 = time.time()
    r_iter = []
    for n in range(n_iter):
        print('{}/{} iterations...'.format(n, n_iter))
        # loop over all brains
        cmats = []
        for brain_fp in brain_filepaths:
            suffix = brain_fp.split('func_volreg_')[-1]
            atlas_fp = os.path.join(data_dir, 'vfb_68_' + suffix)
            roi_mask, _ = RegionConnectivity.loadAtlasData(atlas_fp, roinames_path, mapping=mapping)

            # Load functional brain
            functional_brain = np.asanyarray(nib.load(brain_fp).dataobj).astype('uint16')

            # get SUBSAMPLED region responses and SC-FC corr
            region_responses_subsampled = []
            for r_ind, mask in enumerate(roi_mask):
                mask_size = np.sum(mask)
                if mask_size >= subsampled_size:
                    pull_inds = np.random.choice(np.arange(roi_size[r_ind]), size=int(subsampled_size), replace=False)
                    region_responses_subsampled.append(np.mean(functional_brain[mask, :][pull_inds,:], axis=0))
                else: # not enough voxels in mask, so just take the whole region
                    region_responses_subsampled.append(np.mean(functional_brain[mask, :], axis=0))

            correlation_matrix = np.corrcoef(np.vstack(region_responses_subsampled))
            # set diag to 0
            np.fill_diagonal(correlation_matrix, 0)
            # fischer z transform (arctanh) and append
            cmats.append(np.arctanh(correlation_matrix))

        cmats = np.stack(cmats, axis=2) # population cmats, z transformed
        CorrelationMatrix_Subsampled = np.mean(cmats, axis=2)

        functional_adjacency_subsampled = CorrelationMatrix_Subsampled[upper_inds]
        r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_subsampled)
        r_iter.append(r_new)

    sf_corr_subsampled.append(r_iter)
    print('Finished size {} (time = {:.1f} sec)'.format(subsampled_size, time.time()-t0))

sf_corr_subsampled = np.vstack(sf_corr_subsampled)

save_fn = os.path.join(analysis_dir, 'subsampled_sf_corrs.npy')
np.save(save_fn, (sf_corr_subsampled, subsampled_sizes))


# # %%
# # plot mean+/-SEM results on top of region size cumulative histogram
# err_y = np.std(r_subsampled, axis=1) / np.sqrt(r_subsampled.shape[1])
# mean_y = np.mean(r_subsampled, axis=1)
#
# fh1, ax1 = plt.subplots(1, 1, figsize=(6,6))
# # ax1.plot(subsampled_sizes, mean_y, 'k-o')
# ax1.errorbar(subsampled_sizes, mean_y, yerr=err_y, color='k')
# ax1.hlines(r_full, subsampled_sizes.min(), subsampled_sizes.max(), color='k', linestyle='--')
# ax1.set_xlabel('Region size (voxels)')
# ax1.set_ylabel('Correlation with anatomical connectivity'.format(voxel_numbers[-1]))
# ax1.set_xscale('log')
# ax2 = ax1.twinx()
# ax2.plot(base[:-1], cumulative)
# ax2.set_ylabel('Cumulative fraction')
# ax2.set_ylim([0, 1.05])
#
#
#
#
# # %% save all figs to collated PDF
# with PdfPages(os.path.join(analysis_dir, 'SC_FC_subsample_{}.pdf'.format(fly_id))) as pdf:
#     pdf.savefig(fh1)
#
#     d = pdf.infodict()
#     d['Title'] = 'SC-FC size ctl'
#     d['Author'] = 'Max Turner'
#     d['ModDate'] = datetime.datetime.today()
#
# plt.close('all')

print('Finished all brains (time = {:.1f} sec)'.format(time.time()-t_total_0))

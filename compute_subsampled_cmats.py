"""Calculate functional cmats for subsampled regions."""
import os
import nibabel as nib
import numpy as np
import time
import glob
import pandas as pd

from scfc import functional_connectivity, bridge

# HP filtering responses
fs = 1.2 # Hz
cutoff = 0.01 # Hz

subsampled_sizes = np.logspace(1, 4.4, 16) # voxels
n_iter = 10 # num iterations for randomly subsampling regions

t_total_0 = time.time()

data_dir = bridge.getUserConfiguration()['data_dir']
analysis_dir = bridge.getUserConfiguration()['analysis_dir']

brain_filepaths = glob.glob(os.path.join(data_dir, 'brain_files', 'func_volreg') + '*')
include_inds_ito, name_list_ito = bridge.getItoNames()

# Get full cmat, avg across flies
t0 = time.time()
cmats_full = []
cmats_subsampled = []
for brain_fp in brain_filepaths:
    t0 = time.time()
    suffix = brain_fp.split('func_volreg_')[-1]
    file_id = suffix.replace('.nii.gz', '')
    atlas_fp = os.path.join(data_dir, 'ito_68_atlas', 'vfb_68_' + suffix)

    mask_brain = np.asarray(np.squeeze(nib.load(atlas_fp).get_fdata()), 'uint16')
    functional_brain = np.asanyarray(nib.load(brain_fp).dataobj).astype('uint16')

    rois = np.unique(mask_brain) # roi ID numbers
    roi_mask = []
    for r_ind, r in enumerate(rois):
        new_roi_mask = np.zeros_like(mask_brain)
        new_roi_mask = mask_brain == r # bool
        roi_mask.append(new_roi_mask)

    # (1) Full region responses:
    # region_responses: n_rois x n_timepoints np array, mean voxel response in each region
    region_responses_full = functional_connectivity.computeRegionResponses(functional_brain, roi_mask)
    region_responses_full = functional_connectivity.filterRegionResponse(region_responses_full, cutoff=cutoff, fs=fs)
    RegionResponses = pd.DataFrame(data=functional_connectivity.trimRegionResponse(file_id, region_responses_full), index=rois)

    # Trim down to included ito regions
    resp_included = RegionResponses.reindex(include_inds_ito).to_numpy()
    # compute cmat
    correlation_matrix = np.corrcoef(resp_included)
    # set diag to 0
    np.fill_diagonal(correlation_matrix, 0)
    # fischer z transform (arctanh) and append
    cmats_full.append(np.arctanh(correlation_matrix))

    # (2) Subsampled regions
    cmats_sizes = []
    for subsampled_size in subsampled_sizes:
        cmats_iter = []
        for n in range(n_iter):
            # get SUBSAMPLED region responses and SC-FC corr
            region_responses_subsampled = []
            for r_ind, mask in enumerate(roi_mask):
                mask_size = np.sum(mask)
                if mask_size >= subsampled_size:
                    pull_inds = np.random.choice(np.arange(mask_size), size=int(subsampled_size), replace=False)
                    region_responses_subsampled.append(np.mean(functional_brain[mask, :][pull_inds, :], axis=0))
                else: # not enough voxels in mask, so just take the whole region
                    region_responses_subsampled.append(np.mean(functional_brain[mask, :], axis=0))

            # get region responses and filter, trim
            region_responses_subsampled = functional_connectivity.filterRegionResponse(np.vstack(region_responses_subsampled), cutoff=cutoff, fs=fs)
            region_responses_subsampled = functional_connectivity.trimRegionResponse(file_id, region_responses_subsampled)
            RegionResponses = pd.DataFrame(data=functional_connectivity.trimRegionResponse(file_id, region_responses_subsampled), index=rois)

            # Trim down to included ito regions
            resp_included = RegionResponses.reindex(include_inds_ito).to_numpy()
            # compute cmat
            correlation_matrix = np.corrcoef(resp_included) # roi x roi
            # set diag to 0
            np.fill_diagonal(correlation_matrix, 0)
            # fischer z transform (arctanh) and append
            cmats_iter.append(np.arctanh(correlation_matrix))
        cmats_iter = np.stack(cmats_iter, axis=2) # roi x roi x iterations
        cmats_sizes.append(cmats_iter)

    cmats_subsampled.append(np.stack(cmats_sizes, axis=3)) # List of [roi x roi x iterations x size]
    print('Finished brain {} (time = {:.1f} sec)'.format(suffix, time.time()-t0))

# Full corrmat
mean_cmat = np.mean(np.stack(cmats_full, axis=2), axis=2) # roi x roi
np.fill_diagonal(mean_cmat, np.nan)
CorrelationMatrix_Full = pd.DataFrame(data=mean_cmat, index=name_list_ito, columns=name_list_ito)

RegionResponses.to_pickle(os.path.join(data_dir, 'subsample_CorrelationMatrix_Full.pkl'))
np.save(os.path.join(data_dir, 'subsample_cmats_full.npy'), (cmats_full))
print('Saved full cmats to {}'.format(data_dir))

# Subsampled corrmat
cmats_subsampled = np.stack(cmats_subsampled, axis=4) # roi x roi x iterations x sizes x brains
np.save(os.path.join(data_dir, 'subsample_cmats_sub.npy'), (cmats_subsampled, subsampled_sizes))
print('Saved subsampled cmats to {}'.format(data_dir))
print('Total time = {:.1f} sec)'.format(time.time()-t_total_0))

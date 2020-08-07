"""
calculate functional cmats for subsampled regions
"""
import os
import nibabel as nib
import numpy as np
import time
import glob
import datetime

from scfc import anatomical_connectivity, functional_connectivity

# HP filtering responses
fs = 1.2 #Hz
cutoff = 0.01 #Hz

t_total_0 = time.time()

analysis_dir = '/oak/stanford/groups/trc/data/Max/flynet/analysis'
data_dir = '/oak/stanford/groups/trc/data/Max/flynet/data'
roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')
mapping = functional_connectivity.getRoiMapping()

brain_filepaths = glob.glob(os.path.join(data_dir, '5d_atlas', 'func_volreg') + '*')

subsampled_sizes = np.logspace(1, 4.4, 16) # voxels
n_iter = 10 # num iterations for randomly subsampling regions

# Get full cmat, avg across flies
t0 = time.time()
cmats = []
for brain_fp in brain_filepaths:
    suffix = brain_fp.split('func_volreg_')[-1]
    file_id = suffix.replace('.nii.gz', '')
    atlas_fp = os.path.join(data_dir, '5d_atlas', 'vfb_68_' + suffix)
    roi_mask, _ = functional_connectivity.loadAtlasData(atlas_fp, roinames_path, mapping=mapping)

    # Load functional brain
    functional_brain = np.asanyarray(nib.load(brain_fp).dataobj).astype('uint16')

    # get region responses and filter, trim
    region_responses_full = functional_connectivity.computeRegionResponses(functional_brain, roi_mask)
    region_responses_full = functional_connectivity.filterRegionResponse(region_responses_full, cutoff=cutoff, fs=fs)
    region_responses_full = functional_connectivity.trimRegionResponse(file_id, region_responses_full)

    # compute cmat
    correlation_matrix = np.corrcoef(region_responses_full)
    # set diag to 0
    np.fill_diagonal(correlation_matrix, 0)
    # fischer z transform (arctanh) and append
    cmats.append(np.arctanh(correlation_matrix))

cmats = np.stack(cmats, axis=2) # population cmats, z transformed

CorrelationMatrix_Full = np.mean(cmats, axis=2) # roi x roi
print('Finished full region cmat (time = {:.1f} sec)'.format(time.time()-t0))

# population cmats for subsampled sizes and iterations
cmats_pop = []
for brain_fp in brain_filepaths:
    suffix = brain_fp.split('func_volreg_')[-1]
    print('Starting brain {}'.format(suffix))
    t0 = time.time()
    atlas_fp = os.path.join(data_dir, '5d_atlas', 'vfb_68_' + suffix)
    roi_mask, _ = functional_connectivity.loadAtlasData(atlas_fp, roinames_path, mapping=mapping)

    # Load functional brain
    functional_brain = np.asanyarray(nib.load(brain_fp).dataobj).astype('uint16')
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
                    region_responses_subsampled.append(np.mean(functional_brain[mask, :][pull_inds,:], axis=0))
                else: # not enough voxels in mask, so just take the whole region
                    region_responses_subsampled.append(np.mean(functional_brain[mask, :], axis=0))

            # get region responses and filter, trim
            region_responses_subsampled = functional_connectivity.filterRegionResponse(np.vstack(region_responses_subsampled), cutoff=cutoff, fs=fs)
            region_responses_subsampled = functional_connectivity.trimRegionResponse(file_id, region_responses_subsampled)

            correlation_matrix = np.corrcoef(region_responses_subsampled) # roi x roi
            # set diag to 0
            np.fill_diagonal(correlation_matrix, 0)
            # fischer z transform (arctanh) and append
            cmats_iter.append(np.arctanh(correlation_matrix))
        cmats_iter = np.stack(cmats_iter, axis=2) # roi x roi x iterations
        cmats_sizes.append(cmats_iter)

    cmats_sizes = np.stack(cmats_sizes, axis=3) # roi x roi x iterations x size    print(cmats_sizes.shape)
    cmats_pop.append(cmats_sizes)
    print('Finished brain {} (time = {:.1f} sec)'.format(suffix, time.time()-t0))

d = datetime.datetime.today()
datestring ='{:02d}'.format(d.year)+'{:02d}'.format(d.month)+'{:02d}'.format(d.day)

cmats_pop = np.stack(cmats_pop, axis=4) # roi x roi x iterations x sizes x brains
save_fn = os.path.join(analysis_dir, 'subsampled_cmats_{}.npy'.format(datestring))
np.save(save_fn, (cmats_pop, CorrelationMatrix_Full, subsampled_sizes))
print('Finished all brains, saved full and population cmats at {}'.format(save_fn))
print('Total time = {:.1f} sec)'.format(time.time()-t_total_0))

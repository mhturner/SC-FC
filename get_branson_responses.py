"""Compute branson supervoxel/region responses from registered brain volume data."""
import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
import time

from scfc import functional_connectivity, bridge
t_total_0 = time.time()

# data_dir = bridge.getUserConfiguration()['data_dir']
data_dir = '/oak/stanford/groups/trc/data/Max/flynet/data'

brain_filepaths = glob.glob(os.path.join(data_dir, '5d_atlas', 'func_volreg') + '*')
roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')

rois = np.arange(1, 1000) # branson rois on [1, 999]


for brain_fp in brain_filepaths:
    t0 = time.time()
    suffix = brain_fp.split('func_volreg_')[-1]
    atlas_fp = os.path.join(data_dir, 'branson_999_atlas', 'vfb_999_' + suffix)

    mask_brain = np.asarray(np.squeeze(nib.load(atlas_fp).get_fdata()), 'uint16')
    functional_brain = np.asanyarray(nib.load(brain_fp).dataobj).astype('uint16')

    roi_mask = []
    for r_ind, r in enumerate(rois):
        new_roi_mask = np.zeros_like(mask_brain)
        new_roi_mask = mask_brain == r # bool
        roi_mask.append(new_roi_mask)

    # region_responses: n_rois x n_timepoints np array, mean voxel response in each region
    region_responses = functional_connectivity.computeRegionResponses(functional_brain, roi_mask)

    # make a pandas series to associate ROI names
    RegionResponses = pd.DataFrame(data=region_responses, index=rois)

    save_fn = 'branson_' + suffix.split('.')[0] + '.pkl'
    save_path = os.path.join(data_dir, 'branson_responses', save_fn)

    RegionResponses.to_pickle(save_path)

    print('Finished brain {} (time = {:.1f} sec)'.format(save_fn, time.time()-t0))

print('Finished all brains (time = {:.1f} sec)'.format(time.time()-t_total_0))

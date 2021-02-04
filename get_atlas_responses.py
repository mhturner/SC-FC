"""
Turner, Mann, Clandinin: Compute Atlas region responses from registered brain volume data.

*Requires raw brain files (large) - contact Max Turner for these data.

https://github.com/mhturner/SC-FC
mhturner@stanford.edu
"""

import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
import time
import sys

from scfc import functional_connectivity, bridge

atlas_id = sys.argv[1] # name of atlas to use: ito or branson

t_total_0 = time.time()

data_dir = bridge.getUserConfiguration()['data_dir']

brain_filepaths = glob.glob(os.path.join(data_dir, 'brain_files', 'func_volreg') + '*')

for brain_fp in brain_filepaths:
    t0 = time.time()
    suffix = brain_fp.split('func_volreg_')[-1]
    if atlas_id == 'ito':
        atlas_fp = os.path.join(data_dir, 'ito_68_atlas', 'vfb_68_' + suffix)
    elif atlas_id == 'branson':
        atlas_fp = os.path.join(data_dir, 'branson_999_atlas', 'vfb_999_' + suffix)
    else:
        print('Unrecognized atlas ID')

    mask_brain = np.asarray(np.squeeze(nib.load(atlas_fp).get_fdata()), 'uint16')
    functional_brain = np.asanyarray(nib.load(brain_fp).dataobj).astype('uint16')

    rois = np.unique(mask_brain) # roi ID numbers

    roi_mask = []
    for r_ind, r in enumerate(rois):
        new_roi_mask = np.zeros_like(mask_brain)
        new_roi_mask = mask_brain == r # bool
        roi_mask.append(new_roi_mask)

    # region_responses: n_rois x n_timepoints np array, mean voxel response in each region
    region_responses = functional_connectivity.computeRegionResponses(functional_brain, roi_mask)

    # make a pandas series to associate ROI names
    RegionResponses = pd.DataFrame(data=region_responses, index=rois)

    save_fn = '{}_'.format(atlas_id) + suffix.split('.')[0] + '.pkl'
    save_path = os.path.join(data_dir, '{}_responses'.format(atlas_id), save_fn)

    RegionResponses.to_pickle(save_path)

    print('Finished brain {} (time = {:.1f} sec)'.format(save_fn, time.time()-t0))

print('Finished all brains (time = {:.1f} sec)'.format(time.time()-t_total_0))

import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
import time

from region_connectivity import RegionConnectivity
t_total_0 = time.time()

data_dir = '/oak/stanford/groups/trc/data/Max/flynet/data'

brain_filepaths = glob.glob(os.path.join(data_dir, '5d_atlas', 'func_volreg') + '*')
roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')

mapping = RegionConnectivity.getRoiMapping()
rois = list(mapping.keys())
rois.sort()

for brain_fp in brain_filepaths:
    t0 = time.time()
    suffix = brain_fp.split('func_volreg_')[-1]
    atlas_fp = os.path.join(data_dir, '5d_atlas', 'vfb_68_' + suffix)

    functional_brain = np.asanyarray(nib.load(brain_fp).dataobj).astype('uint16')

    roi_mask, _ = RegionConnectivity.loadAtlasData(atlas_fp, roinames_path, mapping=mapping)

    # region_responses: n_rois x n_timepoints np array, mean voxel response in each region
    region_responses = RegionConnectivity.computeRegionResponses(functional_brain, roi_mask)

    # make a pandas series to associate ROI names
    RegionResponses = pd.DataFrame(data=region_responses, index=rois)

    save_fn = suffix.split('.')[0] + '.pkl'
    save_path = os.path.join(data_dir, 'region_responses', save_fn)

    RegionResponses.to_pickle(save_path)

    print('Finished brain {} (time = {:.1f} sec)'.format(save_fn, time.time()-t0))

print('Finished all brains (time = {:.1f} sec)'.format(time.time()-t_total_0))

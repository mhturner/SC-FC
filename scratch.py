import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from region_connectivity import RegionConnectivity

data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')
roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')
# %%
mapping = RegionConnectivity.getRoiMapping()
rois = list(mapping.keys())
rois.sort()

roi_mask, roi_size = RegionConnectivity.loadAtlasData(atlas_path=atlas_path, roinames_path=roinames_path, mapping=mapping)

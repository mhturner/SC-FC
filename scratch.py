import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from region_connectivity import RegionConnectivity

data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'

motion_filepaths = glob.glob(os.path.join(data_dir, 'behavior_data', 'motion_events') + '*')

fh, ax = plt.subplots(6, 1, figsize=(16, 8))
for ind, motion_fp in enumerate(motion_filepaths):
    suffix = motion_fp.split('motion_events_')[-1].split('.')[0]
    is_behaving = RegionConnectivity.getBehavingBinary(motion_fp)
    ax[ind].plot(is_behaving)

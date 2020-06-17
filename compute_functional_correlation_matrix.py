import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from region_connectivity import RegionConnectivity
import datetime

analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC'
data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'

response_filepaths = glob.glob(os.path.join(data_dir, 'region_responses') + '/' + '*.pkl')

CorrelationMatrix_Functional, cmats = RegionConnectivity.getFunctionalConnectivity(response_filepaths)


# %% compare recomputed with precomputed matrices
import seaborn as sns

mapping = RegionConnectivity.getRoiMapping()

roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')
cmat_path = os.path.join(data_dir, 'functional_connectivity', 'full_cmat.txt')
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')

cmat_precomp = RegionConnectivity.loadFunctionalData(cmat_path=cmat_path, roinames_path=roinames_path, mapping=mapping)

fh, ax = plt.subplots(1, 3, figsize=(18,6))
sns.heatmap(cmat_precomp, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Correlation (z)', 'shrink': .8}, cmap="cividis", rasterized=True)
ax[0].set_aspect('equal')
ax[0].set_title('Precomputed');

sns.heatmap(CorrelationMatrix_Functional, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True)
ax[1].set_aspect('equal')
ax[1].set_title('Re-computed from raw data');


upper_inds = np.triu_indices(CorrelationMatrix_Functional.shape[0], k=1) # k=1 excludes main diagonal

ax[2].plot(CorrelationMatrix_Functional.to_numpy()[upper_inds], cmat_precomp.to_numpy()[upper_inds], 'ko')
ax[2].plot([0, 1.5], [0, 1.5], 'k--')
ax[2].set_xlabel('Recomputed')
ax[2].set_ylabel('Precomputed')

fh.savefig('Precomputed_vs_recomputed.png')

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

cmats_z = []
for resp_fp in response_filepaths:
    region_responses = pd.read_pickle(resp_fp)
    correlation_matrix = np.corrcoef(np.vstack(region_responses.to_numpy()))
    # set diag to 0
    np.fill_diagonal(correlation_matrix, 0)
    # fischer z transform (arctanh) and append
    cmats_z.append(np.arctanh(correlation_matrix))

cmats_z = np.stack(cmats_z, axis=2)

# Make pd Dataframe
mean_cmat = np.mean(cmats_z, axis=2)
np.fill_diagonal(mean_cmat, np.nan)
CorrelationMatrix_Functional = pd.DataFrame(data=mean_cmat, index=region_responses.index, columns=region_responses.index)

# Save
d = datetime.datetime.today()
datestring ='{:02d}'.format(d.year)+'{:02d}'.format(d.month)+'{:02d}'.format(d.day)
save_path = os.path.join(data_dir, 'functional_connectivity', 'CorrelationMatrix_Functional_{}.pkl'.format(datestring))
CorrelationMatrix_Functional.to_pickle(save_path)


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

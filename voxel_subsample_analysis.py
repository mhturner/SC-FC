import os
import nibabel as nib
import numpy as np
import ants
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from region_connectivity import RegionConnectivity
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import time
import socket
 # TODO: add dates, more fly data, and loop thru flies for pop'n subsampling

if socket.gethostname() == 'max-laptop':
    analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/hemibrain_analysis/roi_connectivity'
    data_dir = '/home/mhturner/CurrentData/resting_state'
elif 'sh' in socket.gethostname():
    analysis_dir = '/oak/stanford/groups/trc/data/Max/Analysis/resting_state'
    data_dir = '/oak/stanford/groups/trc/data/Max/Analysis/resting_state/data'

date_id = ''
fly_id = ['fly1']


# %% filter to select rois of interest from mapping
mapping = RegionConnectivity.getRoiMapping()
rois = list(mapping.keys())
rois.sort()
roi_mask, roi_size = RegionConnectivity.loadAtlasData(data_dir=data_dir, mapping=mapping)
s_inds = np.argsort(roi_size)
np.sort(roi_size)

bins = np.arange(np.floor(np.min(roi_size)), np.ceil(np.max(roi_size)))
values, base = np.histogram(roi_size, bins=bins, density=True)
cumulative = np.cumsum(values)

s_inds
np.array(rois)[s_inds]
# %% calculate corr with anatomy for subsampled regions
"""
Fig 1: calculate SC-FC correlation for subsampled regions, bootstrap over some iterations
"""

# # # Get anatomical connectivity data # # #
t0 = time.time()
WeakConnections = pd.read_pickle(os.path.join(data_dir, 'WeakConnections_computed_20200507.pkl'))
MediumConnections = pd.read_pickle(os.path.join(data_dir, 'MediumConnections_computed_20200507.pkl'))
StrongConnections = pd.read_pickle(os.path.join(data_dir, 'StrongConnections_computed_20200507.pkl'))
conn_mat = WeakConnections + MediumConnections + StrongConnections
roi_names = conn_mat.index
# set diag to nan
tmp_mat = conn_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, np.nan)
# symmetrize anatomical adjacency matrix by just adding it to its transpose and dividing by 2. Ignores directionality
ConnectivityMatrix_Symmetrized = pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=roi_names, columns=roi_names)
upper_inds = np.triu_indices(ConnectivityMatrix_Symmetrized.shape[0], k=1) # k=1 excludes main diagonal
anatomical_adjacency = ConnectivityMatrix_Symmetrized.to_numpy().copy()[upper_inds]
print('Loaded anatomical connectivity (time = {:.1f} sec)'.format(time.time()-t0))

# # # Get subsampled correlation matrices for each fly # # #
for fly in flies:
    atlas_brain = np.asarray(ants.image_read(os.path.join(data_dir, fly_id[fly], 'vfb_68_Original.nii.gz')).numpy(), 'uint8')
    functional_brain = np.asanyarray(nib.load(os.path.join(data_dir, fly_id[fly], 'func_volreg.nii.gz')).dataobj).astype('uint16')


    # get full region corr matrix
    full_responses = []
    for r_ind, mask in enumerate(roi_mask):
        full_responses.append(np.mean(functional_brain[mask, :], axis=0))
    full_responses = np.vstack(full_responses)
    CorrelationMatrix_Full = np.corrcoef(np.vstack(full_responses))
    functional_adjacency_full = CorrelationMatrix_Full[upper_inds]
    r_full, _ = pearsonr(anatomical_adjacency, functional_adjacency_full)

    # now for subsampled regions: compute SC-FC corr
    subsampled_sizes = np.logspace(1, 4.4, 20) # voxels
    n_iter = 10 # num iterations for randomly subsampling regions

    r_subsampled = []
    for subsampled_size in subsampled_sizes:
        r_iter = []
        for n in range(n_iter):
            roi_resp = []
            for r_ind, mask in enumerate(roi_mask):
                mask_size = np.sum(mask)
                if mask_size >= subsampled_size:
                    pull_inds = np.random.choice(np.arange(roi_size[r_ind]), size=int(subsampled_size), replace=False)
                    roi_resp.append(np.mean(functional_brain[mask, :][pull_inds,:], axis=0))
                else: # not enough voxels in mask, so just take the whole region
                    roi_resp.append(np.mean(functional_brain[mask, :], axis=0))

            CorrelationMatrix_Subsampled = np.corrcoef(np.vstack(roi_resp))
            functional_adjacency_Subsampled = CorrelationMatrix_Subsampled[upper_inds]
            r_new, _ = pearsonr(anatomical_adjacency, functional_adjacency_Subsampled)
            r_iter.append(r_new)

        r_subsampled.append(r_iter)
    r_subsampled = np.vstack(r_subsampled)

# %%
# plot mean+/-SEM results on top of region size cumulative histogram
err_y = np.std(r_subsampled, axis=1) / np.sqrt(r_subsampled.shape[1])
mean_y = np.mean(r_subsampled, axis=1)

fh1, ax1 = plt.subplots(1, 1, figsize=(6,6))
# ax1.plot(subsampled_sizes, mean_y, 'k-o')
ax1.errorbar(subsampled_sizes, mean_y, yerr=err_y, color='k')
ax1.hlines(r_full, subsampled_sizes.min(), subsampled_sizes.max(), color='k', linestyle='--')
ax1.set_xlabel('Region size (voxels)')
ax1.set_ylabel('Correlation with anatomical connectivity'.format(voxel_numbers[-1]))
ax1.set_xscale('log')
ax2 = ax1.twinx()
ax2.plot(base[:-1], cumulative)
ax2.set_ylabel('Cumulative fraction')
ax2.set_ylim([0, 1.05])




# %% save all figs to collated PDF
with PdfPages(os.path.join(analysis_dir, 'SC_FC_subsample_{}.pdf'.format(fly_id))) as pdf:
    pdf.savefig(fh1)

    d = pdf.infodict()
    d['Title'] = 'SC-FC size ctl'
    d['Author'] = 'Max Turner'
    d['ModDate'] = datetime.datetime.today()

plt.close('all')

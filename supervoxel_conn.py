"""."""

import matplotlib.pyplot as plt
from neuprint import Client, fetch_neurons, fetch_custom, NeuronCriteria
import numpy as np
import os
import pandas as pd
import seaborn as sns
import glob
import nibabel as nib
import nrrd
from skimage import io
from scfc import bridge, functional_connectivity
import time
from scipy.stats import pearsonr, spearmanr

data_dir = bridge.getUserConfiguration()['data_dir']
analysis_dir = bridge.getUserConfiguration()['analysis_dir']
token = bridge.getUserConfiguration()['token']

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.2', token=token)
# Get FunctionalConnectivity object
FC = functional_connectivity.FunctionalConnectivity(data_dir=data_dir, fs=1.2, cutoff=0.01, mapping=bridge.getRoiMapping())


# %% load branson atlas responses, compute FC matrix
response_filepaths = glob.glob(os.path.join(data_dir, 'branson_responses') + '/' + '*.pkl')

# (1) Select branson regions to include. Do some matching to Ito atlas naming. Sort alphabetically.
decoder_ring = pd.read_csv(os.path.join(data_dir, 'branson_999_atlas') + '/atlas_roi_values', header=None)
include_regions = ['AL_R', 'OTU_R', 'ATL_R', 'ATL_L', 'AVLP_R', 'LB_R', 'LB_L', 'CAN_R', 'CRE_R', 'CRE_L', 'EB', 'EPA_R', 'FB', 'GOR_R', 'GOR_L'
                   'IB_R', 'IB_L', 'ICL_R', 'LAL_R', 'LH_R', 'MB_R', 'MB_L', 'NO', 'PB', 'PLP_R', 'PVLP_R', 'SCL_R', 'SIP_R', 'SLP_R', 'SMP_R',
                   'SMP_L', 'SPS_R', 'VES_R', 'WED_R'] # LB = bulb

# ???? CAN # TODO: figure out where CAN went in Branson

include_inds = []
name_list = []
for ind in decoder_ring.index:
    row = decoder_ring.loc[ind].values[0]
    region = row.split(':')[0]
    start = row.split(' ')[1]
    end = row.split(' ')[3]
    if region in include_regions:
        include_inds.append(np.arange(int(start), int(end)+1))
        if 'LB' in region:
            region = region.replace('LB', 'BU') # to match name convention of ito atlas
        if 'OTU' in region:
            region = region.replace('OTU', 'AOTU')
        name_list.append(np.repeat(region, int(end)-int(start)+1))
include_inds = np.hstack(include_inds)
name_list = np.hstack(name_list)
sort_inds = np.argsort(name_list)
name_list = name_list[sort_inds]
include_inds = include_inds[sort_inds]

# (2) Compute cmat for each individual fly and compute across-average mean cmat
cmats_z = []
for resp_fp in response_filepaths:
    tmp = functional_connectivity.getProcessedRegionResponse(resp_fp, cutoff=0.01, fs=1.2)
    resp_included = tmp.loc[include_inds]
    # resp_included = tmp

    correlation_matrix = np.corrcoef(resp_included)
    # set diag to 0
    np.fill_diagonal(correlation_matrix, 0)
    # fischer z transform (arctanh) and append
    new_cmat_z = np.arctanh(correlation_matrix)
    cmats_z.append(new_cmat_z)

# Make mean pd Dataframe
mean_cmat = np.nanmean(np.stack(cmats_z, axis=2), axis=2)
np.fill_diagonal(mean_cmat, np.nan)
CorrelationMatrix = pd.DataFrame(data=mean_cmat, index=name_list, columns=name_list)
# %%
tt = response_filepaths[0]

tmp = pd.read_pickle(tt)
tmp.shape
np.unique(np.where(np.isnan(tmp.to_numpy()))[0])
tmp.shape

# tmp = functional_connectivity.getProcessedRegionResponse(tt, cutoff=0.01, fs=1.2)

np.any(np.isnan(tmp))
fh, ax = plt.subplots(1, 1, figsize=(8,4))
ax.plot(tmp);
# %%
np.any(np.isnan(cm))

np.where(np.isnan(cm))
plt.imshow(cm)

len(r_val)
# %% Plot across-animal average cmat heatmap. Compute corr between mean and individual fly cmats
fh1, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.heatmap(CorrelationMatrix, ax=ax, cmap='cividis')
fh1.savefig(os.path.join(analysis_dir, 'figpanels', 'branson_mean_FCmat.png'), format='png', transparent=True, dpi=400)
ax.set_aspect('equal')

meanvals = CorrelationMatrix.to_numpy()[np.triu_indices(len(name_list), k=1)]
t_inds = np.where(~np.isnan(meanvals))[0]
r_val = []
for cm in cmats_z:
    r, p = pearsonr(meanvals[t_inds], cm[np.triu_indices(len(name_list), k=1)][t_inds])
    r_val.append(r)

print('r = {:.2f} +/- {:.2f}'.format(np.mean(r_val), np.std(r_val)))

# %% For comparison: compute corr between mean and individual fly cmats for Ito atlas data

meanvals = FC.CorrelationMatrix.to_numpy()[np.triu_indices(36, k=1)]
t_inds = np.where(~np.isnan(meanvals))[0]
r_vals = []
for c_ind in range(FC.cmats.shape[2]):
    cmat = FC.cmats[:, :, c_ind]
    r, p = pearsonr(meanvals[t_inds], cmat[np.triu_indices(36, k=1)][t_inds])
    r_vals.append(r)

print('r = {:.2f} +/- {:.2f}'.format(np.mean(r_vals), np.std(r_vals)))

# %% Load branson_cellcount_matrix and branson_synmask from R / natverse script
count_matrix_jfrc2 = pd.read_csv(os.path.join(data_dir, 'JFRC2_branson_cellcount_matrix.csv'), header=0).to_numpy()[:, 1:]
count_matrix_jfrc2 = pd.DataFrame(data=count_matrix_jfrc2, index=np.arange(1, 1000), columns=np.arange(1, 1000))

count_matrix_jrc2018 = pd.read_csv(os.path.join(data_dir, 'JRC2018F_branson_cellcount_matrix.csv'), header=0).to_numpy()[:, 1:]
count_matrix_jrc2018 = pd.DataFrame(data=count_matrix_jrc2018, index=np.arange(1, 1000), columns=np.arange(1, 1000))

# filter and sort count_matrix by include_inds
Connectivity_JFRC2 = pd.DataFrame(data=np.zeros_like(mean_cmat), index=name_list, columns=name_list)
Connectivity_JRC2018 = pd.DataFrame(data=np.zeros_like(mean_cmat), index=name_list, columns=name_list)

for s_ind, src in enumerate(include_inds):
    for t_ind, trg in enumerate(include_inds):
        Connectivity_JFRC2.iloc[s_ind, t_ind] = count_matrix_jfrc2.loc[src, trg]
        Connectivity_JRC2018.iloc[s_ind, t_ind] = count_matrix_jrc2018.loc[src, trg]
# %%




# %%
fh0, ax0 = plt.subplots(1, 3, figsize=(18, 6))
sns.heatmap(CorrelationMatrix, ax=ax0[0], cmap='cividis')
ax0[0].set_title('Fxnal')

sns.heatmap(Connectivity_JFRC2, ax=ax0[1], cmap='cividis')
ax0[1].set_title('JFRC2 atlas')

sns.heatmap(Connectivity_JRC2018, ax=ax0[2], cmap='cividis')
ax0[2].set_title('JRC2018 atlas')

# %% load synmask tifs
branson_jfrc2 = io.imread(os.path.join(data_dir, 'AnatomySubCompartments20150108_ms999centers.tif'))
synmask_jfrc2 = io.imread(os.path.join(data_dir, 'JFRC2_branson_synmask.tif'))

branson_jrc2018 = io.imread(os.path.join(data_dir, '2018_999_atlas.tif'))
synmask_jrc2018 = io.imread(os.path.join(data_dir, 'JRC2018F_branson_synmask.tif'))

# %%

x = Connectivity_JRC2018.to_numpy()[np.triu_indices(len(name_list), k=1)]
keep_inds = np.where(x > 0)
x = np.log10(x[keep_inds])

y = CorrelationMatrix.to_numpy()[np.triu_indices(len(name_list), k=1)][keep_inds]

r, p = pearsonr(x, y)
coef = np.polyfit(x, y, 1)
linfit = np.poly1d(coef)


fh0, ax0 = plt.subplots(1, 1, figsize=(4, 4))
ax0.plot(10**x, y, color=[0.5, 0.5, 0.5], marker='.', linestyle='none', alpha=1.0)
xx = np.linspace(x.min(), x.max(), 100)
ax0.plot(10**xx, linfit(xx), color='k', linewidth=2, marker=None)
ax0.set_xscale('log')
ax0.set_xlabel('Cell Count')
ax0.set_ylabel('Functional corr. (z)')
ax0.annotate('r = {:.2f}'.format(r), xy=(1, 1.05))
ax0.tick_params(axis='x', labelsize=10)
ax0.tick_params(axis='y', labelsize=10)



# %%

fh1, ax1 = plt.subplots(2, 3, figsize=(18, 6))
ax1[0, 0].imshow(branson_jfrc2.sum(axis=2))
ax1[1, 0].imshow(synmask_jfrc2.sum(axis=2))
ax1[0, 1].imshow(branson_jfrc2.sum(axis=1))
ax1[1, 1].imshow(synmask_jfrc2.sum(axis=1))
ax1[0, 2].imshow(branson_jfrc2.sum(axis=0))
ax1[1, 2].imshow(synmask_jfrc2.sum(axis=0))

fh2, ax2 = plt.subplots(2, 3, figsize=(18, 6))
ax2[0, 0].imshow(branson_jrc2018.sum(axis=2))
ax2[1, 0].imshow(synmask_jrc2018.sum(axis=2))
ax2[0, 1].imshow(branson_jrc2018.sum(axis=1))
ax2[1, 1].imshow(synmask_jrc2018.sum(axis=1))
ax2[0, 2].imshow(branson_jrc2018.sum(axis=0))
ax2[1, 2].imshow(synmask_jrc2018.sum(axis=0))

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
from scfc import bridge, functional_connectivity
import time
from scipy.stats import pearsonr

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
mean_cmat = np.mean(np.stack(cmats_z, axis=2), axis=2)
np.fill_diagonal(mean_cmat, np.nan)
CorrelationMatrix = pd.DataFrame(data=mean_cmat, index=name_list, columns=name_list)

# %% Plot across-animal average cmat heatmap. Compute corr between mean and individual fly cmats
fh1, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.heatmap(CorrelationMatrix, ax=ax, cmap='cividis')
fh1.savefig(os.path.join(analysis_dir, 'figpanels', 'branson_mean_FCmat.png'), format='png', transparent=True, dpi=400)
ax.set_aspect('equal')

meanvals = CorrelationMatrix.to_numpy()[np.triu_indices(290, k=1)]
t_inds = np.where(~np.isnan(meanvals))[0]
r_val = []
for cm in cmats_z:
    r, p = pearsonr(meanvals[t_inds], cm[np.triu_indices(290, k=1)][t_inds])
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
branson_atlas_path = '/home/mhturner/Downloads/transforms/AnatomySubCompartments20150108_ms999centers.nrrd'
branson_brain, branson_header = nrrd.read(branson_atlas_path)
regions = np.unique(branson_brain)[1:] # cut out first region (0), which is empty


count_matrix = pd.read_csv('/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data/branson_cellcount_matrix.csv', header=0).to_numpy()[:, 1:]
count_matrix = pd.DataFrame(data=count_matrix, index=regions, columns=regions)


sns.heatmap(count_matrix)

# %% Load body_ids that make connections in the Ito, 36 region atlas data
body_ids = np.load(os.path.join(data_dir, 'connectome_connectivity', 'body_ids_20210114.npy'), allow_pickle=True)

np.savetxt(os.path.join(data_dir, 'body_ids.csv'), body_ids, delimiter=",")

body_ids = np.random.choice(body_ids, 100, replace=False)


# %% Go through body_ids cell by cell, figure out which branson atlas voxel contains each pre + post-synapse
t0 = time.time()

res = 0.68 # um/voxel of atlas
# res = 2 # um/voxel of atlas


regions = np.unique(branson_brain)[1:] # cut out first region (0), which is empty

count_matrix = pd.DataFrame(data=np.zeros((len(regions), len(regions))), index=regions, columns=regions)

syn_mask = np.zeros_like(branson_brain)
for body in body_ids:
    Q = """\
        MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)
        WHERE n.status = "Traced"
        AND n.bodyId = {}
        RETURN DISTINCT [s.location.x * 8/1000, s.location.y * 8/1000, s.location.z * 8/1000] AS LOC, s.type

        """.format(body)
    df = fetch_custom(Q, format='pandas', client=neuprint_client)
    if len(df) > 0:
        inds = np.round(np.vstack(df['LOC']) / res).astype('int') # convert from microns to atlas voxels. Indices in mask where synapses are located
        df['region'] = branson_brain[inds[:, 1], inds[:, 0], inds[:, 2]] # region number for each row (synapse) associated with this cell

        target_regions = np.unique(df[df['s.type'] == 'pre'].region)
        source_regions = np.unique(df[df['s.type'] == 'post'].region)

        count_matrix.loc[source_regions[source_regions>0], target_regions[target_regions>0]] += 1
        syn_mask[inds[:, 1], inds[:, 0], inds[:, 2]] += 1

print('Done ({:.2f} sec)'.format(time.time() - t0))


# %% TODO: filter count_matrix by include_inds

branson_brain.shape

branson_brain.shape
# %%

fh0, ax0 = plt.subplots(1, 1, figsize=(4, 4))
ax0.imshow(count_matrix)
count_matrix

fh1, ax1 = plt.subplots(2, 3, figsize=(18, 6))
ax1[0, 0].imshow(branson_brain.sum(axis=2))
ax1[1, 0].imshow(syn_mask.sum(axis=2))

ax1[0, 1].imshow(branson_brain.sum(axis=1))
ax1[1, 1].imshow(syn_mask.sum(axis=1))

ax1[0, 2].imshow(branson_brain.sum(axis=0))
ax1[1, 2].imshow(syn_mask.sum(axis=0))

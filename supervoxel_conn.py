"""."""

import matplotlib.pyplot as plt
import matplotlib
from neuprint import Client
import numpy as np
import os
import pandas as pd
import seaborn as sns
import glob
from skimage import io
from scfc import bridge, functional_connectivity, plotting
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
response_filepaths.pop(6) # NOTE: this brain had one Branson atlas region missing, probably from the warping


# (1) Select branson regions to include. Do some matching to Ito atlas naming. Sort alphabetically.
decoder_ring = pd.read_csv(os.path.join(data_dir, 'branson_999_atlas') + '/atlas_roi_values', header=None)
include_regions = ['AL_R', 'OTU_R', 'ATL_R', 'ATL_L', 'AVLP_R', 'LB_R', 'LB_L', 'CRE_R', 'CRE_L', 'EB', 'EPA_R', 'FB', 'GOR_R', 'GOR_L'
                   'IB_R', 'IB_L', 'ICL_R', 'IVLP_R', 'LAL_R', 'LH_R', 'MB_R', 'MB_L', 'NO', 'PB', 'PLP_R', 'PVLP_R', 'SCL_R', 'SIP_R', 'SLP_R', 'SMP_R',
                   'SMP_L', 'SPS_R', 'VES_R', 'WED_R'] # LB = bulb

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
    resp_included = tmp.loc[include_inds].to_numpy()

    correlation_matrix = np.corrcoef(resp_included)
    correlation_matrix[np.where(np.isnan(correlation_matrix))] = 0
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
ax.set_aspect('equal')

meanvals = CorrelationMatrix.to_numpy()[np.triu_indices(len(name_list), k=1)]
r_val = []
for cm in cmats_z:
    r, p = pearsonr(meanvals, cm[np.triu_indices(len(name_list), k=1)])
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

# load synmask tifs
branson_jfrc2 = io.imread(os.path.join(data_dir, 'AnatomySubCompartments20150108_ms999centers.tif'))
synmask_jfrc2 = io.imread(os.path.join(data_dir, 'JFRC2_branson_synmask.tif'))

branson_jrc2018 = io.imread(os.path.join(data_dir, '2018_999_atlas.tif'))
synmask_jrc2018 = io.imread(os.path.join(data_dir, 'JRC2018F_branson_synmask.tif'))

# %% SC-FC correlation
fh0, ax0 = plt.subplots(1, 3, figsize=(14, 4))
sns.heatmap(FC.CorrelationMatrix, ax=ax0[0], cmap='cividis', yticklabels=True, xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)', 'shrink': .75})
ax0[0].set_title('Ito-36: functional')
ax0[0].set_aspect('equal')
ax0[0].tick_params(axis='both', which='major', labelsize=6)

sns.heatmap(CorrelationMatrix, ax=ax0[1], cmap='cividis', cbar_kws={'label': 'Functional Correlation (z)', 'shrink': .75})
ax0[1].set_title('Branson-295: functional')
ax0[1].set_aspect('equal')
ax0[1].tick_params(axis='both', which='major', labelsize=8)

tmp = Connectivity_JRC2018.to_numpy()
np.fill_diagonal(tmp, np.nan)
Connectivity_JRC2018 = pd.DataFrame(data=tmp, index=name_list, columns=name_list)
sns.heatmap(np.log10(Connectivity_JRC2018).replace([np.inf, -np.inf], 0), ax=ax0[2], cmap="cividis", rasterized=True, cbar=False)
cb = fh0.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.SymLogNorm(vmin=1, vmax=np.nanmax(Connectivity_JRC2018.to_numpy()), base=10, linthresh=0.1, linscale=1), cmap="cividis"), ax=ax0[2], shrink=0.75, label='Connecting cells')
cb.outline.set_linewidth(0)
ax0[2].set_title('Branson-295: structural')
ax0[2].set_aspect('equal')
ax0[2].tick_params(axis='both', which='major', labelsize=8)

x = Connectivity_JRC2018.to_numpy()[np.triu_indices(len(name_list), k=1)]
keep_inds = np.where(x > 0)
x = np.log10(x[keep_inds])

y = CorrelationMatrix.to_numpy()[np.triu_indices(len(name_list), k=1)][keep_inds]

r, p = pearsonr(x, y)
coef = np.polyfit(x, y, 1)
linfit = np.poly1d(coef)

# # scatter plot
# fh1, ax1 = plt.subplots(1, 1, figsize=(4, 4))
# ax1.plot(10**x, y, color=[0.5, 0.5, 0.5], marker='.', linestyle='none', alpha=1.0)
# xx = np.linspace(x.min(), x.max(), 100)
# ax1.plot(10**xx, linfit(xx), color='k', linewidth=2, marker=None)
# ax1.set_xscale('log')
# ax1.set_xlabel('Cell Count')
# ax1.set_ylabel('Functional corr. (z)')
# ax1.annotate('r = {:.2f}'.format(r), xy=(1, 1.05))
# ax1.tick_params(axis='x', labelsize=10)
# ax1.tick_params(axis='y', labelsize=10)

# hexbin plot
fh1, ax1 = plt.subplots(1, 1, figsize=(5, 4))
hb = ax1.hexbin(x, y, bins='log', gridsize=40)
xx = np.linspace(x.min(), x.max(), 100)
ax1.plot(xx, linfit(xx), color='w', linewidth=2, marker=None)
ax1.set_xlabel('Cell Count')
ax1.set_ylabel('Functional corr. (z)')
ax1.annotate('r = {:.2f}'.format(r), xy=(0.05, 1.05), color='w')
ax1.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='y', labelsize=10)
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$'])
cb = fh1.colorbar(hb, ax=ax1)
# %%

unique_regions = np.unique(name_list)

fh, ax = plt.subplots(4, 5, figsize=(10, 8))
ax = ax.ravel()
[x.set_xticks([]) for x in ax]
[x.set_yticks([]) for x in ax]
ct = 0
for ind, ur in enumerate(unique_regions):
    pull_inds = np.where(ur == name_list)[0]
    if len(pull_inds) > 3:

        intra_sc = Connectivity_JRC2018.loc[ur, ur]
        intra_fc = CorrelationMatrix.loc[ur, ur]
        n_roi = intra_sc.shape[0]
        x = intra_sc.to_numpy()[np.triu_indices(n_roi, k=1)]
        y = intra_fc.to_numpy()[np.triu_indices(n_roi, k=1)]
        ax[ct].plot(x, y, 'k.')
        r, p = pearsonr(np.log10(x[x>0]), y[x>0])
        ax[ct].set_title(ur)
        ax[ct].annotate('r={:.2f}'.format(r), xy=(1.2, 0.9))
        ax[ct].set_ylim([0, 1])
        ax[ct].set_xlim([1, 2000])
        ax[ct].set_xscale('log')
        ax[ct].set_xticks([])

        ct += 1



# %% Atlas vs. synapse density

fh2, ax2 = plt.subplots(2, 3, figsize=(12, 4))
[x.set_xticks([]) for x in ax2.ravel()]
[x.set_yticks([]) for x in ax2.ravel()]
ax2[0, 0].set_ylabel('JRC2018F Atlas')
ax2[1, 0].set_ylabel('Hemibrain TBar density')
ax2[0, 0].imshow(branson_jrc2018.sum(axis=2))
ax2[1, 0].imshow(synmask_jrc2018.sum(axis=2))
ax2[0, 1].imshow(branson_jrc2018.sum(axis=1))
ax2[1, 1].imshow(synmask_jrc2018.sum(axis=1))
ax2[0, 2].imshow(branson_jrc2018.sum(axis=0))
ax2[1, 2].imshow(synmask_jrc2018.sum(axis=0))

# %% atlas alignment images
np.random.seed(1)
tmp = np.random.rand(1000, 3)
tmp[0, :] = [1, 1, 1]
cmap = matplotlib.colors.ListedColormap(tmp)

fh4, ax4 = plt.subplots(1, 3, figsize=(12, 6))

ax4[0].imshow(branson_jfrc2[108, :, :], cmap=cmap, interpolation='None')
ax4[0].set_title('JFRC2')
ax4[0].set_axis_off()

ax4[1].imshow(branson_jrc2018[250, :, :], cmap=cmap, interpolation='None')
ax4[1].set_title('JRC2018')
ax4[1].set_axis_off()

im = ax4[2].imshow(synmask_jrc2018[:, :, :].mean(axis=0), interpolation='None')
ax4[2].set_title('JRC2018 - hemi')
ax4[2].set_axis_off()
cb = fh4.colorbar(im, ax=ax4, shrink=0.25)

# %%
save_dpi = 400
fh0.savefig(os.path.join(analysis_dir, 'figpanels', 'branson_fh0.svg'), format='svg', transparent=True, dpi=save_dpi)
fh1.savefig(os.path.join(analysis_dir, 'figpanels', 'branson_fh1.svg'), format='svg', transparent=True, dpi=save_dpi)
fh2.savefig(os.path.join(analysis_dir, 'figpanels', 'branson_fh2.svg'), format='svg', transparent=True, dpi=save_dpi)
fh4.savefig(os.path.join(analysis_dir, 'figpanels', 'branson_fh4.svg'), format='svg', transparent=True, dpi=save_dpi)

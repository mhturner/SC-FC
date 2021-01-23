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
from scfc import bridge, functional_connectivity
from scipy.stats import pearsonr
import nibabel as nib

data_dir = bridge.getUserConfiguration()['data_dir']
analysis_dir = bridge.getUserConfiguration()['analysis_dir']
token = bridge.getUserConfiguration()['token']

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.2', token=token)
# Get FunctionalConnectivity object
FC = functional_connectivity.FunctionalConnectivity(data_dir=data_dir, fs=1.2, cutoff=0.01, mapping=bridge.getRoiMapping())

# %%


def getBransonNames():
    """."""
    decoder_ring = pd.read_csv(os.path.join(data_dir, 'branson_999_atlas') + '/atlas_roi_values', header=None)
    # Branson regions to include. Do some matching to Ito atlas naming. Sort alphabetically.
    include_regions_branson = ['AL_R', 'OTU_R', 'ATL_R', 'ATL_L', 'AVLP_R', 'LB_R', 'LB_L', 'CRE_R', 'CRE_L', 'EB', 'EPA_R', 'FB', 'GOR_R', 'GOR_L'
                               'IB_R', 'IB_L', 'ICL_R', 'IVLP_R', 'LAL_R', 'LH_R', 'MB_R', 'MB_L', 'NO', 'PB', 'PLP_R', 'PVLP_R', 'SCL_R', 'SIP_R', 'SLP_R', 'SMP_R',
                               'SMP_L', 'SPS_R', 'VES_R', 'WED_R'] # LB = bulb

    include_inds = []
    name_list = []
    for ind in decoder_ring.index:
        row = decoder_ring.loc[ind].values[0]
        region = row.split(':')[0]
        start = row.split(' ')[1]
        end = row.split(' ')[3]
        if region in include_regions_branson:
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

    return include_inds, name_list


def getItoNames():
    """."""
    include_regions_ito = ['AL_R', 'AOTU_R', 'ATL_R', 'ATL_L', 'AVLP_R', 'BU_R', 'BU_L', 'CAN_R', 'CRE_R', 'CRE_L', 'EB', 'EPA_R', 'FB', 'GOR_R', 'GOR_L',
                           'IB_R', 'IB_L', 'ICL_R', 'LAL_R', 'LH_R', 'MB_CA_R', 'MB_ML_R', 'MB_ML_L', 'MB_PED_R', 'MB_VL_R', 'NO', 'PB', 'PLP_R', 'PVLP_R', 'SCL_R', 'SIP_R', 'SLP_R', 'SMP_R',
                           'SMP_L', 'SPS_R', 'VES_R', 'WED_R']

    decoder_ring = pd.read_csv(os.path.join(data_dir, 'ito_68_atlas') + '/Original_Index_panda_full.csv', header=0)

    include_inds = np.array(decoder_ring.loc[[np.where(x == decoder_ring.to_numpy()[:, 0])[0][0] for x in include_regions_ito], 'num'])

    return include_inds, include_regions_ito


def getCmat(response_filepaths, include_inds, name_list):
    """."""
    cmats_z = []
    for resp_fp in response_filepaths:
        tmp = functional_connectivity.getProcessedRegionResponse(resp_fp, cutoff=0.01, fs=1.2)
        resp_included = tmp.reindex(include_inds).to_numpy()

        correlation_matrix = np.corrcoef(resp_included)

        np.fill_diagonal(correlation_matrix, np.nan)
        # fischer z transform (arctanh) and append
        new_cmat_z = np.arctanh(correlation_matrix)
        cmats_z.append(new_cmat_z)

    # Make mean pd Dataframe
    mean_cmat = np.nanmean(np.stack(cmats_z, axis=2), axis=2)
    np.fill_diagonal(mean_cmat, np.nan)
    CorrelationMatrix = pd.DataFrame(data=mean_cmat, index=name_list, columns=name_list)

    return CorrelationMatrix, cmats_z


# %%  # # # BRANSON ATLAS # # #

response_filepaths = glob.glob(os.path.join(data_dir, 'branson_responses') + '/' + '*.pkl')

include_inds_branson, name_list_branson = getBransonNames()

CorrelationMatrix_branson, cmats_branson = getCmat(response_filepaths, include_inds_branson, name_list_branson)

# # Compute corr between mean and individual fly cmats
meanvals = CorrelationMatrix_branson.to_numpy()[np.triu_indices(len(name_list_branson), k=1)]
r_val = []
for cm in cmats_branson:
    current = cm[np.triu_indices(len(name_list_branson), k=1)]
    keep_inds = np.where(~np.isnan(current))
    r, p = pearsonr(meanvals[keep_inds], current[keep_inds])
    r_val.append(r)

print('Individual to mean r = {:.2f} +/- {:.2f}'.format(np.mean(r_val), np.std(r_val)))


# %% Ito correlation matrix
response_filepaths = glob.glob(os.path.join(data_dir, 'ito_responses') + '/' + '*.pkl')
include_inds_ito, name_list_ito = getItoNames()

CorrelationMatrix_ito, cmats_ito = getCmat(response_filepaths, include_inds_ito, name_list_ito)

# # Compute corr between mean and individual fly cmats
meanvals = CorrelationMatrix_ito.to_numpy()[np.triu_indices(len(name_list_ito), k=1)]
r_val = []
for cm in cmats_ito:
    current = cm[np.triu_indices(len(name_list_ito), k=1)]
    keep_inds = np.where(~np.isnan(current))
    r, p = pearsonr(meanvals[keep_inds], current[keep_inds])
    r_val.append(r)

print('Individual to mean r = {:.2f} +/- {:.2f}'.format(np.mean(r_val), np.std(r_val)))

# %% For comparison: compute corr between mean and individual fly cmats for Ito atlas data

meanvals = FC.CorrelationMatrix.to_numpy()[np.triu_indices(36, k=1)]
t_inds = np.where(~np.isnan(meanvals))[0]
r_vals = []
for c_ind in range(FC.cmats.shape[2]):
    cmat = FC.cmats[:, :, c_ind]
    r, p = pearsonr(meanvals[t_inds], cmat[np.triu_indices(36, k=1)][t_inds])
    r_vals.append(r)

print('r = {:.2f} +/- {:.2f}'.format(np.mean(r_vals), np.std(r_vals)))

# %% Load cellcount_matrix from R / natverse script
#   Branson atlas
JRC2018_branson_cellcount_matrix = pd.read_csv(os.path.join(data_dir, 'hemi_2_atlas', 'JRC2018_branson_cellcount_matrix.csv'), header=0).to_numpy()[:, 1:]
JRC2018_branson_cellcount_matrix = pd.DataFrame(data=JRC2018_branson_cellcount_matrix, index=np.arange(1, 1000), columns=np.arange(1, 1000))

# filter and sort count_matrix by include_inds
Branson_JRC2018 = pd.DataFrame(data=np.zeros_like(CorrelationMatrix_branson), index=name_list_branson, columns=name_list_branson)

for s_ind, src in enumerate(include_inds_branson):
    for t_ind, trg in enumerate(include_inds_branson):
        Branson_JRC2018.iloc[s_ind, t_ind] = JRC2018_branson_cellcount_matrix.loc[src, trg]

#   Ito atlas
JRC2018_ito_cellcount_matrix = pd.read_csv(os.path.join(data_dir, 'hemi_2_atlas', 'JRC2018_ito_cellcount_matrix.csv'), header=0).to_numpy()[:, 1:]
JRC2018_ito_cellcount_matrix = pd.DataFrame(data=JRC2018_ito_cellcount_matrix, index=np.arange(1, 87), columns=np.arange(1, 87))

# filter and sort count_matrix by include_inds
Ito_JRC2018 = pd.DataFrame(data=np.zeros_like(CorrelationMatrix_ito), index=name_list_ito, columns=name_list_ito)

for s_ind, src in enumerate(include_inds_ito):
    for t_ind, trg in enumerate(include_inds_ito):
        Ito_JRC2018.iloc[s_ind, t_ind] = JRC2018_ito_cellcount_matrix.loc[src, trg]


# %%

fh, ax = plt.subplots(2, 3, figsize=(14, 8))
sns.heatmap(CorrelationMatrix_branson, cmap='cividis', ax=ax[0, 0])
sns.heatmap(Branson_JRC2018, cmap='cividis', ax=ax[1, 0])

sns.heatmap(CorrelationMatrix_ito, cmap='cividis', ax=ax[0, 1])
sns.heatmap(Ito_JRC2018, cmap='cividis', ax=ax[1, 1])


sns.heatmap(FC.CorrelationMatrix, cmap='cividis', ax=ax[0, 2])

# %%



plt.plot(x, y, 'ko')

r, p = pearsonr(x, y)
r



# %%
# load synmask tifs and atlases
synmask_jrc2018 = io.imread(os.path.join(data_dir, 'hemi_2_atlas', 'JRC2018_synmask.tif'))
synmask_jfrc2 = io.imread(os.path.join(data_dir, 'hemi_2_atlas', 'JFRC2_synmask.tif'))

branson_jfrc2 = io.imread(os.path.join(data_dir, 'template_brains', 'AnatomySubCompartments20150108_ms999centers.tif'))
branson_jrc2018 = io.imread(os.path.join(data_dir, 'template_brains', '2018_999_atlas.tif'))

ito_jfrc2 = io.imread(os.path.join(data_dir, 'template_brains', 'JFRCtempate2010.mask130819_Original.tif'))
ito_jrc2018 = io.imread(os.path.join(data_dir, 'template_brains', 'ito_2018.tif'))

# %% SC-FC correlation
fh0, ax0 = plt.subplots(2, 2, figsize=(10, 8))
sns.heatmap(CorrelationMatrix_ito, ax=ax0[0, 0], cmap='cividis', yticklabels=True, xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)', 'shrink': .75})
# ax0[0, 0].set_title('Ito-36: functional')
ax0[0, 0].set_aspect('equal')
ax0[0, 0].tick_params(axis='both', which='major', labelsize=6)

tmp = Ito_JRC2018.to_numpy()
np.fill_diagonal(tmp, np.nan)
conn_mat = pd.DataFrame(data=tmp, index=name_list_ito, columns=name_list_ito)
sns.heatmap(np.log10(conn_mat).replace([np.inf, -np.inf], 0), ax=ax0[0, 1], cmap="cividis", yticklabels=True, xticklabels=True, rasterized=True, cbar=False)
cb = fh0.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.SymLogNorm(vmin=1, vmax=np.nanmax(conn_mat.to_numpy()), base=10, linthresh=0.1, linscale=1), cmap="cividis"), ax=ax0[0, 1], shrink=0.75, label='Connecting cells')
cb.outline.set_linewidth(0)
# ax0[0, 1].set_title('Ito-36: structural')
ax0[0, 1].set_aspect('equal')
ax0[0, 1].tick_params(axis='both', which='major', labelsize=6)


sns.heatmap(CorrelationMatrix_branson, ax=ax0[1, 0], cmap='cividis', cbar_kws={'label': 'Functional Correlation (z)', 'shrink': .75})
# ax0[1, 0].set_title('Branson-295: functional')
ax0[1, 0].set_aspect('equal')
ax0[1, 0].tick_params(axis='both', which='major', labelsize=8)

tmp = Branson_JRC2018.to_numpy()
np.fill_diagonal(tmp, np.nan)
conn_mat = pd.DataFrame(data=tmp, index=name_list_branson, columns=name_list_branson)
sns.heatmap(np.log10(conn_mat).replace([np.inf, -np.inf], 0), ax=ax0[1, 1], cmap="cividis", rasterized=True, cbar=False)
cb = fh0.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.SymLogNorm(vmin=1, vmax=np.nanmax(conn_mat.to_numpy()), base=10, linthresh=0.1, linscale=1), cmap="cividis"), ax=ax0[1, 1], shrink=0.75, label='Connecting cells')
cb.outline.set_linewidth(0)
# ax0[1, 1].set_title('Ito-36: structural')
ax0[1, 1].set_aspect('equal')
ax0[1, 1].tick_params(axis='both', which='major', labelsize=8)

fh0.savefig(os.path.join(analysis_dir, 'figpanels', 'branson_fh0.png'), format='png', transparent=True, dpi=400)

# %%

# corr: ito
x = Ito_JRC2018.to_numpy()[np.triu_indices(len(name_list_ito), k=1)]
keep_inds = np.where(x > 0)
x = np.log10(x[keep_inds])
y = CorrelationMatrix_ito.to_numpy()[np.triu_indices(len(name_list_ito), k=1)]
y = y[keep_inds]

r, p = pearsonr(x, y)
coef = np.polyfit(x, y, 1)
linfit = np.poly1d(coef)

# scatter plot
fh1, ax1 = plt.subplots(2, 1, figsize=(4, 8))
ax1[0].plot(10**x, y, color=[0.5, 0.5, 0.5], marker='.', linestyle='none', alpha=1.0)
xx = np.linspace(x.min(), x.max(), 100)
ax1[0].plot(10**xx, linfit(xx), color='k', linewidth=2, marker=None)
ax1[0].set_xscale('log')
ax1[0].set_xlabel('Cell Count')
ax1[0].set_ylabel('Functional corr. (z)')
ax1[0].annotate('r = {:.2f}'.format(r), xy=(1, 1.05))
ax1[0].tick_params(axis='x', labelsize=10)
ax1[0].tick_params(axis='y', labelsize=10)

# corr: branson
x = Branson_JRC2018.to_numpy()[np.triu_indices(len(name_list_branson), k=1)]
keep_inds = np.where(x > 0)
x = np.log10(x[keep_inds])
y = CorrelationMatrix_branson.to_numpy()[np.triu_indices(len(name_list_branson), k=1)]
y = y[keep_inds]

r, p = pearsonr(x, y)
coef = np.polyfit(x, y, 1)
linfit = np.poly1d(coef)

# scatter plot
# ax1[1].plot(10**x, y, color=[0.5, 0.5, 0.5], marker='.', linestyle='none', alpha=1.0)
# xx = np.linspace(x.min(), x.max(), 100)
# ax1[1].plot(10**xx, linfit(xx), color='k', linewidth=2, marker=None)
# ax1[1].set_xscale('log')
# ax1[1].set_xlabel('Cell Count')
# ax1[1].set_ylabel('Functional corr. (z)')
# ax1[1].annotate('r = {:.2f}'.format(r), xy=(1, 1.05))
# ax1[1].tick_params(axis='x', labelsize=10)
# ax1[1].tick_params(axis='y', labelsize=10)

# hexbin plot
hb = ax1[1].hexbin(x, y, bins='log', gridsize=40)
xx = np.linspace(x.min(), x.max(), 100)
ax1[1].plot(xx, linfit(xx), color='w', linewidth=2, marker=None)
ax1[1].set_xlabel('Cell Count')
ax1[1].set_ylabel('Functional corr. (z)')
ax1[1].annotate('r = {:.2f}'.format(r), xy=(0.05, 1.05), color='w')
ax1[1].tick_params(axis='x', labelsize=10)
ax1[1].tick_params(axis='y', labelsize=10)
ax1[1].set_xticks([0, 1, 2, 3])
ax1[1].set_xticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$'])
# cb = fh1.colorbar(hb, ax=ax1)

fh1.savefig(os.path.join(analysis_dir, 'figpanels', 'branson_fh1.png'), format='png', transparent=True, dpi=400)
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

fh4, ax4 = plt.subplots(3, 2, figsize=(12, 8))
[x.set_xticks([]) for x in ax4.ravel()]
[x.set_yticks([]) for x in ax4.ravel()]
ax4[0, 0].set_title('JFRC2')
ax4[0, 1].set_title('JRC2018')

ax4[0, 0].set_ylabel('Branson')
ax4[1, 0].set_ylabel('Ito')
ax4[2, 0].set_ylabel('Synapses')

ax4[0, 0].imshow(branson_jfrc2[108, :, :], cmap=cmap, interpolation='None')
ax4[0, 1].imshow(branson_jrc2018[250, :, :], cmap=cmap, interpolation='None')


im = ax4[2, 1].imshow(synmask_jrc2018[250:300, :, :].mean(axis=0), interpolation='None')
cb = fh4.colorbar(im, ax=ax4, shrink=0.3)

# Ito atlas
np.random.seed(1)
tmp = np.random.rand(68, 3)
tmp[0, :] = [1, 1, 1]
cmap = matplotlib.colors.ListedColormap(tmp)

ax4[1, 0].imshow(ito_jfrc2[108, :, :], cmap=cmap, interpolation='None')

ax4[1, 1].imshow(ito_jrc2018[250, :, :], cmap=cmap, interpolation='None')



# %%
# save_dpi = 400
# fh0.savefig(os.path.join(analysis_dir, 'figpanels', 'branson_fh0.svg'), format='svg', transparent=True, dpi=save_dpi)
# fh1.savefig(os.path.join(analysis_dir, 'figpanels', 'branson_fh1.svg'), format='svg', transparent=True, dpi=save_dpi)
# fh2.savefig(os.path.join(analysis_dir, 'figpanels', 'branson_fh2.svg'), format='svg', transparent=True, dpi=save_dpi)
# fh4.savefig(os.path.join(analysis_dir, 'figpanels', 'branson_fh4.svg'), format='svg', transparent=True, dpi=save_dpi)

# %%
from scipy.stats import zscore

# # compute difference matrix using original, asymmetric anatomical connectivity matrix
anatomical_mat = Branson_JRC2018.to_numpy().copy()
functional_mat = CorrelationMatrix_branson.to_numpy().copy()
np.fill_diagonal(functional_mat, 0)
np.fill_diagonal(anatomical_mat, 0)

# log transform anatomical connectivity values
keep_inds_diff = np.where(anatomical_mat > 0)
functional_adjacency_diff = functional_mat[keep_inds_diff]
anatomical_adjacency_diff = np.log10(anatomical_mat[keep_inds_diff])

F_zscore = zscore(functional_adjacency_diff)
A_zscore = zscore(anatomical_adjacency_diff)
diff = F_zscore - A_zscore

diff_m = np.zeros_like(anatomical_mat)
diff_m[keep_inds_diff] = diff
DifferenceMatrix = pd.DataFrame(data=diff_m, index=name_list_branson, columns=name_list_branson)

sns.heatmap(DifferenceMatrix, cmap='cividis')
# %%
means = DifferenceMatrix.mean(axis=1).groupby(DifferenceMatrix.columns).mean()
stds = DifferenceMatrix.mean(axis=1).groupby(DifferenceMatrix.columns).std()



DifferenceMatrix.groupby(by=DifferenceMatrix.columns, axis=1).mean()

fh, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(DifferenceMatrix.groupby(by=DifferenceMatrix.columns, axis=1).mean().mean(axis=0).sort_values(ascending=False), 'ko')
ax.tick_params(axis='both', which='major', labelsize=10, rotation=90)

# %%

from scipy.stats import zscore

# # compute difference matrix using original, asymmetric anatomical connectivity matrix
anatomical_mat = Ito_JRC2018.to_numpy().copy()
functional_mat = CorrelationMatrix_ito.to_numpy().copy()
np.fill_diagonal(functional_mat, 0)
np.fill_diagonal(anatomical_mat, 0)

# log transform anatomical connectivity values
keep_inds_diff = np.where(anatomical_mat > 0)
functional_adjacency_diff = functional_mat[keep_inds_diff]
anatomical_adjacency_diff = np.log10(anatomical_mat[keep_inds_diff])

F_zscore = zscore(functional_adjacency_diff)
A_zscore = zscore(anatomical_adjacency_diff)
diff = F_zscore - A_zscore

diff_m = np.zeros_like(anatomical_mat)
diff_m[keep_inds_diff] = diff
DifferenceMatrix = pd.DataFrame(data=diff_m, index=name_list_ito, columns=name_list_ito)

sns.heatmap(DifferenceMatrix)
# %%


regions = {'AL/LH': ['AL_R', 'LH_R'],
           'MB': ['MB_CA_R', 'MB_ML_R', 'MB_ML_L', 'MB_PED_R', 'MB_VL_R'],
           'CX': ['EB', 'FB', 'PB', 'NO'],
           'LX': ['BU_L', 'BU_R', 'LAL_R'],
           'INP': ['CRE_L', 'CRE_R', 'SCL_R', 'ICL_R', 'IB_L', 'IB_R', 'ATL_L', 'ATL_R'],
           'VMNP': ['VES_R', 'EPA_R', 'GOR_L', 'GOR_R', 'SPS_R'],
           'SNP': ['SLP_R', 'SIP_R', 'SMP_R', 'SMP_L'],
           'VLNP': ['AOTU_R', 'AVLP_R', 'PVLP_R', 'PLP_R', 'WED_R'],
           # 'PENP': ['CAN_R'], # only one region
           }


# log transform anatomical connectivity values
anatomical_mat = Ito_JRC2018.to_numpy().copy()
np.fill_diagonal(anatomical_mat, 0)

keep_inds_diff = np.where(anatomical_mat > 0)
anatomical_adj = np.log10(anatomical_mat[keep_inds_diff])

diff_by_region = []
for c_ind in range(len(cmats_ito)): # loop over fly
    cmat = cmats_ito[c_ind]
    functional_adj = cmat[keep_inds_diff]

    F_zscore_fly = zscore(functional_adj)
    A_zscore_fly = zscore(anatomical_adj)

    diff = F_zscore_fly - A_zscore_fly

    diff_m = np.zeros_like(anatomical_mat)
    diff_m[keep_inds_diff] = diff
    diff_by_region.append(diff_m.mean(axis=0))

diff_by_region = np.vstack(diff_by_region).T  # region x fly
sort_inds = np.argsort(diff_by_region.mean(axis=1))[::-1]
diff_by_region.mean(axis=1)
colors = sns.color_palette('deep', 8)
fig4_3, ax = plt.subplots(1, 1, figsize=(6.0, 3.0))

plot_position = 0
for r_ind in sort_inds:
    current_roi = name_list_ito[r_ind]
    if current_roi == 'CAN_R':
        continue

    super_region_ind = np.where([current_roi in regions[reg_key] for reg_key in regions.keys()])[0][0]
    color = colors[super_region_ind]

    new_mean = np.mean(diff_by_region[r_ind, :])
    new_err = np.std(diff_by_region[r_ind, :]) / np.sqrt(diff_by_region.shape[1])
    ax.plot(plot_position, new_mean, linestyle='None', marker='o', color=color)
    ax.plot([plot_position, plot_position], [new_mean-new_err, new_mean+new_err], linestyle='-', linewidth=2, marker='None', color=color)
    ax.annotate(current_roi, (plot_position-0.25, 1.2), rotation=90, fontsize=8, color=color, fontweight='bold')

    plot_position += 1

ax.set_ylim([-1.1, 1.3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.axhline(0, color=[0.8, 0.8, 0.8], linestyle='-', zorder=0)
ax.set_ylabel('Region avg. diff.\n(FC - SC)')
ax.set_xticks([])

sns.palplot(colors)
fig4_3.savefig(os.path.join(analysis_dir, 'figpanels', 'branson_fh4_3.png'), format='png', transparent=True, dpi=400)

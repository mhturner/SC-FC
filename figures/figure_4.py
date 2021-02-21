"""
Turner, Mann, Clandinin: Figure generation script: Fig. 4.

https://github.com/mhturner/SC-FC
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import zscore, spearmanr
import pandas as pd
import seaborn as sns
import glob
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_validate

from scfc import bridge, anatomical_connectivity, functional_connectivity
from matplotlib import rcParams
rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})
rcParams.update({'axes.spines.right': False})
rcParams.update({'axes.spines.top': False})
rcParams['svg.fonttype'] = 'none'  # let illustrator handle the font type

data_dir = bridge.getUserConfiguration()['data_dir']
analysis_dir = bridge.getUserConfiguration()['analysis_dir']

plot_colors = plt.get_cmap('tab10')(np.arange(8)/8)
save_dpi = 400

# %% Difference matrix
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')
include_inds_ito, name_list_ito = bridge.getItoNames()

# # compute difference matrix using original, asymmetric anatomical connectivity matrix
anatomical_mat = anatomical_connectivity.getAtlasConnectivity(include_inds_ito, name_list_ito, 'ito').to_numpy().copy()
np.fill_diagonal(anatomical_mat, 0)

response_filepaths = glob.glob(os.path.join(data_dir, 'ito_responses') + '/' + '*.pkl')
functional_mat, cmats_z = functional_connectivity.getCmat(response_filepaths, include_inds_ito, name_list_ito)
functional_mat = functional_mat.to_numpy().copy()
np.fill_diagonal(functional_mat, 0)


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


# %% sort difference matrix by most to least different rois
diff_by_roi = DifferenceMatrix.mean()

sort_inds = np.argsort(diff_by_roi)[::-1]
sort_keys = DifferenceMatrix.index[sort_inds]
sorted_diff = pd.DataFrame(data=np.zeros_like(DifferenceMatrix), columns=sort_keys, index=sort_keys)
for r_ind, r_key in enumerate(sort_keys):
    for c_ind, c_key in enumerate(sort_keys):
        sorted_diff.iloc[r_ind, c_ind]=DifferenceMatrix.loc[[r_key], [c_key]].to_numpy()


fig4_0, ax = plt.subplots(1, 1, figsize=(1.6, 1.6))
lim = np.nanmax(np.abs(DifferenceMatrix.to_numpy().ravel()))
ax.scatter(10**anatomical_adjacency_diff, functional_adjacency_diff, color='k', marker='.', rasterized=True, s=4)
ax.set_xscale('log')
ax.set_xlim([np.min(10**anatomical_adjacency_diff), np.max(10**anatomical_adjacency_diff)])
ax.set_xlabel('SC (cells)', fontsize=10)
ax.set_ylabel('FC (z)', fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=8)


fig4_1, ax = plt.subplots(1, 1, figsize=(1.6, 1.6))
lim = np.nanmax(np.abs(DifferenceMatrix.to_numpy().ravel()))
ax.scatter(A_zscore, F_zscore, c=diff, cmap="RdBu_r", vmin=-lim, vmax=lim, marker='.', rasterized=True, s=4)
ax.plot([-3.5, 3.5], [-3.5, 3.5], 'k-')
ax.axhline(color='k', zorder=0, alpha=0.5)
ax.axvline(color='k', zorder=0, alpha=0.5)
ax.set_xticks([-2, 2])
ax.set_yticks([-2, 2])
ax.set_xticklabels(['-2$\sigma$', '+2$\sigma$'])
ax.set_yticklabels(['-2$\sigma$', '+2$\sigma$'])
ax.set_xlabel('SC', fontsize=10)
ax.set_ylabel('FC', fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.set_aspect(1)

fig4_2, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.heatmap(sorted_diff, ax=ax,
            yticklabels=[bridge.displayName(x) for x in sorted_diff.columns],
            xticklabels=[bridge.displayName(x) for x in sorted_diff.index],
            cbar_kws={'label': 'Difference (FC - SC)', 'shrink': .65}, cmap="RdBu_r", rasterized=True, vmin=-lim, vmax=lim)
ax.set_aspect('equal')
ax.tick_params(axis='both', which='major', labelsize=6)

fig4_0.savefig(os.path.join(analysis_dir, 'figpanels', 'fig4_0.svg'), format='svg', transparent=True, dpi=save_dpi)
fig4_1.savefig(os.path.join(analysis_dir, 'figpanels', 'fig4_1.svg'), format='svg', transparent=True, dpi=save_dpi)
fig4_2.savefig(os.path.join(analysis_dir, 'figpanels', 'fig4_2.svg'), format='svg', transparent=True, dpi=save_dpi)


# %% Average diff for each region, cluster and sort by super-regions

regions = {'AL/LH': ['AL_R', 'LH_R'],
           'MB': ['MB_CA_R', 'MB_ML_R', 'MB_ML_L', 'MB_PED_R', 'MB_VL_R'],
           'CX': ['EB', 'FB', 'PB', 'NO'],
           'LX': ['BU_L', 'BU_R', 'LAL_R'],
           'INP': ['CRE_L', 'CRE_R', 'SCL_R', 'ICL_R', 'IB_L', 'IB_R', 'ATL_L', 'ATL_R'],
           'VMNP': ['VES_R', 'EPA_R', 'GOR_L', 'GOR_R', 'SPS_R'],
           'SNP': ['SLP_R', 'SIP_R', 'SMP_R', 'SMP_L'],
           'VLNP': ['AOTU_R', 'AVLP_R', 'PVLP_R', 'PLP_R', 'WED_R'],
           }


# log transform anatomical connectivity values

anatomical_mat = anatomical_connectivity.getAtlasConnectivity(include_inds_ito, name_list_ito, 'ito').to_numpy().copy()
np.fill_diagonal(anatomical_mat, 0)
keep_inds_diff = np.where(anatomical_mat > 0)
anatomical_adj = np.log10(anatomical_mat[keep_inds_diff])

diff_by_region = []
for c_ind in range(len(cmats_z)): # loop over fly
    cmat = cmats_z[c_ind]
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
fig4_3, ax = plt.subplots(1, 1, figsize=(5.5, 3.0))

plot_position = 0
for r_ind in sort_inds:
    current_roi = name_list_ito[r_ind]
    if current_roi == 'CAN_R': # All by itself in the PENP
        continue

    super_region_ind = np.where([current_roi in regions[reg_key] for reg_key in regions.keys()])[0][0]
    color = colors[super_region_ind]

    new_mean = np.mean(diff_by_region[r_ind, :])
    new_err = np.std(diff_by_region[r_ind, :]) / np.sqrt(diff_by_region.shape[1])
    ax.plot(plot_position, new_mean, linestyle='None', marker='o', color=color)
    ax.plot([plot_position, plot_position], [new_mean-new_err, new_mean+new_err], linestyle='-', linewidth=2, marker='None', color=color)
    ax.annotate(bridge.displayName(current_roi), (plot_position-0.25, 1.1), rotation=90, fontsize=8, color=color, fontweight='bold')

    plot_position += 1

ax.set_ylim([-1.1, 1.1])
ax.spines['right'].set_visible(False)
ax.axhline(0, color=[0.8, 0.8, 0.8], linestyle='-', zorder=0)
ax.set_ylabel('Region avg. diff.\n(FC - SC)')
ax.set_xticks([])

sns.palplot(colors)
# np.array(colors)
fig4_3.savefig(os.path.join(analysis_dir, 'figpanels', 'fig4_3.svg'), format='svg', transparent=True, dpi=save_dpi)

# Groups of ROIs sig different than rest of distr.?
alpha = 0.01
m = len(regions)
p_cutoff = alpha / m
xx = diff_by_region.mean(axis=1)
for r in regions:
    rois = regions[r]

    h, p = stats.ttest_ind(diff_by_region[~np.array([x in rois for x in name_list_ito]), :].ravel(), diff_by_region[np.array([x in rois for x in name_list_ito]), :].ravel())
    print(r)
    print(p)
    print(p < p_cutoff)
    print('------------')


# %%

# Shortest path distance:
anat_connect = anatomical_connectivity.getAtlasConnectivity(include_inds_ito, name_list_ito, 'ito')
shortest_path_dist, shortest_path_steps, shortest_path_weight, hub_count = bridge.getShortestPathStats(anat_connect)

shortest_path_dist = shortest_path_dist.to_numpy()[~np.eye(len(name_list_ito), dtype=bool)]

# Direct distance:
tmp = anatomical_connectivity.getAtlasConnectivity(include_inds_ito, name_list_ito, 'ito').to_numpy().copy()
np.fill_diagonal(tmp, 0)
direct_dist = (1/tmp)[~np.eye(len(name_list_ito), dtype=bool)]
direct_dist[np.isinf(direct_dist)] = np.nan

# FC-SC difference:
diff = DifferenceMatrix.to_numpy()[~np.eye(len(name_list_ito), dtype=bool)]

fig4_4, ax = plt.subplots(1, 2, figsize=(6, 3))
lim = np.nanmax(np.abs(DifferenceMatrix.to_numpy().ravel()))
sc = ax[0].scatter(direct_dist, shortest_path_dist, c=diff, s=12, alpha=1.0, cmap='RdBu_r', marker='.', vmin=-lim, vmax=lim, rasterized=True)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('Direct distance')
ax[0].set_ylabel('Shortest path distance')
ax[0].plot([2e-4, 1], [2e-4, 1], color='k', linewidth=2, alpha=1.0, linestyle='-', zorder=0)
ax[0].set_ylim([2e-4, 6e-2])
fig4_4.colorbar(sc, ax=ax[0])

shortest_path_factor = direct_dist / shortest_path_dist
x = shortest_path_factor
y = diff
keep_inds = np.where(x>1)

x = x[keep_inds]
y = y[keep_inds]

ax[1].scatter(x, y, c=diff[keep_inds], marker='.', s=12, alpha=1.0, linestyle='None', cmap='RdBu_r', vmin=-lim, vmax=lim, rasterized=True)
ax[1].axhline(color='k', linestyle='--')
ax[1].set_xscale('log')
ax[1].set_xlabel('Indirect path factor')
ax[1].set_ylabel('Diff. (FC-SC)')
r, p = spearmanr(x, y)
ax[1].annotate(r'$\rho$={:.2f}'.format(r), (90, 2.5))
ax[1].set_xticks([1, 10, 100])

bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 7)
num_bins = len(bins)-1

for b_ind in range(num_bins):
    b_start = bins[b_ind]
    b_end = bins[b_ind+1]
    inds = np.where(np.logical_and(x > b_start, x < b_end))
    bin_mean_x = x[inds].mean()
    bin_mean_y = y[inds].mean()
    ax[1].plot(bin_mean_x, bin_mean_y, color='k', marker='s', alpha=1, linestyle='none')

    err_x = x[inds].std()/np.sqrt(len(inds))
    ax[1].plot([bin_mean_x - err_x, bin_mean_x + err_x], [bin_mean_y, bin_mean_y], linestyle='-', marker='None', color='k', alpha=1, linewidth=2)

    err_y = y[inds].std()/np.sqrt(len(inds))
    ax[1].plot([bin_mean_x, bin_mean_x], [bin_mean_y - err_y, bin_mean_y + err_y], linestyle='-', marker='None', color='k', alpha=1, linewidth=2)

fig4_4.savefig(os.path.join(analysis_dir, 'figpanels', 'fig4_4.svg'), format='svg', transparent=True, dpi=save_dpi)

# %% supp: multiple regression model on direct + shortest path


def fitLinReg(x, y):
    """
    Fit multiple linear regression model.

    x: indep variable / predictors
    y: dep variable to predict
    """
    rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=0)
    regressor = LinearRegression()
    regressor.fit(x, y)
    pred = regressor.predict(x)

    cv_results = cross_validate(regressor, x, measured_fc, cv=rkf, scoring='r2')
    avg_r2 = cv_results['test_score'].mean()
    err_r2 = cv_results['test_score'].std()

    return pred, avg_r2, err_r2


metrics = ['cellcount', 'tbar']
for ind in range(2):
    metric = metrics[ind]

    # direct connectivity
    Structural_Matrix = anatomical_connectivity.getAtlasConnectivity(include_inds_ito, name_list_ito, 'ito', metric=metric).to_numpy().copy()
    Structural_Matrix = (Structural_Matrix + Structural_Matrix.T) / 2 # symmetrize

    keep_inds = np.where(Structural_Matrix[np.triu_indices(len(name_list_ito), k=1)] > 0)
    direct_connect = np.log10(Structural_Matrix[np.triu_indices(len(name_list_ito), k=1)][keep_inds])

    # shortest path
    anat_connect = anatomical_connectivity.getAtlasConnectivity(include_inds_ito, name_list_ito, 'ito', metric=metric)
    measured_sp, measured_steps, _, measured_hub = bridge.getShortestPathStats(anat_connect)
    shortest_path = np.log10(((measured_sp.T + measured_sp.T)/2).to_numpy()[np.triu_indices(len(name_list_ito), k=1)][keep_inds])

    # Predicted: FC
    response_filepaths = glob.glob(os.path.join(data_dir, 'ito_responses') + '/' + '*.pkl')
    Functional_Matrix, _ = functional_connectivity.getCmat(response_filepaths, include_inds_ito, name_list_ito)
    measured_fc = Functional_Matrix.to_numpy()[np.triu_indices(len(name_list_ito), k=1)][keep_inds]

    figS4_0, ax = plt.subplots(1, 3, figsize=(9, 3))

    # # # # Direct only # # #
    x = np.vstack([direct_connect]).T
    pred, avg_r2, err_r2 = fitLinReg(x=x, y=measured_fc)

    print('r2 = {:.2f}+/-{:.2f}'.format(avg_r2, err_r2))
    ax[0].plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
    ax[0].scatter(pred, measured_fc, color='k', marker='.', alpha=1.0, rasterized=True)
    ax[0].annotate('$r^2$={:.2f}'.format(avg_r2), (-0.15, 0.95))
    ax[0].set_ylabel('Measured FC (z)')
    ax[0].set_xlabel('Predicted FC (z)')
    ax[0].set_xlim([-0.2, 1.0])
    ax[0].set_aspect('equal')
    ax[0].set_title('Direct connectivity')

    # # # Shortest only # # #
    x = np.vstack([shortest_path]).T
    pred, avg_r2, err_r2 = fitLinReg(x=x, y=measured_fc)

    print('r2 = {:.2f}+/-{:.2f}'.format(avg_r2, err_r2))
    ax[1].plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
    ax[1].scatter(pred, measured_fc, color='k', marker='.', alpha=1.0, rasterized=True)
    ax[1].annotate('$r^2$={:.2f}'.format(avg_r2), (-0.15, 0.95))
    ax[1].set_ylabel('Measured FC (z)')
    ax[1].set_xlabel('Predicted FC (z)')
    ax[1].set_xlim([-0.2, 1.0])
    ax[1].set_aspect('equal')
    ax[1].set_title('Shortest path')

    # # # Shortest path + direct # # #
    x = np.vstack([direct_connect,
                   shortest_path]).T
    pred, avg_r2, err_r2 = fitLinReg(x=x, y=measured_fc)

    print('r2 = {:.2f}+/-{:.2f}'.format(avg_r2, err_r2))
    ax[2].plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
    ax[2].scatter(pred, measured_fc, color='k', marker='.', alpha=1.0, rasterized=True)
    ax[2].annotate('$r^2$={:.2f}'.format(avg_r2), (-0.15, 0.95))
    ax[2].set_ylabel('Measured FC (z)')
    ax[2].set_xlabel('Predicted FC (z)')
    ax[2].set_xlim([-0.2, 1.0])
    ax[2].set_aspect('equal')
    ax[2].set_title('Direct + shortest path')

    figS4_0.savefig(os.path.join(analysis_dir, 'figpanels', 'figS4_{}.svg'.format(ind)), format='svg', transparent=True, dpi=save_dpi)

# %%

response_filepaths = glob.glob(os.path.join(data_dir, 'branson_responses') + '/' + '*.pkl')

include_inds_branson, name_list_branson = bridge.getBransonNames()

CorrelationMatrix_branson, cmats_branson = functional_connectivity.getCmat(response_filepaths, include_inds_branson, name_list_branson)
Branson_JRC2018 = anatomical_connectivity.getAtlasConnectivity(include_inds_branson, name_list_branson, 'branson')

# %%
# Shortest path distance:
shortest_path_dist = bridge.getShortestPathStats(Branson_JRC2018)

# %%
shortest_path_dist = shortest_path_dist.to_numpy()[~np.eye(len(name_list_ito), dtype=bool)]

# Direct distance:
tmp = anatomical_connectivity.getAtlasConnectivity(include_inds_ito, name_list_ito, 'ito').to_numpy().copy()
np.fill_diagonal(tmp, 0)
direct_dist = (1/tmp)[~np.eye(len(name_list_ito), dtype=bool)]
direct_dist[np.isinf(direct_dist)] = np.nan

# FC-SC difference:
diff = DifferenceMatrix.to_numpy()[~np.eye(len(name_list_ito), dtype=bool)]

figS4_1, ax = plt.subplots(1, 2, figsize=(6, 3))
lim = np.nanmax(np.abs(DifferenceMatrix.to_numpy().ravel()))
sc = ax[0].scatter(direct_dist, shortest_path_dist, c=diff, s=12, alpha=1.0, cmap='RdBu_r', marker='.', vmin=-lim, vmax=lim, rasterized=True)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('Direct distance')
ax[0].set_ylabel('Shortest path distance')
ax[0].plot([2e-4, 1], [2e-4, 1], color='k', linewidth=2, alpha=1.0, linestyle='-', zorder=0)
ax[0].set_ylim([2e-4, 6e-2])
fig4_4.colorbar(sc, ax=ax[0])

shortest_path_factor = direct_dist / shortest_path_dist
x = shortest_path_factor
y = diff
keep_inds = np.where(x>1)

x = x[keep_inds]
y = y[keep_inds]

ax[1].scatter(x, y, c=diff[keep_inds], marker='.', s=12, alpha=1.0, linestyle='None', cmap='RdBu_r', vmin=-lim, vmax=lim, rasterized=True)
ax[1].axhline(color='k', linestyle='--')
ax[1].set_xscale('log')
ax[1].set_xlabel('Indirect path factor')
ax[1].set_ylabel('Diff. (FC-SC)')
r, p = spearmanr(x, y)
ax[1].annotate(r'$\rho$={:.2f}'.format(r), (90, 2.5))
ax[1].set_xticks([1, 10, 100])

bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 7)
num_bins = len(bins)-1

for b_ind in range(num_bins):
    b_start = bins[b_ind]
    b_end = bins[b_ind+1]
    inds = np.where(np.logical_and(x > b_start, x < b_end))
    bin_mean_x = x[inds].mean()
    bin_mean_y = y[inds].mean()
    ax[1].plot(bin_mean_x, bin_mean_y, color='k', marker='s', alpha=1, linestyle='none')

    err_x = x[inds].std()/np.sqrt(len(inds))
    ax[1].plot([bin_mean_x - err_x, bin_mean_x + err_x], [bin_mean_y, bin_mean_y], linestyle='-', marker='None', color='k', alpha=1, linewidth=2)

    err_y = y[inds].std()/np.sqrt(len(inds))
    ax[1].plot([bin_mean_x, bin_mean_x], [bin_mean_y - err_y, bin_mean_y + err_y], linestyle='-', marker='None', color='k', alpha=1, linewidth=2)

# figS4_1.savefig(os.path.join(analysis_dir, 'figpanels', 'figS4_1.svg'), format='svg', transparent=True, dpi=save_dpi)

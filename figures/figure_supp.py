import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import os
import glob
from scipy.stats import pearsonr
import socket

from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
import matplotlib
from matplotlib import rcParams
rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})

rcParams['svg.fonttype'] = 'none' # let illustrator handle the font type

if socket.gethostname() == 'MHT-laptop':  # windows
    data_dir = r'C:\Users\mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
    analysis_dir = r'C:\Users\mhturner/Dropbox/ClandininLab/Analysis/SC-FC'
elif socket.gethostname() == 'max-laptop':  # linux
    data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
    analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC'

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token=bridge.getNeuprintToken())

# Get FunctionalConnectivity object
FC = functional_connectivity.FunctionalConnectivity(data_dir=data_dir, fs=1.2, cutoff=0.01, mapping=bridge.getRoiMapping())

# Get AnatomicalConnectivity object
AC = anatomical_connectivity.AnatomicalConnectivity(data_dir=data_dir, neuprint_client=neuprint_client, mapping=bridge.getRoiMapping())

plot_colors = plt.get_cmap('tab10')(np.arange(8)/8)


# %% subsampled region cmats and SC-FC corr

atlas_fns = glob.glob(os.path.join(data_dir, 'atlas_data', 'vfb_68_2*'))
sizes = []
for fn in atlas_fns:
    _, roi_size = FC.loadAtlasData(atlas_path=fn)
    sizes.append(roi_size)

sizes = np.vstack(sizes)
roi_size = np.mean(sizes, axis=0)

np.sort(roi_size)

anatomical_adjacency, keep_inds = AC.getAdjacency('CellCount', do_log=True)

bins = np.arange(np.floor(np.min(roi_size)), np.ceil(np.max(roi_size)))
values, base = np.histogram(roi_size, bins=bins, density=True)
cumulative = np.cumsum(values)


# Load precomputed subsampled Cmats for each brain
load_fn = os.path.join(data_dir, 'functional_connectivity', 'subsampled_cmats_20200626.npy')
(cmats_pop, CorrelationMatrix_Full, subsampled_sizes) = np.load(load_fn, allow_pickle=True)

# mean cmat over brains for each subsampledsize and iteration
cmats_popmean = np.mean(cmats_pop, axis=4) # roi x roi x iterations x sizes
scfc_r = np.zeros(shape=(cmats_popmean.shape[2], cmats_popmean.shape[3])) # iterations x sizes
for s_ind, sz in enumerate(subsampled_sizes):
    for it in range(cmats_popmean.shape[2]):
        functional_adjacency_tmp = cmats_popmean[:, :, it, s_ind][FC.upper_inds][keep_inds]
        new_r, _ = pearsonr(anatomical_adjacency, functional_adjacency_tmp)
        scfc_r[it, s_ind] = new_r

# plot mean+/-SEM results on top of region size cumulative histogram
err_y = np.std(scfc_r, axis=0)
mean_y = np.mean(scfc_r, axis=0)

figS3, ax1 = plt.subplots(1, 1, figsize=(4,4))
ax1.plot(subsampled_sizes, mean_y, 'ko')
ax1.errorbar(subsampled_sizes, mean_y, yerr=err_y, color='k')
ax1.hlines(mean_y[-1], subsampled_sizes.min(), subsampled_sizes.max(), color='k', linestyle='--')
ax1.set_xlabel('Region size (voxels)')
ax1.set_ylabel('Correlation with anatomical connectivity')
ax1.set_xscale('log')
ax2 = ax1.twinx()
ax2.plot(bins[:-1], cumulative)
ax2.set_ylabel('Cumulative fraction')
ax2.set_ylim([0, 1.05])
ax2.set_xscale('log')

figS3.savefig(os.path.join(analysis_dir, 'figpanels', 'FigS3.svg'), format='svg', transparent=True)

# %% Dominance analysis
cell_ct, keep_inds = AC.getAdjacency('CellCount', do_log=True)
commoninput, _ = AC.getAdjacency('CommonInputFraction')
completeness = (AC.CompletenessMatrix.to_numpy() + AC.CompletenessMatrix.to_numpy().T) / 2

X = np.vstack([
               cell_ct,
               commoninput[keep_inds],
               FC.SizeMatrix.to_numpy()[FC.upper_inds][keep_inds],
               FC.DistanceMatrix.to_numpy()[FC.upper_inds][keep_inds],
               completeness[AC.upper_inds][keep_inds],
               FC.CorrelationMatrix.to_numpy()[FC.upper_inds][keep_inds]]).T

figS4, ax = plt.subplots(1, 1, figsize=(2, 2.2))
# linear regression model prediction:
regressor = LinearRegression()
regressor.fit(X[:, :-1], X[:, -1]);
pred = regressor.predict(X[:, :-1])
score = regressor.score(X[:, :-1], X[:, -1])
ax.plot(pred, X[:, -1], 'k.')
ax.plot([-0.2, 1.1], [-0.2, 1.1], 'k--')
ax.set_xlabel('Predicted', fontsize=10)
ax.set_ylabel('Measured', fontsize=10)
ax.set_xticks([0, 1.0])
ax.set_yticks([0, 1.0])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# cross-validate linear regression model
rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=0)
cv_results = cross_validate(regressor, X[:, :-1], X[:, -1], cv=rkf, scoring='r2')
avg_r2 = cv_results['test_score'].mean()
err = cv_results['test_score'].std()
print('r2 = {:.2f}+/-{:.2f}'.format(avg_r2, err))
ax.set_title('$r^2$={:.2f}'.format(avg_r2));

r, p = pearsonr(pred, X[:, -1])

fc_df = pd.DataFrame(data=X, columns=['Cell count', 'Common Input', 'ROI size', 'ROI Distance', 'Completeness', 'fc'])
dominance_regression=Dominance(data=fc_df,target='fc',objective=1)

incr_variable_rsquare=dominance_regression.incremental_rsquare()
keys = np.array(list(incr_variable_rsquare.keys()))
vals = np.array(list(incr_variable_rsquare.values()))
s_inds = np.argsort(vals)[::-1]

figS5, ax = plt.subplots(1, 1, figsize=(4.75, 3.5))
sns.barplot(x=[x.replace(' ','\n') for x in keys[s_inds]], y=vals[s_inds], ax=ax, color=plot_colors[0])
ax.set_ylabel('Incremental $r^2$')
ax.tick_params(axis='both', which='major', labelsize=8)


figS4.savefig(os.path.join(analysis_dir, 'figpanels', 'FigS4.svg'), format='svg', transparent=True)
figS5.savefig(os.path.join(analysis_dir, 'figpanels', 'FigS5.svg'), format='svg', transparent=True)

"""
https://nest-simulator.readthedocs.io/en/latest/auto_examples/gif_pop_psc_exp.html

"""

from scipy.signal import butter, lfilter
import numpy as np
import nest
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import explained_variance_score

from scfc import functional_connectivity

analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC'
data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'

dt = 1.0 # msec
dt_rec = 10.0 # msec
time_stop = 10 * 1000  #sec -> msec
gi = 0
ge = 100

# load measured fxnal connectivity
roinames_path = os.path.join(data_dir, 'atlas_data', 'Original_Index_panda_full.csv')
atlas_path = os.path.join(data_dir, 'atlas_data', 'vfb_68_Original.nii.gz')
response_filepaths = glob.glob(os.path.join(data_dir, 'region_responses') + '/' + '*.pkl')
fs = 1.2 # Hz
cutoff = 0.01 # Hz
meas_cmat, _ = functional_connectivity.getFunctionalConnectivity(response_filepaths, cutoff=cutoff, fs=fs)
upper_inds = np.triu_indices(meas_cmat.shape[0], k=1) # k=1 excludes main diagonal


# Load anatomical stuff:
WeakConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'WeakConnections_computed_20200626.pkl'))
MediumConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'MediumConnections_computed_20200626.pkl'))
StrongConnections = pd.read_pickle(os.path.join(data_dir, 'connectome_connectivity', 'StrongConnections_computed_20200626.pkl'))
conn_mat = WeakConnections + MediumConnections + StrongConnections
roi_names = conn_mat.index
# set diag
tmp_mat = conn_mat.to_numpy().copy()
np.fill_diagonal(tmp_mat, 0)
ConnectivityCount = pd.DataFrame(data=tmp_mat, index=conn_mat.index, columns=conn_mat.index)

C = ConnectivityCount.to_numpy() / ConnectivityCount.to_numpy().max()

# %%
nest.ResetKernel()

nest.SetKernelStatus({'resolution': dt})
nodes = C.shape[0]
node_pops = nest.Create("gif_pop_psc_exp", nodes, {"tau_m": 20.0, "len_kernel": -1})

nest_mm = nest.Create('multimeter', params={'record_from': ['mean', 'n_events'], 'interval': dt_rec})
nez = nest.Create('noise_generator', nodes)

# step input
step_amplitude = 10.0
step_on = 2000.0
step_off = step_on + 1000
step_times = list(np.sort(np.hstack([step_on, step_off])))
step_values = list(np.array([+1., 0]) * step_amplitude)

step_input = nest.Create('step_current_generator', nodes)
nest.SetStatus(step_input, {'amplitude_times': step_times, 'amplitude_values': step_values})

input_region = 'MBVL(R)'
input_ind = np.where(input_region == ConnectivityCount.index)[0][0]

for n_ind, nod in enumerate(node_pops):
    nest.SetStatus([nez[n_ind]], {'mean': 0.0, 'std': 0.75, 'dt': 100.0})
    nest.Connect([nez[n_ind]], [nod])
    if n_ind == input_ind:
        nest.Connect([step_input[n_ind]], [nod])

for i, nest_i in enumerate(node_pops):
    for j, nest_j in enumerate(node_pops):
        if i == j:
            nest.Connect([nest_i], [nest_j], conn_spec='all_to_all', syn_spec={'model': 'static_synapse', 'weight': -gi*C[i, j], 'delay': 6.0})
        else:
            nest.Connect([nest_i], [nest_j], conn_spec='all_to_all', syn_spec={'model': 'static_synapse', 'weight': ge*C[i, j], 'delay': 3.0})

nest.Connect(nest_mm, node_pops)

nest.Simulate(time_stop)

# %%
def lpfilter(signal, cut):
    b, a = butter(1, cut, btype='low')
    y = lfilter(b, a, signal)
    return y


cutoff_frequency = 20 # hz
cutoff_period = (1 / cutoff_frequency) * 1000 # sec -> msec
cut_frac = dt_rec / cutoff_period # number of dt_recs in the cutoff sample period

data_mm = nest.GetStatus(nest_mm)[0]['events']
node_responses = np.zeros((len(node_pops), int(time_stop/dt_rec-1)))
for i, nest_i in enumerate(node_pops):
    a_i = data_mm['mean'][data_mm['senders'] == nest_i] / (dt/1000) # -> Spikes/sec

    # rectify and lp filter
    a_i[a_i<0] = 0
    a_i = lpfilter(a_i, cut=cut_frac)

    node_responses[i, :] = a_i

time_vec = np.arange(0, time_stop, dt_rec)[:-1] / 1000

trim_start = int(node_responses.shape[1]/10)
# trim_start = 0
node_responses = node_responses[:, trim_start:]
time_vec = time_vec[trim_start:]

fh, ax = plt.subplots(1, 1, figsize=(14,6))
ax.plot(time_vec, node_responses.T);

# %%


# calc predicted cmat, z transform to compare to measured mean cmat
cmat = np.arctanh(np.corrcoef(node_responses))
np.fill_diagonal(cmat, np.nan)
pred_cmat = pd.DataFrame(data=cmat, index=ConnectivityCount.index, columns=ConnectivityCount.columns)

fh, ax = plt.subplots(1, 2, figsize=(14,6))
sns.heatmap(pred_cmat, ax=ax[0], xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True)
sns.heatmap(meas_cmat, ax=ax[1], xticklabels=True, cbar_kws={'label': 'Functional Correlation (z)','shrink': .8}, cmap="cividis", rasterized=True)

r2 = explained_variance_score(pred_cmat.to_numpy()[upper_inds], meas_cmat.to_numpy()[upper_inds])
r, _ = pearsonr(pred_cmat.to_numpy()[upper_inds], meas_cmat.to_numpy()[upper_inds])
fh, ax = plt.subplots(1, 1, figsize=(6,6))
ax.plot(pred_cmat.to_numpy()[upper_inds], meas_cmat.to_numpy()[upper_inds], 'ko')
ax.plot([-0.2, 1.0], [-0.2, 1.0], 'k--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Measured')
ax.set_title('r = {:.3f}'.format(r))

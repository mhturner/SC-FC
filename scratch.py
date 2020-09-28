
import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import os
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns
import socket
import glob

from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
import matplotlib
from matplotlib import rcParams
rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})
rcParams.update({'axes.spines.right': False})
rcParams.update({'axes.spines.top': False})
rcParams['svg.fonttype'] = 'none' # let illustrator handle the font type
rcParams['pdf.fonttype'] = 42

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
# %%

from scipy.fftpack import fft, fftfreq, fftshift

resp_filepaths = glob.glob(os.path.join(data_dir, 'region_responses', '*.pkl'))
p_spect = []
for resp_fp in resp_filepaths:

    fs = 1.2  # Hz
    cutoff = 0.01


    file_id = resp_fp.split('/')[-1].replace('.pkl', '')
    region_response = pd.read_pickle(resp_fp)

    # To Df/f
    region_response = (region_response.to_numpy() - np.mean(region_response.to_numpy(), axis=1)[:, None]) / np.mean(region_response.to_numpy(), axis=1)[:, None]
    region_response = region_response[:, :2000]


    N = region_response.shape[1]
    T = fs * N

    x = np.linspace(0.0, T, N)

    power = np.abs(fft(region_response, axis=1))**2

    freq = fftfreq(N, 1/fs)

    p_spect.append(power)


p_spect = np.dstack(p_spect)
# %%

avg_pspect = np.mean(p_spect, axis=2)

freq = freq[1:N//2]
avg_pspect = avg_pspect[:, 1:N//2]

fh, ax = plt.subplots(1, 1, figsize=(6,6))
ax.plot(freq, avg_pspect.T, alpha=1.0);
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([1e-2, 5e-1])
ax.set_ylim([1e-1, 1e3])
ax.set_xlabel('Freq. (Hz)')
ax.set_ylabel('Power')
fh.savefig(os.path.join(analysis_dir, 'region_pspect.png'))

# %%
ind = np.where(freq>7e-2)[0][0]
ind2 = np.where(freq<0.1)[0][-1]

hf_sum = np.sum(avg_pspect[:, ind:ind2], axis=1)
hf_sum.shape

fh, ax = plt.subplots(1, 1, figsize=(8,3))
ax.plot(hf_sum, 'kx')
ax.set_xticks(np.arange(0,36))
ax.set_xticklabels(AC.rois, rotation=90);

# %%
from scipy.stats import zscore, pearsonr

# # compute difference matrix using original, asymmetric anatomical connectivity matrix
anatomical_mat = AC.getConnectivityMatrix('CellCount', diag=0).to_numpy().copy()
functional_mat = FC.CorrelationMatrix.to_numpy().copy()
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
DifferenceMatrix = pd.DataFrame(data=diff_m, index=FC.rois, columns=FC.rois)

diff_by_region = DifferenceMatrix.mean()

# %%
fh, ax = plt.subplots(1, 1, figsize=(6,6))

ax.plot(hf_sum, diff_by_region, 'kx')


r, p = pearsonr(hf_sum, diff_by_region)
ax.set_title(r)

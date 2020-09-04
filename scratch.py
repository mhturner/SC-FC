import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import networkx as nx
from scipy.stats import pearsonr
import os
import socket

from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
import matplotlib
from matplotlib import rcParams
rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})
rcParams.update({'axes.spines.right': False})
rcParams.update({'axes.spines.top': False})
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
# %% Distribution of edge weights

anat_connect = AC.getConnectivityMatrix('CellCount', diag=None)
edge_weights = anat_connect.to_numpy().copy().ravel()

bins = np.logspace(0, np.log10(edge_weights.max()), 40)
vals, bins = np.histogram(edge_weights, bins=bins)
bin_ctrs = bins[:-1] + np.diff(bins).mean()/2
fh, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(bin_ctrs, vals, 'k-x')
ax.set_xscale('log')
ax.set_yscale('log')



# %% threshold to get binary adjacency matrix
anat_connect = AC.getConnectivityMatrix('CellCount', diag=0)

thresh_quant = 0.55 # quantile
threshold = np.quantile(anat_connect.to_numpy().ravel(), thresh_quant)
adj_data = anat_connect > threshold
adj_data = adj_data.to_numpy().astype('int')
G_data = nx.from_numpy_matrix(adj_data, create_using=nx.DiGraph)

K = np.sum(adj_data.ravel())
N = adj_data.shape[0]

# Random adjacency
rand_prob = K / (N**2 - N) # exclude diag, which stays 0

iterations = 1000
random_path_lens = []
random_clustering = []
random_degree = []
for it in range(iterations):
    adj_random = np.random.binomial(1, rand_prob, size=(N,N))
    np.fill_diagonal(adj_random, 0)
    G_random = nx.from_numpy_matrix(adj_random, create_using=nx.DiGraph)
    random_path_lens.append(nx.average_shortest_path_length(G_random))
    random_clustering.append(nx.average_clustering(G_random))
    random_degree.append(np.array([val for (node, val) in G_random.degree(weight='weight')]))

random_degree = np.vstack(random_degree)

# %%
fh, ax = plt.subplots(1, 2, figsize=(5, 2.5))
ax[0].set_axis_off()
ax[0].imshow(adj_data, cmap='Greys', vmin=-0.5, vmax=1)
ax[0].set_title('Connectome')
ax[1].set_axis_off()
ax[1].imshow(adj_random, cmap='Greys', vmin=-0.5, vmax=1)
ax[1].set_title('Random')

fh, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(nx.average_shortest_path_length(G_data), nx.average_clustering(G_data), 'bs', label='Connectome')
ax.plot(random_path_lens, random_clustering, 'ko', label='Random', alpha=0.25)
ax.set_xlabel('Path length')
ax.set_ylabel('Clustering')
ax.legend()

# %%
fh, ax = plt.subplots(1, 1, figsize=(3, 2))
data_degree = np.array([val for (node, val) in G_data.degree(weight='weight')])
vals, bins = np.histogram(data_degree, 10, density=True)
ax.plot(bins[:-1], vals, 'b-', label='Connectome')
vals, bins = np.histogram(random_degree.ravel(), 10, density=True)
ax.plot(bins[:-1], vals, 'k-', label='Random')
ax.set_xlabel('Degree')
ax.set_ylabel('Prob.')
ax.set_xlim([0, 55])
# ax.legend()

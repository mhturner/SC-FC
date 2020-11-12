import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import networkx as nx
import os
import socket

from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
import matplotlib
from matplotlib import rcParams
rcParams.update({'font.size': 12})
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
save_dpi = 400

# %% get adjacency matrices for graphs
anat_position = {}
for r in range(len(FC.coms)):
    anat_position[r] = FC.coms[r, :]

adjacency_anat = AC.getConnectivityMatrix('CellCount', symmetrize=True, diag=0).to_numpy()
adjacency_fxn = FC.CorrelationMatrix.to_numpy().copy()
np.fill_diagonal(adjacency_fxn, 0)

# normalize each adjacency
adjacency_anat = adjacency_anat / adjacency_anat.max()
adjacency_fxn = adjacency_fxn / adjacency_fxn.max()
# %% Plot fxn and anat graphs in 3D brain space

take_top_edges = 100
roilabels_to_skip = ['ATL(R)', 'IB', 'MPED(R)', 'SIP(R)', 'PLP(R)', 'SPS(R)', 'GOR(R)', 'GOR(L)', 'ICL(R)','BU(L)', 'BU(R)', 'SCL(R)', 'CRE(R)']

cmap = plt.get_cmap('magma')

cutoff = np.sort(adjacency_anat.ravel())[::-1][take_top_edges]
print('Threshold included {} of {} edges in anatomical connectivity matrix'.format((adjacency_anat>=cutoff).sum(), adjacency_anat.size))
temp_adj_anat = adjacency_anat.copy() # threshold only for display graph
temp_adj_anat[temp_adj_anat<cutoff] = 0
G_anat = nx.from_numpy_matrix(temp_adj_anat/temp_adj_anat.max(), create_using=nx.DiGraph)

cutoff = np.sort(adjacency_fxn.ravel())[::-1][take_top_edges]
print('Threshold included {} of {} edges in functional connectivity matrix'.format((adjacency_fxn>=cutoff).sum(), (adjacency_fxn>0).sum()))
temp_adj_fxn = adjacency_fxn.copy() # threshold only for display graph
temp_adj_fxn[temp_adj_fxn<cutoff] = 0
G_fxn = nx.from_numpy_matrix(temp_adj_fxn/temp_adj_fxn.max(), create_using=nx.DiGraph)

fig3_0 = plt.figure(figsize=(9, 4.5))
ax_anat = fig3_0.add_subplot(1, 2, 1, projection='3d')
ax_fxn = fig3_0.add_subplot(1, 2, 2, projection='3d')

ax_anat.view_init(-70, -95)
ax_anat.set_axis_off()
# ax_anat.set_title('Structural', fontweight='bold', fontsize=12)

ax_fxn.view_init(-70, -95)
ax_fxn.set_axis_off()
# ax_fxn.set_title('Functional', fontweight='bold', fontsize=12)

for key, value in anat_position.items():
    xi = value[0]
    yi = value[1]
    zi = value[2]

    # Plot nodes
    ax_anat.scatter(xi, yi, zi, c='b', s=5+40*G_anat.degree(weight='weight')[key], edgecolors='k', alpha=0.25)
    ax_fxn.scatter(xi, yi, zi, c='b', s=5+20*G_fxn.degree(weight='weight')[key], edgecolors='k', alpha=0.25)
    if FC.rois[key] not in roilabels_to_skip:
        ax_anat.text(xi, yi, zi+2, FC.rois[key], zdir=(0,0,0), fontsize=7, fontweight='bold')
        ax_fxn.text(xi, yi, zi+2, FC.rois[key], zdir=(0,0,0), fontsize=7, fontweight='bold')

# plot connections
for i,j in enumerate(G_anat.edges()):
    x = np.array((anat_position[j[0]][0], anat_position[j[1]][0]))
    y = np.array((anat_position[j[0]][1], anat_position[j[1]][1]))
    z = np.array((anat_position[j[0]][2], anat_position[j[1]][2]))

    # Plot the connecting lines
    line_wt = (G_anat.get_edge_data(j[0], j[1], default={'weight':0})['weight'] + G_anat.get_edge_data(j[1], j[0], default={'weight':0})['weight'])/2
    color = cmap(line_wt)
    ax_anat.plot(x, y, z, c=plot_colors[0], alpha=0.25, linewidth=2)


for i,j in enumerate(G_fxn.edges()):
    x = np.array((anat_position[j[0]][0], anat_position[j[1]][0]))
    y = np.array((anat_position[j[0]][1], anat_position[j[1]][1]))
    z = np.array((anat_position[j[0]][2], anat_position[j[1]][2]))

    # Plot the connecting lines
    line_wt = (G_fxn.get_edge_data(j[0], j[1], default={'weight':0})['weight'] + G_fxn.get_edge_data(j[1], j[0], default={'weight':0})['weight'])/2
    color = cmap(line_wt)
    ax_fxn.plot(x, y, z, c=plot_colors[0], alpha=0.25, linewidth=2)

fig3_0.subplots_adjust(wspace=0.01)
fig3_0.savefig(os.path.join(analysis_dir, 'figpanels', 'fig3_0.svg'), format='svg', transparent=True, dpi=save_dpi)
# %% compare anat + fxnal graph metrics: degree and clustering

roilabels_to_show = ['BU(R)', 'AVLP(R)', 'MBML(R)', 'PVLP(R)', 'AL(R)', 'LH(R)', 'EB', 'PLP(R)', 'AOTU(R)']

# Plot clustering and degree using full adjacency to make graphs
G_anat = nx.from_numpy_matrix(adjacency_anat, create_using=nx.DiGraph)
G_fxn = nx.from_numpy_matrix(adjacency_fxn, create_using=nx.DiGraph)

fig3_1, ax = plt.subplots(1, 2, figsize=(8.0, 3.5))
deg_fxn = np.array([val for (node, val) in G_fxn.degree(weight='weight')])
deg_anat = np.array([val for (node, val) in G_anat.degree(weight='weight')])
plotting.addLinearFit(ax[0], deg_anat, deg_fxn, alpha=0.5)
ax[0].plot(deg_anat, deg_fxn, alpha=1.0, marker='o', linestyle='none')
for r_ind, r in enumerate(FC.rois):
    if r in roilabels_to_show:
        ax[0].annotate(r, (deg_anat[r_ind]+0.4, deg_fxn[r_ind]-0.2), fontsize=8, fontweight='bold')

ax[0].set_xlabel('Structural degree')
ax[0].set_ylabel('Functional degree')
ax[0].set_ylim([0, 37])

clust_fxn = np.real(np.array(list(nx.clustering(G_fxn, weight='weight').values())))
clust_anat = np.array(list(nx.clustering(G_anat, weight='weight').values()))
plotting.addLinearFit(ax[1], clust_anat, clust_fxn, alpha=0.5)
ax[1].plot(clust_anat, clust_fxn, alpha=1.0, marker='o', linestyle='none')
for r_ind, r in enumerate(FC.rois):
    if r in roilabels_to_show:
        ax[1].annotate(r, (clust_anat[r_ind]+0.002, clust_fxn[r_ind]-0.003), fontsize=8, fontweight='bold')
ax[1].set_xlabel('Structural clustering')
ax[1].set_ylabel('Functional clustering')
ax[1].set_ylim([0, 0.445])
ax[1].set_xlim([0, 0.124])

fig3_1.subplots_adjust(wspace=0.5, hspace=0.1)
fig3_1.savefig(os.path.join(analysis_dir, 'figpanels', 'fig3_1.svg'), format='svg', transparent=True, dpi=save_dpi)

# %% Illustration schematics of graph metrics

# Illustration schematic: node degree
G = nx.Graph()
G.add_edge(1, 2, weight=1)
G.add_edge(1, 3, weight=1)
G.add_edge(3, 2, weight=10)

fig3_2, ax = plt.subplots(1, 3, figsize=(3.5, 1))
[x.set_xlim([-1, 1]) for x in ax.ravel()]
[x.set_ylim([-1, 1]) for x in ax.ravel()]
# graph 1: low degree
position = nx.circular_layout(G, scale=0.7)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
deg = [val for (node, val) in G.degree(weight='weight')]
nx.draw(G, ax=ax[0], pos=position, width=np.array(weights)/3, node_color=['r', 'k', 'k'], node_size=75)
nx.draw_networkx_edge_labels(G, ax=ax[0], pos=position, edge_labels=nx.get_edge_attributes(G,'weight'))
ax[0].annotate(deg[0], position[1] + [-0.0, 0.2], fontsize=12, weight='bold')
# graph 2: mod degree
G.add_edge(1, 2, weight=0)
G.add_edge(1, 3, weight=10)
G.add_edge(3, 2, weight=10)
position = nx.circular_layout(G, scale=0.7)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
deg = [val for (node, val) in G.degree(weight='weight')]
nx.draw(G, ax=ax[1], pos=position, width=np.array(weights)/3, node_color=['r', 'k', 'k'], node_size=75)
nx.draw_networkx_edge_labels(G, ax=ax[1], pos=position, edge_labels=nx.get_edge_attributes(G,'weight'))
ax[1].annotate(deg[0], position[1] + [-0.0, 0.2], fontsize=12, weight='bold')

# graph 3: high degree
G.add_edge(1, 2, weight=10)
G.add_edge(1, 3, weight=10)
G.add_edge(3, 2, weight=1)
position = nx.circular_layout(G, scale=0.7)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
deg = [val for (node, val) in G.degree(weight='weight')]
nx.draw(G, ax=ax[2], pos=position, width=np.array(weights)/3, node_color=['r', 'k', 'k'], node_size=75)
nx.draw_networkx_edge_labels(G, ax=ax[2], pos=position, edge_labels=nx.get_edge_attributes(G,'weight'))
ax[2].annotate(deg[0], position[1] + [-0.0, 0.2], fontsize=12, weight='bold');

# Illustration schematic: clustering coefficient
high = 10
low = 1
G = nx.Graph()
G.add_edge(1, 2, weight=high)
G.add_edge(1, 3, weight=high)
G.add_edge(3, 2, weight=0)
G.add_edge(3, 4, weight=0)
G.add_edge(2, 4, weight=0)
G.add_edge(1, 4, weight=high)


fig3_3, ax = plt.subplots(1, 3, figsize=(3.5, 1))
[x.set_xlim([-1, 1]) for x in ax.ravel()]
[x.set_ylim([-1, 1]) for x in ax.ravel()]
# graph 1: low clustering
position = nx.circular_layout(G, scale=0.7)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
clust = list(nx.clustering(G, weight='weight').values())
nx.draw(G, ax=ax[0], pos=position, width=np.array(weights)/3, node_color=['r', 'k', 'k', 'k'], node_size=75)
ax[0].annotate('{:.2f}'.format(clust[0]), position[1] + [-0.0, 0.2], fontsize=12, weight='bold')

# graph 2: mod clustering
G = nx.Graph()
G.add_edge(1, 2, weight=high)
G.add_edge(1, 3, weight=high)
G.add_edge(3, 2, weight=low)
G.add_edge(3, 4, weight=low)
G.add_edge(2, 4, weight=low)
G.add_edge(1, 4, weight=high)
position = nx.circular_layout(G, scale=0.7)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
clust = list(nx.clustering(G, weight='weight').values())
nx.draw(G, ax=ax[1], pos=position, width=np.array(weights)/3, node_color=['r', 'k', 'k', 'k'], node_size=75)
ax[1].annotate('{:.2f}'.format(clust[0]), position[1] + [-0.0, 0.2], fontsize=12, weight='bold')
high
# graph 3: high clustering
G = nx.Graph()
G.add_edge(1, 2, weight=high)
G.add_edge(1, 3, weight=high)
G.add_edge(3, 2, weight=high)
G.add_edge(3, 4, weight=high)
G.add_edge(2, 4, weight=high)
G.add_edge(1, 4, weight=high)
position = nx.circular_layout(G, scale=0.7)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
clust = list(nx.clustering(G, weight='weight').values())
nx.draw(G, ax=ax[2], pos=position, width=np.array(weights)/3, node_color=['r', 'k', 'k', 'k'], node_size=75)
ax[2].annotate('{:.2f}'.format(clust[0]), position[1] + [-0.0, 0.2], fontsize=12, weight='bold');

fig3_2.savefig(os.path.join(analysis_dir, 'figpanels', 'fig3_2.svg'), format='svg', transparent=True, dpi=save_dpi)
fig3_3.savefig(os.path.join(analysis_dir, 'figpanels', 'fig3_3.svg'), format='svg', transparent=True, dpi=save_dpi)

# %% Supp: connectome degree stats: scale free + small world comparisons

# 1) Binarize and compute random adjacency
anat_connect = AC.getConnectivityMatrix('CellCount', diag=0)

thresh_quant = 0.5 # quantile
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

# %% Plot edge weight distribution

fig3_4 = plt.figure(figsize=(9, 4))
ax = fig3_4.add_subplot(1, 2, 2)
anat_connect = AC.getConnectivityMatrix('CellCount', diag=None)
edge_weights = anat_connect.to_numpy().copy().ravel()
bins = np.logspace(0, np.log10(edge_weights.max()), 50)
vals, bins = np.histogram(edge_weights, bins=bins, density=True)

# plot simple power law scaling
xx = np.arange(1, 1000)
a = -1.0
yy =  xx**(a)
ax.plot(xx, yy/np.sum(yy), linestyle='-', linewidth=2, alpha=1.0, color=[0,0,0])
ax.plot(bins[:-1], vals, marker='o', color=plot_colors[0], linestyle='None', alpha=1.0, rasterized=True)
ax.annotate(r'$p(w) \propto w^{{-1}}$', (200, 2e-2))


ax.set_xlim([1, edge_weights.max()])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Edge weight')
ax.set_ylabel('P(weight)')


# plot path length and clustering for random vs. connectome
ax = fig3_4.add_subplot(2, 4, 1)
ax.set_axis_off()
ax.imshow(adj_data, cmap='Greys', vmin=-0.5, vmax=1, interpolation='nearest', rasterized=False)
ax.set_title('Connectome', color=plot_colors[0])
ax = fig3_4.add_subplot(2, 4, 5)
ax.set_axis_off()
ax.imshow(adj_random, cmap='Greys', vmin=-0.5, vmax=1, interpolation='nearest', rasterized=False)
ax.set_title('Random')

ax = fig3_4.add_subplot(2, 4, 2)
ax.hist(random_path_lens, bins=20, density=False, color='k')
measured_path = nx.average_shortest_path_length(G_data)
ax.axvline(measured_path)
ax.set_xlim(1.4, 1.58)
ax.set_xlabel('Path length')
sigma_diff_path = (measured_path-np.mean(random_path_lens)) / np.std(random_path_lens)
factor_diff_path = (measured_path-np.mean(random_path_lens)) / np.mean(random_path_lens)
print('Measured path length is {:.2f} larger than random ({:.2f} sigma)'.format(factor_diff_path, sigma_diff_path))



ax = fig3_4.add_subplot(2, 4, 6)
ax.hist(random_clustering, bins=20, density=False, color='k')
measured_cluster = nx.average_clustering(G_data)
ax.axvline(measured_cluster)
ax.set_xlim(0.45, 0.85)
ax.set_xlabel('clustering')
sigma_diff_cluster = (measured_cluster-np.mean(random_clustering)) / np.std(random_clustering)
factor_diff_cluster = (measured_cluster-np.mean(random_clustering)) / np.mean(random_clustering)
print('Measured clustering is {:.2f} larger than random ({:.2f} sigma)'.format(factor_diff_cluster, sigma_diff_cluster))


fig3_4.subplots_adjust(wspace=0.5, hspace=0.5)
fig3_4.savefig(os.path.join(analysis_dir, 'figpanels', 'fig3_4.svg'), format='svg', transparent=True, dpi=save_dpi)

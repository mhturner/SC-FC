import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import networkx as nx
from scipy.stats import pearsonr, powerlaw, ttest_1samp
import os
import socket
import seaborn as sns

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

take_top_pct = 0.2 # top fraction to include in network graphs
roilabels_to_skip = ['ATL(R)', 'IB', 'MPED(R)', 'SIP(R)', 'PLP(R)', 'SPS(R)', 'GOR(R)', 'GOR(L)', 'ICL(R)','BU(L)', 'BU(R)', 'SCL(R)', 'CRE(R)']

cmap = plt.get_cmap('Blues')

cutoff = np.quantile(adjacency_anat, 1-take_top_pct)
print('Threshold included {} of {} edges in anatomical connectivity matrix'.format((adjacency_anat>=cutoff).sum(), adjacency_anat.size))
temp_adj_anat = adjacency_anat.copy() # threshold only for display graph
temp_adj_anat[temp_adj_anat<cutoff] = 0
G_anat = nx.from_numpy_matrix(temp_adj_anat/temp_adj_anat.max(), create_using=nx.DiGraph)

cutoff = np.quantile(adjacency_fxn[adjacency_fxn>0], 1-take_top_pct)
print('Threshold included {} of {} sig edges in functional connectivity matrix'.format((adjacency_fxn>=cutoff).sum(), (adjacency_fxn>0).sum()))
temp_adj_fxn = adjacency_fxn.copy() # threshold only for display graph
temp_adj_fxn[temp_adj_fxn<cutoff] = 0
G_fxn = nx.from_numpy_matrix(temp_adj_fxn/temp_adj_fxn.max(), create_using=nx.DiGraph)

fig4_0 = plt.figure(figsize=(9, 4.5))
ax_anat = fig4_0.add_subplot(1, 2, 1, projection='3d')
ax_fxn = fig4_0.add_subplot(1, 2, 2, projection='3d')

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
    ax_anat.plot(x, y, z, c=color, alpha=line_wt, linewidth=2)

    line_wt = (G_fxn.get_edge_data(j[0], j[1], default={'weight':0})['weight'] + G_fxn.get_edge_data(j[1], j[0], default={'weight':0})['weight'])/2
    color = cmap(line_wt)
    ax_fxn.plot(x, y, z, c=color, alpha=line_wt, linewidth=2)

fig4_0.savefig(os.path.join(analysis_dir, 'figpanels', 'fig4_0.svg'), format='svg', transparent=True)
# %% compare anat + fxnal graph metrics: degree and clustering

roilabels_to_show = ['BU(R)', 'AVLP(R)', 'MBML(R)', 'PVLP(R)', 'AL(R)', 'LH(R)', 'EB', 'PLP(R)', 'AOTU(R)']

# Plot clustering and degree using full adjacency to make graphs
G_anat = nx.from_numpy_matrix(adjacency_anat, create_using=nx.DiGraph)
G_fxn = nx.from_numpy_matrix(adjacency_fxn, create_using=nx.DiGraph)

fig4_1, ax = plt.subplots(1, 2, figsize=(7, 3.5))
deg_fxn = np.array([val for (node, val) in G_fxn.degree(weight='weight')])
deg_anat = np.array([val for (node, val) in G_anat.degree(weight='weight')])
plotting.addLinearFit(ax[0], deg_anat, deg_fxn, alpha=0.5)
ax[0].plot(deg_anat, deg_fxn, alpha=1.0, marker='o', linestyle='none')
for r_ind, r in enumerate(FC.rois):
    if r in roilabels_to_show:
        ax[0].annotate(r, (deg_anat[r_ind]+0.4, deg_fxn[r_ind]-0.2), fontsize=8, fontweight='bold')

ax[0].set_xlabel('Structural')
ax[0].set_ylabel('Functional')
ax[0].set_ylim([0, 37])

clust_fxn = np.real(np.array(list(nx.clustering(G_fxn, weight='weight').values())))
clust_anat = np.array(list(nx.clustering(G_anat, weight='weight').values()))
plotting.addLinearFit(ax[1], clust_anat, clust_fxn, alpha=0.5)
ax[1].plot(clust_anat, clust_fxn, alpha=1.0, marker='o', linestyle='none')
for r_ind, r in enumerate(FC.rois):
    if r in roilabels_to_show:
        ax[1].annotate(r, (clust_anat[r_ind]+0.002, clust_fxn[r_ind]-0.003), fontsize=8, fontweight='bold')
ax[1].set_xlabel('Structural')
ax[1].set_ylabel('Functional')
ax[1].set_ylim([0, 0.445])
ax[1].set_xlim([0, 0.124])
#
# cent_fxn = np.real(np.array(list(nx.eigenvector_centrality(G_fxn, weight='weight').values())))
# cent_anat = np.array(list(nx.eigenvector_centrality(G_anat, weight='weight').values()))
# plotting.addLinearFit(ax[2], cent_anat, cent_fxn, alpha=0.5)
# ax[2].plot(cent_anat, cent_fxn, alpha=1.0, marker='o', linestyle='none')
# for r_ind, r in enumerate(FC.rois):
#     if r in roilabels_to_show:
#         ax[2].annotate(r, (cent_anat[r_ind]+0.002, cent_fxn[r_ind]-0.003), fontsize=8, fontweight='bold')
# ax[2].set_xlabel('Structural')
# ax[2].set_ylabel('Functional')
# ax[2].set_ylim([0, 0.26])
# ax[2].set_xlim([0, 0.39])

fig4_1.savefig(os.path.join(analysis_dir, 'figpanels', 'fig4_1.svg'), format='svg', transparent=True)

# %% Illustration schematics of graph metrics

# Illustration schematic: node degree
G = nx.Graph()
G.add_edge(1, 2, weight=1)
G.add_edge(1, 3, weight=1)
G.add_edge(3, 2, weight=10)

fig4_2, ax = plt.subplots(1, 3, figsize=(3.5, 1))
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


fig4_3, ax = plt.subplots(1, 3, figsize=(3.5, 1))
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

fig4_2.savefig(os.path.join(analysis_dir, 'figpanels', 'fig4_2.svg'), format='svg', transparent=True)
fig4_3.savefig(os.path.join(analysis_dir, 'figpanels', 'fig4_3.svg'), format='svg', transparent=True)

# %% Shortest path analysis:
anat_connect = AC.getConnectivityMatrix('CellCount', diag=None)
shortest_path_distance, shortest_path_steps, shortest_path_weight, hub_count = bridge.getShortestPathStats(anat_connect)

# for anatomical network: direct cell weight vs connectivity weight of shortest path
direct_dist = (1/AC.getConnectivityMatrix('CellCount', diag=None).to_numpy())
fig4_4, ax = plt.subplots(1, 2, figsize=(7, 3.5))
step_count = shortest_path_steps - 1
steps = np.unique(step_count.to_numpy()[AC.upper_inds])
colors = plt.get_cmap('Dark2')(np.arange(len(steps))/len(steps))
ax[0].plot([2e-4, 1], [2e-4, 1], color=[0.8, 0.8, 0.8], alpha=0.5, linestyle='-')
for s_ind, s in enumerate(steps):
    pull_inds = np.where(step_count == s)
    ax[0].plot(direct_dist[pull_inds], shortest_path_distance.to_numpy()[pull_inds], linestyle='none', marker='.', color=colors[s_ind], label='{:d}'.format(int(s)), alpha=1.0)

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('Direct distance (1/cells)')
ax[0].set_ylabel('Shortest path distance (1/cells)');
ax[0].legend(fontsize='small', fancybox=True);
ax[0].set_ylim([2e-4, 3e-2])


x = np.log10(shortest_path_distance.to_numpy()[AC.upper_inds])  # adjacency matrix gets symmetrized for shortest path algorithms
y = FC.CorrelationMatrix.to_numpy()[AC.upper_inds]
steps = shortest_path_steps.to_numpy()[AC.upper_inds]

fc_pts = []
len_pts = []
for step_no in range(2, 8):
    pull_inds = np.where(steps == step_no)
    fc_pts.append(y[pull_inds])
    len_pts.append(x[pull_inds])

h, p = ttest_1samp(FC.cmats, 0, axis=2)
p_cutoff = 0.05 / p.size # Bonferroni corrected p cutoff
p_vals = p[AC.upper_inds]
p_vals = p_vals<p_cutoff
c = p_vals

r, p = pearsonr(x, y)
coef = np.polyfit(x, y, 1)
linfit = np.poly1d(coef)
xx = np.linspace(x.min(), x.max(), 100)
ax[1].scatter(10**x, y, c=c, alpha=0.5, marker='o', cmap='RdGy')
ax[1].plot(10**xx, linfit(xx), color='k', linestyle='--', linewidth=2, marker=None)
ax[1].set_title('r = {:.2f}'.format(r));
ax[1].set_xlabel('Shortest path distance')
ax[1].set_ylabel('Functional connectivity (z)')
ax[1].set_xscale('log')

num_bins = 10 # equally populated bins
points_per_bin = int(len(x)/num_bins)
for b_ind in range(num_bins):
    inds = np.argsort(x)[(b_ind*points_per_bin):(b_ind+1)*points_per_bin]
    bin_mean_x = x[inds].mean()
    bin_mean_y = y[inds].mean()
    ax[1].plot(10**bin_mean_x, bin_mean_y, 'ks', alpha=1, linestyle='none')

    bin_spread_x = np.quantile(x[inds], (0.05, 0.95))
    ax[1].plot(10**bin_spread_x, [bin_mean_y, bin_mean_y], linestyle='-', marker='None', color='k', alpha=1, linewidth=2)

    bin_spread_y = np.quantile(y[inds], (0.05, 0.95))
    ax[1].plot([10**bin_mean_x, 10**bin_mean_x], bin_spread_y, linestyle='-', marker='None', color='k', alpha=1, linewidth=2)


# fig4_4.savefig(os.path.join(analysis_dir, 'figpanels', 'fig4_4.svg'), format='svg', transparent=True)


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

# %% Plot degree distribution vs power law distribution

figS4_0 = plt.figure(figsize=(8, 4))
ax = figS4_0.add_subplot(1, 2, 2)
anat_connect = AC.getConnectivityMatrix('CellCount', diag=None)
edge_weights = anat_connect.to_numpy().copy().ravel()
bins = np.arange(0, edge_weights.max(), 1)
vals, bins = np.histogram(edge_weights, bins=bins, density=True)

# fit power law
fit_thresh = np.quantile(edge_weights, 0.1) # exclude weakest 10% of connections in fit
p_opt = powerlaw.fit(edge_weights[edge_weights>fit_thresh])
a = p_opt[0]
xx = np.arange(fit_thresh, 1000)
yy = a * xx**(a-1)
ax.plot(xx, yy/yy.sum(), linestyle='-', linewidth=2, alpha=1.0, color=[0,0,0])
ax.annotate('$p(w)=a*w^{{a-1}}$ \na={:.2f}'.format(a), (200, 2e-2))

ax.plot(bins[:-1], vals, marker='o', color=plot_colors[0], linestyle='None', alpha=0.50)

ax.set_xlim([1, edge_weights.max()])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Edge weight')
ax.set_ylabel('P(weight)')

# plot path length and clustering for random vs. connectome
ax = figS4_0.add_subplot(2, 4, 1)
ax.set_axis_off()
ax.imshow(adj_data, cmap='Greys', vmin=-0.5, vmax=1)
ax.set_title('Connectome', color=plot_colors[0])
ax = figS4_0.add_subplot(2, 4, 5)
ax.set_axis_off()
ax.imshow(adj_random, cmap='Greys', vmin=-0.5, vmax=1)
ax.set_title('Random')

ax = figS4_0.add_subplot(2, 4, 2)
ax.hist(random_path_lens, bins=20, density=True, color='k')
ax.axvline(nx.average_shortest_path_length(G_data))
ax.set_xlim(1.4, 1.58)
ax.set_xlabel('Path length')


ax = figS4_0.add_subplot(2, 4, 6)
ax.hist(random_clustering, bins=20, density=True, color='k')
ax.axvline(nx.average_clustering(G_data))
ax.set_xlim(0.45, 0.85)
ax.set_xlabel('clustering')


# figS4_0.savefig(os.path.join(analysis_dir, 'figpanels', 'figS4_0.svg'), format='svg', transparent=True)

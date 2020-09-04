import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
import networkx as nx
from scipy.stats import pearsonr
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
    ax_anat.plot(x, y, z, c=color, alpha=line_wt, linewidth=2)

    line_wt = (G_fxn.get_edge_data(j[0], j[1], default={'weight':0})['weight'] + G_fxn.get_edge_data(j[1], j[0], default={'weight':0})['weight'])/2
    color = cmap(line_wt)
    ax_fxn.plot(x, y, z, c=color, alpha=line_wt, linewidth=2)

fig3_0.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig3_0.svg'), format='svg', transparent=True)
# %% compare anat + fxnal graph metrics: degree and clustering

roilabels_to_show = ['BU(R)', 'AVLP(R)', 'MBML(R)', 'PVLP(R)', 'AL(R)', 'LH(R)', 'EB', 'PLP(R)', 'AOTU(R)']

# Plot clustering and degree using full adjacency to make graphs
G_anat = nx.from_numpy_matrix(adjacency_anat, create_using=nx.DiGraph)
G_fxn = nx.from_numpy_matrix(adjacency_fxn, create_using=nx.DiGraph)

fig3_1, ax = plt.subplots(1, 2, figsize=(7.5, 3.5))
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

fig3_1.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig3_1.svg'), format='svg', transparent=True)
# %% Shortest path analysis:
anat_connect = AC.getConnectivityMatrix('CellCount', diag=None)
shortest_path_distance, shortest_path_steps, shortest_path_weight, hub_count = bridge.getShortestPathStats(anat_connect)

# for anatomical network: direct cell weight vs connectivity weight of shortest path
direct_dist = (1/AC.getConnectivityMatrix('CellCount', diag=None).to_numpy())
fig3_2, ax = plt.subplots(1, 2, figsize=(7, 3.5))
step_count = shortest_path_steps - 1
steps = np.unique(step_count.to_numpy()[AC.upper_inds])
colors = plt.get_cmap('Set1')(np.arange(len(steps))/len(steps))
ax[0].plot([1e-4, 1], [1e-4, 1], color=[0.8, 0.8, 0.8], alpha=0.5, linestyle='-')
for s_ind, s in enumerate(steps):
    pull_inds = np.where(step_count == s)
    ax[0].plot(direct_dist[pull_inds], shortest_path_distance.to_numpy()[pull_inds], linestyle='none', marker='.', color=colors[s_ind], label='{:d}'.format(int(s)), alpha=1.0)

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('Direct distance (1/cells)')
ax[0].set_ylabel('Shortest path distance (1/cells)');
ax[0].legend(fontsize='small', fancybox=True);


x = np.log10(shortest_path_distance.to_numpy()[AC.upper_inds])  # adjacency matrix gets symmetrized for shortest path algorithms
y = FC.CorrelationMatrix.to_numpy()[AC.upper_inds]
steps = shortest_path_steps.to_numpy()[AC.upper_inds]

fc_pts = []
len_pts = []
for step_no in range(2, 8):
    pull_inds = np.where(steps == step_no)
    fc_pts.append(y[pull_inds])
    len_pts.append(x[pull_inds])


r, p = pearsonr(x, y)
coef = np.polyfit(x, y, 1)
linfit = np.poly1d(coef)
xx = np.linspace(x.min(), x.max(), 100)
ax[1].plot(10**x, y, 'ko', alpha=0.5, marker='o', color=[0.5, 0.5, 0.5])
ax[1].plot(10**xx, linfit(xx), color='k', linestyle='--', linewidth=2, marker=None)
ax[1].set_title('r = {:.2f}'.format(r));
ax[1].set_xlabel('Shortest path distance')
ax[1].set_ylabel('Functional connectivity (z)')
ax[1].set_xscale('log')

bins = np.linspace(x.min(), x.max(), 6)
for b_ind in range(len(bins)-1):
    inds = np.logical_and(x>bins[b_ind], x<bins[b_ind+1])
    bin_mean_x = x[inds].mean()
    bin_mean = y[inds].mean()
    bin_spread = np.quantile(y[inds], (0.05, 0.95))
    ax[1].plot(10**bin_mean_x, bin_mean, 'ks', alpha=1, linestyle='none')
    ax[1].plot([10**bin_mean_x, 10**bin_mean_x], bin_spread, 'k-', alpha=1, linewidth=2)

fig3_2.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig3_2.svg'), format='svg', transparent=True)

# %% shuffle analysis
import pandas as pd
from scipy.stats import norm

anat_connect = AC.getConnectivityMatrix('CellCount', diag=0)
sp, sp_steps, _, hub_count = bridge.getShortestPathStats(anat_connect)

iterations = 20
sp_logshuff = []
sp_allshuff = []
sp_rowshuff = []
sp_colshuff = []
for it in range(iterations):
    logshuff_new, _, _, _ = bridge.getShortestPathStats(bridge.getLognormShuffle(anat_connect))
    sp_logshuff.append(logshuff_new)

    allshuff_new, _, _, _ = bridge.getShortestPathStats(bridge.getAllShuffle(anat_connect))
    sp_allshuff.append(allshuff_new)

    rowshuff_new, _, _, _ = bridge.getShortestPathStats(bridge.getRowShuffle(anat_connect))
    sp_rowshuff.append(rowshuff_new)

    colshuff_new, _, _, _ = bridge.getShortestPathStats(bridge.getColShuffle(anat_connect))
    sp_colshuff.append(colshuff_new)

sp_logshuff = np.dstack(sp_logshuff) # rois x rois x iterations
sp_allshuff = np.dstack(sp_allshuff) # rois x rois x iterations
sp_rowshuff = np.dstack(sp_rowshuff) # rois x rois x iterations
sp_colshuff = np.dstack(sp_colshuff) # rois x rois x iterations

# %%
from scipy.stats import ttest_rel


data_shortest_paths = sp.to_numpy()[AC.upper_inds]
fh, ax = plt.subplots(1, 4, figsize=(12, 3))
[x.plot([4e-4, 3e-2], [4e-4, 3e-2], 'k--') for x in ax]
[x.set_xscale('log') for x in ax]
[x.set_yscale('log') for x in ax]
# LOG
shuff = np.mean(sp_logshuff, axis=2)[AC.upper_inds]
ax[0].plot(data_shortest_paths, shuff, 'ko', alpha=0.25)
ax[0].set_title('lognorm')
ax[0].plot(np.mean(data_shortest_paths), np.mean(shuff), 'rs')

# ALL
shuff = np.mean(sp_allshuff, axis=2)[AC.upper_inds]
ax[1].plot(data_shortest_paths, shuff, 'ko', alpha=0.25)
ax[1].set_title('all')
ax[1].plot(np.mean(data_shortest_paths), np.mean(shuff), 'rs')

# ROW
shuff = np.mean(sp_rowshuff, axis=2)[AC.upper_inds]
ax[2].plot(data_shortest_paths, shuff, 'ko', alpha=0.25)
ax[2].set_title('rows')
ax[2].plot(np.mean(data_shortest_paths), np.mean(shuff), 'rs')

# COL
shuff = np.mean(sp_colshuff, axis=2)[AC.upper_inds]
ax[3].plot(data_shortest_paths, shuff, 'ko', alpha=0.25)
ax[3].set_title('columns')
ax[3].plot(np.mean(data_shortest_paths), np.mean(shuff), 'rs')

ax[1].set_xlabel('Data')
ax[0].set_ylabel('Shuffled')

# %%
anat_connect = AC.getConnectivityMatrix('CellCount', diag=0)
sp, sp_steps, _, hub_count = bridge.getShortestPathStats(anat_connect)


rand_connect = bridge.getRandomAdjacency(anat_connect, model='norm')
sp_rand, sp_steps_rand, _, hub_count_rand = bridge.getShortestPathStats(rand_connect)

# %%
anat_connect = AC.getConnectivityMatrix('CellCount', diag=None)
edge_weights = anat_connect.to_numpy().copy().ravel()

bins = np.logspace(0, np.log10(edge_weights.max()), 20)
vals, bins = np.histogram(edge_weights, bins=bins)
bin_ctrs = bins[:-1] + np.diff(bins).mean()/2
fh, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(bin_ctrs, vals, 'k-x')
ax.set_xscale('log')
ax.set_yscale('log')

# %%
fh, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot([4e-5, 3e-2], [4e-5, 3e-2], 'k--')
ax.set_xscale('log')
ax.set_yscale('log')
data_shortest_paths = sp.to_numpy()[AC.upper_inds]
rand_shortest_paths = sp_rand.to_numpy()[AC.upper_inds]

ax.plot(data_shortest_paths, rand_shortest_paths, 'ko', alpha=0.25)
ax.set_title('shortest path distance')
ax.plot(np.mean(data_shortest_paths), np.mean(rand_shortest_paths), 'rs')
ax.set_xlabel('Data')
ax.set_ylabel('Shuffled')
# %%
nx.average_clustering(G_data, weight='weight')

G_data = nx.from_numpy_matrix(anat_connect.to_numpy(), create_using=nx.Graph)

G_shuff = nx.from_numpy_matrix(rand_connect.to_numpy(), create_using=nx.DiGraph)

# %%
anat_connect = AC.getConnectivityMatrix('CellCount', diag=0)
G_data = nx.from_numpy_matrix(anat_connect.to_numpy(), create_using=nx.DiGraph)

C = nx.average_clustering(G_data, weight='weight')
sp, _, _, _ = bridge.getShortestPathStats(anat_connect)
L = np.mean(sp.to_numpy().ravel())

sp_rand, _, _, _ = bridge.getShortestPathStats(bridge.getRandomAdjacency(anat_connect))
Lr = np.mean(sp_rand.to_numpy().ravel())

omega = Lr/L - C/Cl



G_lattice = nx.grid_2d_graph(6, 6)



# %%
direct_dist = (1/new_shuff.to_numpy())
fig3_2, ax = plt.subplots(1, 2, figsize=(7, 3.5))
step_count = shortest_path_steps - 1
steps = np.unique(step_count.to_numpy()[AC.upper_inds])
colors = plt.get_cmap('Set1')(np.arange(len(steps))/len(steps))
ax[0].plot([1e-4, 1], [1e-4, 1], color=[0.8, 0.8, 0.8], alpha=0.5, linestyle='-')
for s_ind, s in enumerate(steps):
    pull_inds = np.where(step_count == s)
    ax[0].plot(direct_dist[pull_inds], shortest_path_distance.to_numpy()[pull_inds], linestyle='none', marker='.', color=colors[s_ind], label='{:d}'.format(int(s)), alpha=1.0)

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('Direct distance (1/cells)')
ax[0].set_ylabel('Shortest path distance (1/cells)');
ax[0].legend(fontsize='small', fancybox=True);

# %%
x = np.log10(shortest_path_distance.to_numpy()[AC.upper_inds])  # adjacency matrix gets symmetrized for shortest path algorithms
y = FC.CorrelationMatrix.to_numpy()[AC.upper_inds]
steps = shortest_path_steps.to_numpy()[AC.upper_inds]

fc_pts = []
len_pts = []
for step_no in range(2, 8):
    pull_inds = np.where(steps == step_no)
    fc_pts.append(y[pull_inds])
    len_pts.append(x[pull_inds])


r, p = pearsonr(x, y)
coef = np.polyfit(x, y, 1)
linfit = np.poly1d(coef)
xx = np.linspace(x.min(), x.max(), 100)
ax[1].plot(10**xx, linfit(xx), color='k', linewidth=2, marker=None)

ax[1].plot(10**x, y, 'ko', alpha=0.25)
ax[1].set_title('r = {:.2f}'.format(r));
ax[1].set_xlabel('Shortest path distance')
ax[1].set_ylabel('Functional connectivity (z)')
ax[1].set_xscale('log')
# fig3_2.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig3_2.svg'), format='svg', transparent=True)

# %%

tmp = anat_connect.to_numpy().copy()


# shuffle rows
rows_shuffled = np.zeros_like
shuffle_inds = np.random.permutation(np.arange(36))


# shuffle columns
shuffle_inds = np.random.permutation(np.arange(36))

shuffle_inds
rows_to_shuffle = np.array([x for x in range(36) if x != ind])
shuffle_inds = rows_to_shuffle.copy()
np.random.shuffle(shuffle_inds)
tmp[rows_to_shuffle,:] = tmp[shuffle_inds,:]

# ToDo: make tools for diff shuffle types.
#   * keep diag
#   1) shuff all
#   2) shuff rows
#   3) shuff columns
#   4) Use fake lognorm distributions to make fake adj matrices. a) mean+std over all regions, or mean+std diff for each region

shuffled_connectivity = np.zeros_like(anat_connect)
for ind in range(36):
    tmp = anat_connect.to_numpy().copy()
    rows_to_shuffle = np.array([x for x in range(36) if x != ind])
    shuffle_inds = rows_to_shuffle.copy()
    np.random.shuffle(shuffle_inds)
    tmp[rows_to_shuffle,:] = tmp[shuffle_inds,:]

    sc = pd.DataFrame(data=tmp, index=anat_connect.index, columns=anat_connect.columns)

    sp, _, _, _ = bridge.getShortestPathStats(sc)
    shuffled_connectivity[ind,:] = sp.iloc[ind, :]

shuff_connect = pd.DataFrame(data=shuffled_connectivity, index=anat_connect.index, columns=anat_connect.columns)

direct_dist = (1/shuff_connect.to_numpy())
fig3_2, ax = plt.subplots(1, 2, figsize=(7, 3.5))
step_count = shortest_path_steps_shuff - 1
steps = np.unique(step_count.to_numpy()[AC.upper_inds])
colors = plt.get_cmap('Set1')(np.arange(len(steps))/len(steps))
ax[0].plot([1e-4, 1], [1e-4, 1], color=[0.8, 0.8, 0.8], alpha=0.5, linestyle='-')
for s_ind, s in enumerate(steps):
    pull_inds = np.where(step_count == s)
    ax[0].plot(direct_dist[pull_inds], shortest_path_distance_shuff.to_numpy()[pull_inds], linestyle='none', marker='.', color=colors[s_ind], label='{:d}'.format(int(s)), alpha=1.0)

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('Direct distance (1/cells)')
ax[0].set_ylabel('Shortest path distance (1/cells)');
ax[0].legend(fontsize='small', fancybox=True);

x = np.log10(shortest_path_distance_shuff.to_numpy()[AC.upper_inds])  # adjacency matrix gets symmetrized for shortest path algorithms
y = FC.CorrelationMatrix.to_numpy()[AC.upper_inds]
steps = shortest_path_steps_shuff.to_numpy()[AC.upper_inds]

fc_pts = []
len_pts = []
for step_no in range(2, 8):
    pull_inds = np.where(steps == step_no)
    fc_pts.append(y[pull_inds])
    len_pts.append(x[pull_inds])


r, p = pearsonr(x, y)
coef = np.polyfit(x, y, 1)
linfit = np.poly1d(coef)
xx = np.linspace(x.min(), x.max(), 100)
ax[1].plot(10**xx, linfit(xx), color='k', linewidth=2, marker=None)

ax[1].plot(10**x, y, 'ko', alpha=0.25)
ax[1].set_title('r = {:.2f}'.format(r));
ax[1].set_xlabel('Shortest path distance')
ax[1].set_ylabel('Functional connectivity (z)')
ax[1].set_xscale('log')

# %% Illustration schematics of graph metrics

# Illustration schematic: node degree
G = nx.Graph()
G.add_edge(1, 2, weight=1)
G.add_edge(1, 3, weight=1)
G.add_edge(3, 2, weight=10)

fig3_3, ax = plt.subplots(1, 3, figsize=(3.5, 1))
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


fig3_4, ax = plt.subplots(1, 3, figsize=(3.5, 1))
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

fig3_3.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig3_3.svg'), format='svg', transparent=True)
fig3_4.savefig(os.path.join(analysis_dir, 'figpanels', 'Fig3_4.svg'), format='svg', transparent=True)

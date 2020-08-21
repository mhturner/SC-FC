from neuprint import Client

from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
import networkx as nx
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from node2vec import Node2Vec
from gensim.models import Word2Vec

data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'


neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1heHdlbGxob2x0ZXR1cm5lckBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpMHJRX0M4akliX0ZrS2h2OU5DSElsWlpnRDY5YUMtVGdNLWVWM3lRP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzY2MTk1MzcwfQ.Q-57D4tX2sXMjWym2LFhHaUGHgHiUsIM_JI9xekxw_0')


dim = 2

FC = functional_connectivity.FunctionalConnectivity(data_dir=data_dir, fs=1.2, cutoff=0.01, mapping=bridge.getRoiMapping())
AC = anatomical_connectivity.AnatomicalConnectivity(data_dir=data_dir, neuprint_client=neuprint_client, mapping=bridge.getRoiMapping())

anat_connect = AC.getConnectivityMatrix('WeightedSynapseCount', diag=None).to_numpy()


G_anat = nx.from_numpy_matrix(anat_connect)

n2v = Node2Vec(graph=G_anat,
                dimensions=dim,
                walk_length=16,
                num_walks=1000,
                workers=1)
node2vec_model = n2v.fit()
embedding = np.vstack([np.array(node2vec_model[str(u)]) for u in sorted(G_anat.nodes)])

pred_conn = np.zeros((len(FC.rois), len(FC.rois), dim))
for r_ind, row in enumerate(FC.rois):
    for c_ind, col in enumerate(FC.rois):
        pred_conn[r_ind, c_ind, :] = embedding[r_ind, :] * embedding[c_ind, :]



# %%

if dim == 2:
    colors = FC.CorrelationMatrix.mean().to_numpy()
    fh, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap='cividis', marker='o', alpha=0.4)
    # for n_ind, n in enumerate(AC.rois):
    #     ax.text(embedding[n_ind, 0], embedding[n_ind, 1], n, rotation=0)

    colors = FC.CorrelationMatrix.to_numpy()[FC.upper_inds]
    pts = pred_conn[FC.upper_inds]
    fh2, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.scatter(pts[:, 0], pts[:, 1], c=colors, cmap='cividis', marker='o', alpha=0.4)

elif dim == 3:
    colors = FC.CorrelationMatrix.mean().to_numpy()
    fh = plt.figure(figsize=(6,6))
    ax = Axes3D(fh)
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[n_ind, 2], c=colors, cmap='cividis')
    for n_ind, n in enumerate(AC.rois):
        ax.text(embedding[n_ind, 0], embedding[n_ind, 1], embedding[n_ind, 2], n, zdir=(0,0,0))

    colors = FC.CorrelationMatrix.to_numpy()[FC.upper_inds]
    pts = pred_conn[FC.upper_inds]
    fh2 = plt.figure(figsize=(6,6))
    ax = Axes3D(fh2)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, cmap='cividis', marker='o')



# %%
import tensorflow as tf
from tensorflow import keras

x = AC.getConnectivityMatrix('ConnectivityWeight', diag=None).to_numpy()
y = FC.CorrelationMatrix.to_numpy()

model = tf.keras.Sequential(layers=None, name=None)
model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Dense(4))
model.compile(optimizer='sgd', loss='mse')
# This builds the model for the first time:
model.fit(x, y, batch_size=36, epochs=10)

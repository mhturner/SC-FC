
from neuprint import Client
import numpy as np
import pandas as pd
import os
from region_connectivity import RegionConnectivity
import datetime
import time
import socket
import pickle

t0 = time.time()

if socket.gethostname() == 'max-laptop':
    analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/hemibrain_analysis/roi_connectivity'
elif 'sh' in socket.gethostname():
    analysis_dir = '/oak/stanford/groups/trc/data/Max/Analysis/Hemibrain'

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.0.1', token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1heHdlbGxob2x0ZXR1cm5lckBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpMHJRX0M4akliX0ZrS2h2OU5DSElsWlpnRDY5YUMtVGdNLWVWM3lRP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzY2MTk1MzcwfQ.Q-57D4tX2sXMjWym2LFhHaUGHgHiUsIM_JI9xekxw_0')

# get rois of interest
mapping = RegionConnectivity.getRoiMapping(neuprint_client)

neuron_count_threshold=np.array([1, 5, 10, 20, 40, 80, 100, 200, 400, 800])
ConnectivityMatrix, NeuronCount, SynapseCount = RegionConnectivity.computeConnectivityMatrix(neuprint_client, mapping, neuron_count_threshold=neuron_count_threshold)

print('Finished computing connectivity matrix (total time = {:.1f} sec)'.format(time.time()-t0))

# %%
d = datetime.datetime.today()
datestring ='{:02d}'.format(d.year)+'{:02d}'.format(d.month)+'{:02d}'.format(d.day)

ConnectivityMatrix.to_pickle(os.path.join(analysis_dir, 'ConnectivityMatrix_computed_{}.pkl'.format(datestring)))
SynapseCount.to_pickle(os.path.join(analysis_dir, 'SynapseCount_computed_{}.pkl'.format(datestring)))

np.save(os.path.join(analysis_dir, 'neuron_count_threshold_{}.npy'.format(datestring)), neuron_count_threshold, allow_pickle=True)

with open(os.path.join(analysis_dir, 'NeuronCount_computed_{}.pkl'.format(datestring)), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(NeuronCount, f, pickle.HIGHEST_PROTOCOL)

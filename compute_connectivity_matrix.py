
from neuprint import (Client, fetch_neurons, NeuronCriteria)
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
mapping = RegionConnectivity.getRoiMapping()

WeakConnections, MediumConnections, StrongConnections, Connectivity = RegionConnectivity.computeConnectivityMatrix(neuprint_client, mapping)

print('Finished computing connectivity matrix (total time = {:.1f} sec)'.format(time.time()-t0))

# %%
d = datetime.datetime.today()
datestring ='{:02d}'.format(d.year)+'{:02d}'.format(d.month)+'{:02d}'.format(d.day)


WeakConnections.to_pickle(os.path.join(analysis_dir, 'WeakConnections_computed_{}.pkl'.format(datestring)))
MediumConnections.to_pickle(os.path.join(analysis_dir, 'MediumConnections_computed_{}.pkl'.format(datestring)))
StrongConnections.to_pickle(os.path.join(analysis_dir, 'StrongConnections_computed_{}.pkl'.format(datestring)))
Connectivity.to_pickle(os.path.join(analysis_dir, 'Connectivity_computed_{}.pkl'.format(datestring)))

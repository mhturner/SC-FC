
from neuprint import (Client)
import os
from scfc import anatomical_connectivity, bridge
import datetime
import time
import socket

t0 = time.time()

if socket.gethostname() == 'max-laptop':
    data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/hemibrain_analysis/roi_connectivity'
elif 'sh' in socket.gethostname():
    data_dir = '/oak/stanford/groups/trc/data/Max/flynet/data/connectome_connectivity'

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1heHdlbGxob2x0ZXR1cm5lckBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpMHJRX0M4akliX0ZrS2h2OU5DSElsWlpnRDY5YUMtVGdNLWVWM3lRP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzY2MTk1MzcwfQ.Q-57D4tX2sXMjWym2LFhHaUGHgHiUsIM_JI9xekxw_0')

# get rois of interest
mapping = bridge.getRoiMapping()

SynapseCountDistribution = anatomical_connectivity.computeSynapseConnectivityDistribution(neuprint_client, mapping)

print('Finished computing connectivity distribution (total time = {:.1f} sec)'.format(time.time()-t0))

# %%
d = datetime.datetime.today()
datestring ='{:02d}'.format(d.year)+'{:02d}'.format(d.month)+'{:02d}'.format(d.day)


SynapseCountDistribution.to_pickle(os.path.join(data_dir, 'SynapseCountDistribution_{}.pkl'.format(datestring)))

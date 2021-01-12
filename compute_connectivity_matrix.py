"""Compute connectivity matrices from hemibrain data."""
from neuprint import (Client)
import os
from scfc import anatomical_connectivity, bridge
import datetime
import time
import numpy as np

t0 = time.time()

data_dir = bridge.getUserConfiguration()['data_dir']
analysis_dir = bridge.getUserConfiguration()['analysis_dir']
token = bridge.getUserConfiguration()['token']

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token=token)

# get rois of interest
mapping = bridge.getRoiMapping()

WeakConnections, MediumConnections, StrongConnections, Connectivity, WeightedSynapseNumber, TBars, body_ids = anatomical_connectivity.computeConnectivityMatrix(neuprint_client, mapping)

print('Finished computing connectivity matrix (total time = {:.1f} sec)'.format(time.time()-t0))

# %%
d = datetime.datetime.today()
datestring ='{:02d}'.format(d.year)+'{:02d}'.format(d.month)+'{:02d}'.format(d.day)

WeakConnections.to_pickle(os.path.join(data_dir, 'WeakConnections_computed_{}.pkl'.format(datestring)))
MediumConnections.to_pickle(os.path.join(data_dir, 'MediumConnections_computed_{}.pkl'.format(datestring)))
StrongConnections.to_pickle(os.path.join(data_dir, 'StrongConnections_computed_{}.pkl'.format(datestring)))
Connectivity.to_pickle(os.path.join(data_dir, 'Connectivity_computed_{}.pkl'.format(datestring)))
WeightedSynapseNumber.to_pickle(os.path.join(data_dir, 'WeightedSynapseNumber_computed_{}.pkl'.format(datestring)))
TBars.to_pickle(os.path.join(data_dir, 'TBars_computed_{}.pkl'.format(datestring)))
np.save(os.path.join(data_dir, 'body_ids_{}.npy'.format(datestring)), body_ids)

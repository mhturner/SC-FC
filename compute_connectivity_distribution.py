"""Compute connectivity distributions from hemibrain data."""
from neuprint import (Client)
import os
from scfc import anatomical_connectivity, bridge
import datetime
import time

t0 = time.time()

data_dir = bridge.getUserConfiguration()['data_dir']
token = bridge.getUserConfiguration()['token']

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token=token)

# get rois of interest
mapping = bridge.getRoiMapping()

SynapseCountDistribution = anatomical_connectivity.computeSynapseConnectivityDistribution(neuprint_client, mapping)

print('Finished computing connectivity distribution (total time = {:.1f} sec)'.format(time.time()-t0))

# %%
d = datetime.datetime.today()
datestring ='{:02d}'.format(d.year)+'{:02d}'.format(d.month)+'{:02d}'.format(d.day)


SynapseCountDistribution.to_pickle(os.path.join(data_dir, 'SynapseCountDistribution_{}.pkl'.format(datestring)))

"""."""

import matplotlib.pyplot as plt
from neuprint import Client, fetch_neurons, fetch_synapses, fetch_custom, NeuronCriteria, SynapseCriteria
import numpy as np
import os
import pandas as pd
import seaborn as sns
import glob
import nibabel as nib
from scfc import bridge, anatomical_connectivity, functional_connectivity, plotting
from scfc import bridge
import time

data_dir = bridge.getUserConfiguration()['data_dir']
analysis_dir = bridge.getUserConfiguration()['analysis_dir']
token = bridge.getUserConfiguration()['token']

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token=token)

# Get FunctionalConnectivity object
FC = functional_connectivity.FunctionalConnectivity(data_dir=data_dir, fs=1.2, cutoff=0.01, mapping=bridge.getRoiMapping())

# %%

sour = 'AL(R)'
targ = 'LH(R)'


Neur, Syn = fetch_neurons(NeuronCriteria(inputRois=sour, outputRois=targ, status='Traced'))
body_ids = Neur.bodyId

# %% cell by cell
t0 = time.time()

mask_brain = np.asarray(np.squeeze(nib.load(FC.atlas_path).get_fdata()), 'uint8')
res = 3 # um/voxel of atlas

regions = np.unique(mask_brain)[1:] # cut out first region (0), which is empty

count_matrix = pd.DataFrame(data=np.zeros((len(regions), len(regions))), index=regions, columns=regions)

for body in body_ids:
    Q = """\
        MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)
        WHERE n.status = "Traced"
        AND n.bodyId = {}
        RETURN DISTINCT [s.location.x * 8/1000, s.location.y * 8/1000, s.location.z * 8/1000] AS LOC, s.type

        """.format(body)
    df = fetch_custom(Q, format='pandas', client=neuprint_client)
    inds = np.round(np.vstack(df['LOC']) / res).astype('int') # convert from microns to atlas voxels. Indices in mask where synapses are located
    df['region'] = mask_brain[inds[:, 0], inds[:, 1], inds[:, 2]] # region number for each row (synapse) associated with this cell

    target_regions = np.unique(df[df['s.type'] == 'pre'].region)
    source_regions = np.unique(df[df['s.type'] == 'post'].region)

    count_matrix.loc[source_regions[source_regions>0], target_regions[target_regions>0]] += 1

print('Done ({:.2f} sec)'.format(time.time() - t0))

# %%
regions
len(regions)
np.max(inds, axis=0)
mask_brain.shape

318/637
291/627
plt.imshow(mask_brain.sum(axis=2))
# %%
source_regions
target_regions
fh, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.imshow(count_matrix)

# %%

x = list(range(7000, 8000))
y = list(range(22000, 23000))
z = list(range(23000, 24000))



xyz.shape
xyz
# x_min = 7000
# x_max = 7200
#
# y_min = 22000
# y_max = 22200
#
# z_min = 23000
# z_max = 23200

xyz
q_target

q_target = """\
    WITH [[7000, 22000, 23000], [7000, 22000, 23001]] AS LOCATIONS
    MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)
    WHERE n.status = "Traced"
    AND n.bodyId = {}
    AND s.type = "pre"
    AND [s.location.x, s.location.y, s.location.z] IN LOCATIONS

    RETURN DISTINCT n.type, n.bodyId, s.location.x, s.location.y, s.location.z, s.type

    """.format(body)
df_target = fetch_custom(q_target, format='pandas', client=neuprint_client)
df_target

# %%
q_target
q_target = """\
    WITH {} AS LOCATIONS
    MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)
    WHERE n.status = "Traced"
    AND n.bodyId = {}
    AND s.type = "pre"
    AND [s.location.x, s.location.y, s.location.z] IN LOCATIONS

    RETURN DISTINCT n.type, n.bodyId, s.location.x, s.location.y, s.location.z, s.type

    """.format(np.array2string(xyz, separator=',').replace('\n', ''), body)
df_target = fetch_custom(q_target, format='pandas', client=neuprint_client)
df_target
# %%

x_min = 7000
x_max = 7200

y_min = 22000
y_max = 22200

z_min = 23000
z_max = 23200

q_source = """\
    MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)
    WHERE n.type = "LC11"
    AND n.status = "Traced"
    AND s.type = "post"
    AND s.location.x >= {}
    AND s.location.x < {}
    AND s.location.y >= {}
    AND s.location.y < {}
    AND s.location.z >= {}
    AND s.location.z < {}
    RETURN DISTINCT n.type, n.bodyId, n.status, s.location.x, s.location.y, s.location.z, s.type

    """.format(x_min, x_max, y_min, y_max, z_min, z_max)

df_source = fetch_custom(q_source, format='pandas', client=neuprint_client)
df_source

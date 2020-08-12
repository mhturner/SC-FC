
# %%


import os
from neuprint import (Client, fetch_neurons, fetch_skeleton, fetch_adjacencies, NeuronCriteria)
import numpy as np
from scfc import anatomical_connectivity

neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1heHdlbGxob2x0ZXR1cm5lckBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpMHJRX0M4akliX0ZrS2h2OU5DSElsWlpnRDY5YUMtVGdNLWVWM3lRP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzY2MTk1MzcwfQ.Q-57D4tX2sXMjWym2LFhHaUGHgHiUsIM_JI9xekxw_0')

sour = 'AL(R)'
targ = 'LH(R)'

def getPrimaryInput(roiInfo):
    inputs = dict.fromkeys(roiInfo.keys(), 0)
    for key in inputs:
        inputs[key] = roiInfo[key].get('post', 0)

    primary_input = max(inputs, key=inputs.get)
    return primary_input

Neur_a, _ = fetch_neurons(NeuronCriteria(outputRois=sour, status='Traced')) # row
Neur_b, _ = fetch_neurons(NeuronCriteria(outputRois=targ, status='Traced'))

Neur_a['primary_input'] = [getPrimaryInput(x) for x in Neur_a.roiInfo]
Neur_b['primary_input'] = [getPrimaryInput(x) for x in Neur_b.roiInfo]

a_from_elsewhere = Neur_a[~Neur_a['primary_input'].isin([sour, targ])]
b_from_elsewhere = Neur_b[~Neur_b['primary_input'].isin([sour, targ])]

shared = np.intersect1d(a_from_elsewhere.bodyId, b_from_elsewhere.bodyId)
# %%

data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
bodyId = 635062078
with open(os.path.join(data_dir, '{}.swc'.format(bodyId)), 'w') as file:
    file.write(neuprint_client.fetch_skeleton(bodyId, format='swc'))
# %%
for key in mapping:
    for roi in mapping[key]:
        neuprint_client.fetch_roi_mesh(roi, export_path=os.path.join(data_dir, 'neuprint_meshes', '{}.obj'.format(roi)))


# %%

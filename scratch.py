
import os
from neuprint import (Client, fetch_neurons, fetch_skeleton, fetch_adjacencies, NeuronCriteria)
from  neuprint.skeleton import heal_skeleton, skeleton_df_to_swc
import numpy as np
from scfc import anatomical_connectivity

neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1heHdlbGxob2x0ZXR1cm5lckBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpMHJRX0M4akliX0ZrS2h2OU5DSElsWlpnRDY5YUMtVGdNLWVWM3lRP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzY2MTk1MzcwfQ.Q-57D4tX2sXMjWym2LFhHaUGHgHiUsIM_JI9xekxw_0')

# %%

data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
bodyId = 635062078
skel = neuprint_client.fetch_skeleton(bodyId, format='pandas')

skel_swc = neuprint_client.fetch_skeleton(bodyId, format='swc')

# %%
skel
healed = heal_skeleton(skel)
healed_swc = skeleton_df_to_swc(healed)
# %%
from bokeh.resources import INLINE
import bokeh
import bokeh.io
bokeh.io.output_notebook(INLINE)
from bokeh.io import export_png
from bokeh.plotting import figure

p = bokeh.plotting.figure()

skel['color'] = '#000000'

segments = skel.merge(skel, 'inner',
                           left_on=['rowId'],
                           right_on=['link'],
                           suffixes=['_child', '_parent'])

p.segment(x0='x_child', x1='x_parent',
          y0='z_child', y1='z_parent',
          color='color_child',
          line_width=3,
          source=segments, legend_label='')

bokeh.plotting.show(p)
# %%


with open(os.path.join(data_dir, 'healed_{}.swc'.format(bodyId)), 'w') as file:
    file.write(healed_swc)

 #%%
import numpy as np
import open3d as o3d


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:6]/255)
pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,6:9])

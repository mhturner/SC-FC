#!/usr/bin/env python
import ctypes
import os
import sys

import pyglet
from pyglet.gl import *

from pywavefront import visualization
import pywavefront
import numpy as np

from scfc import bridge
import matplotlib.pyplot as plt

resource_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data/neuprint_meshes'

rotation = 0

mapping = bridge.getRoiMapping()
pull_regions = ['AL(R)', 'CAN(R)', 'LH(R)', 'SPS(R)']


cmap = plt.get_cmap('Set3')
colors = cmap(np.arange(len(pull_regions))/len(pull_regions))

scenes = []
# for key in mapping:
#     for roi in mapping[key]:
#         if roi == 'aL(R)':
#             roi = 'aLobe(R)'
#         new_scene = pywavefront.Wavefront(os.path.join(resource_dir, '{}.obj'.format(roi)))
#         if roi in pull_regions:
#             c_ind = np.where(np.array(pull_regions) == roi)[0][0]
#             color = list(colors[c_ind, :])
#             color[3] = 0.75
#         else:
#             color = [0.25, 0.25, 0.25, 0.25]
#
#         new_scene.materials['default0'].set_ambient(color)
#         scenes.append(new_scene)
#         print('Loaded mesh: {}'.format(roi))

bodyId = 635062078
skeleton_scene = pywavefront.Wavefront(os.path.join(resource_dir, '{}.obj'.format(bodyId)))
skeleton_scene.materials['default0'].set_ambient([1, 0, 0, 1])
print('Loaded skelton')

window = pyglet.window.Window(resizable=True)
lightfv = ctypes.c_float * 4

glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

@window.event
def on_resize(width, height):
    viewport_width, viewport_height = window.get_framebuffer_size()
    glViewport(0, 0, viewport_width, viewport_height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., float(width)/height, 1., 100000.)

    glMatrixMode(GL_MODELVIEW)

    return True


@window.event
def on_draw():
    window.clear()
    glLoadIdentity()

    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-1.0, 1.0, 1.0, 0.0))
    glEnable(GL_LIGHT0)

    glTranslated(-25000.0, 0.0, -10000.0)
    glRotatef(225, 1.0, 0.0, 0.0)


    glEnable(GL_LIGHTING)


    for s_ind, scene in enumerate(scenes):
        visualization.draw(scene, lighting_enabled=True, textures_enabled=True)

    visualization.draw(skeleton_scene, lighting_enabled=True, textures_enabled=True)


def update(dt):
    global rotation
    rotation += 45.0 * dt

    if rotation > 720.0:
        rotation = 0.0


pyglet.clock.schedule(update)
pyglet.app.run()

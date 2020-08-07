#!/usr/bin/env python

from pathlib import Path
from pyrr import Matrix44
import moderngl
import moderngl_window as mglw
from moderngl_window.scene.camera import KeyboardCamera

import numpy as np
import os
from region_connectivity import RegionConnectivity


class CameraWindow(mglw.WindowConfig):
    """Base class with built in 3D camera support"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = KeyboardCamera(self.wnd.keys, aspect_ratio=self.wnd.aspect_ratio)
        self.camera_enabled = True

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        if action == keys.ACTION_PRESS:
            if key == keys.C:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.SPACE:
                self.timer.toggle_pause()

    def mouse_position_event(self, x: int, y: int, dx, dy):
        if self.camera_enabled:
            self.camera.rot_state(-dx, -dy)

    def resize(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)


class CubeModel(CameraWindow):
    aspect_ratio = 16 / 9
    title = "Cube Model"
    resource_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data/neuprint_meshes'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        mapping = RegionConnectivity.getRoiMapping()

        self.scenes = []
        # for key in mapping:
        #     for roi in mapping[key]:
        #         if roi == 'aL(R)':
        #             roi = 'aLobe(R)'
        #         self.scenes.append(self.load_scene('{}.obj'.format(roi)))

        self.scenes.append(self.load_scene('{}.obj'.format('AL(R)')))

        self.camera.projection.update(near=0.1, far=100000.0)

    def render(self, time: float, frametime: float):
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        translation = Matrix44.from_translation((0, 0, 0))
        rotation = Matrix44.from_eulers((0, 0, 0))
        model_matrix = translation * rotation
        camera_matrix = self.camera.matrix * model_matrix

        for scene in self.scenes:
            scene.apply_mesh_programs()
            scene.draw(
                projection_matrix=self.camera.projection.matrix,
                camera_matrix=camera_matrix,
                time=time
            )


if __name__ == '__main__':
    mglw.run_window_config(CubeModel)

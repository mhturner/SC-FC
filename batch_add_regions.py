import os
import bpy


def main():

    resource_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data/neuprint_meshes/regions'

    pull_regions = ['AL(R)', 'CAN(R)', 'LH(R)', 'SPS(R)']


    file_list = sorted(os.listdir(resource_dir))


    # loop through the strings in obj_list and add the files to the scene
    for item in file_list:
        path_to_file = os.path.join(resource_dir, item)
        # Load the object
        bpy.ops.import_scene.obj(filepath = path_to_file)
        # Scaling to get down to viewable region
        bpy.ops.transform.resize(value=(0.001, 0.001, 0.001))

import trimesh
import numpy as np

mesh = trimesh.load('00075213_9d4173fb39d54d16a718fe39_trimesh_007.obj')

vertices = mesh.vertices

xyz_max = vertices.max(axis=0)
xyz_min = vertices.min(axis=0)

scale = (xyz_max-xyz_min).max()
transform = np.eye(4)
transform[0,0] = 1/scale
transform[1,1] = 1/scale
transform[2,2] = 1/scale
transform[0,3] = -0.5
transform[1,3] = -0.5
transform[2,3] = -0.5

mesh.apply_transform(transform)


scene = trimesh.Scene(mesh)

# viewer = trimesh.viewer.SceneViewer(scene)

camera = trimesh.scene.cameras.Camera(resolution=(512,512),fov=(60,60),z_near=0.01,z_far=3)


import pdb; pdb.set_trace()
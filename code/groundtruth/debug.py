import sys
sys.path.append('../code')
import argparse
import GPUtil
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from collections import defaultdict
import open3d as o3d

import trimesh

conf = ConfigFactory().parse_file('confs/dtu-offset.conf')
dataset_conf = conf.get_config('dataset')

scan_id = 24

dataset_conf['scan_id'] = scan_id

dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)

cameras = np.load(dataset.cam_file)

global_scale_mat = cameras['scale_mat_0']

# mesh = o3d.io.read_triangle_mesh('/home/xn/datasets/DTU/SampleSet/MVS Data/Surfaces/tola/tola006_l3_surf_11_trim_8.ply') #TODO: load the correct pose
# mesh = o3d.io.read_point_cloud('/home/xn/datasets/DTU/Points/stl/stl016_total.ply')
mesh = o3d.io.read_triangle_mesh('/home/xn/datasets/DTU/Surfaces/tola/tola024_l3_surf_11_trim_8.ply')
# mesh = o3d.io.read_triangle_mesh('/home/xn/datasets/DTU/Surfaces/tola/tola016_l3_surf_11.ply')

# o3d.visualization.draw_geometries([mesh])

mesh.transform(np.linalg.inv(global_scale_mat))
# o3d.io.write_triangle_mesh('groundtruth/scaled_mesh.ply', mesh)
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(mesh)

lines3d_all = []

for it, inputs, gt in tqdm(dataset):
# input['lines_uniq] : lines by 
    lines = inputs['lines_uniq'] #(xyxy,score) //Nx5
    pose = inputs['pose'] #c2w, 4x4
    K = inputs['intrinsics'] #
    K_inv = np.linalg.inv(K)

    ones = np.ones([lines.shape[0], 2])

    junc1 = np.concatenate([lines[:, 0:2], ones], axis=-1)  # N, 4
    junc2 = np.concatenate([lines[:, 2:4], ones], axis=-1)  # N, 4
    # junc3 = 0.5*(junc1+junc2)
    
    junc1_3d = (pose @ (K_inv @ junc1.T))[:3]  # 3, N
    junc2_3d = (pose @ (K_inv @ junc2.T))[:3]
    # junc3_3d = (pose @ (K_inv @ junc3.T))[:3]

    ray_o = np.concatenate([np.zeros([3, 1]), np.ones([1, 1])], axis=0)
    ray_o = (pose @ ray_o)[:3]  # 3, 1
    ray_o = np.repeat(ray_o, lines.shape[0], axis=1)  # 3, N

    
    ray_d1 = junc1_3d - ray_o
    ray_d2 = junc2_3d - ray_o
    # ray_d3 = junc3_3d - ray_o
    ray_d1 = ray_d1 / (np.linalg.norm(ray_d1, axis=0, keepdims=True)+1e-10)
    ray_d2 = ray_d2 / (np.linalg.norm(ray_d2, axis=0, keepdims=True)+1e-10)  # 3, N
    # ray_d3 = ray_d3 / (np.linalg.norm(ray_d3, axis=0, keepdims=True)+1e-10)  # 3, N

    ray1 = np.concatenate([ray_o, ray_d1], axis=0).transpose(1, 0)  # N, 6
    ray2 = np.concatenate([ray_o, ray_d2], axis=0).transpose(1, 0)  # N, 6
    # ray3 = np.concatenate([ray_o, ray_d3], axis=0).transpose(1,0)

    ans1 = scene.cast_rays(ray1.astype(np.float32))
    dis1 = ans1['t_hit'].numpy().reshape(1, -1)
    dis1[dis1 > 100] = 0
    ans2 = scene.cast_rays(ray2.astype(np.float32))
    dis2 = ans2['t_hit'].numpy().reshape(1, -1)
    dis2[dis2 > 100] = 0
    # ans3 = scene.cast_rays(ray3.astype(np.float32))
    # dis3 = ans3['t_hit'].numpy().reshape(1,-1)
    # dis3[dis3 > 100] = 0
    mask = (dis1 > 0.0001) & (dis2 > 0.0001)
    # print(mask.sum())
    for t in np.linspace(0, 1,32)[1:-1]:
        junct = t*junc1 + (1-t)*junc2
        junct_3d =  (pose @ (K_inv @ junct.T))[:3]
        ray_dt = junct_3d - ray_o
        rayt = np.concatenate([ray_o, ray_dt],axis=0).transpose(1,0)
        anst = scene.cast_rays(rayt.astype(np.float32))
        dist = anst['t_hit'].numpy().reshape(1,-1)
        dist[dist>100] = 0
        mask = mask & (dist > 0.0001)
        # print(mask.sum())

    # import pdb; pdb.set_trace()




    junc1_3d = ray_o + ray_d1 * dis1 * mask.astype(np.float32)  # 3, N
    junc2_3d = ray_o + ray_d2 * dis2 * mask.astype(np.float32)

    for t in np.linspace(0, 1,32)[1:-1]:
        x = junc1_3d*t + junc2_3d*(1-t)
        x = x.t().float().numpy()
        f = scene.compute_distance(x)
        f = f.numpy()
        mask = mask*(f.reshape(1,-1)<4e-3)

    lines3d = np.stack([junc1_3d.transpose(1, 0), junc2_3d.transpose(1, 0)], axis=1)
    lines3d = lines3d[mask.flatten()]

    scale_mat = np.load(dataset.cam_file)['scale_mat_0']
    # trimesh.load_path(lines3d).show()
    # junc3_3d = ray_o + ray_d3 * dis3 * mask.astype(np.float32)
    # lines3d = np.stack([junc1_3d.transpose(1, 0), junc2_3d.transpose(1, 0)], axis=1)  # N, 2, 3

    lines3d_all.append(lines3d.astype(np.float32))

    # path = trimesh.load_path(lines3d)
    
    # center = trimesh.points.PointCloud(junc3_3d.transpose(1,0))
    # center.colors = np.array([255,0,0])
    # trimesh.Scene([path]).show()

    # trimesh.load_path(np.concatenate(lines3d_all,axis=0)).show()
    # import pdb; pdb.set_trace()
    # break
    # if it > 10:
        # break

lines3d_all = np.concatenate(lines3d_all, axis=0)

trimesh.load_path(lines3d_all).show()

t = np.linspace(0,1,32).reshape(1,-1,1)

points = lines3d_all[:,:1]*t + lines3d_all[:,1:]*(1-t)
points = points.reshape(-1,3)

points = (global_scale_mat[:3,:3]@points.transpose(1,0)) + global_scale_mat[:3,3:]
points = points.transpose(1,0)
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.reshape(-1,3)))
np.savez('groundtruth/gt_wireframe.npz', lines3d=lines3d_all)
o3d.io.write_point_cloud('groundtruth/gt_wireframe.ply',pcd)
# np.savetxt('groundtruth/gt.txt', lines3d_all.reshape(-1, 3))

# gt_mesh = trimesh.load_mesh('groundtruth/scaled_mesh.ply')
# scene = trimesh.scene.scene.Scene()
# scene.add_geometry(lines3d_all)
# scene.show()
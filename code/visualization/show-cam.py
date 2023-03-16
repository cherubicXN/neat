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
import trimesh
import open3d as o3d 
import cv2
import glob
import copy
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from pyquaternion import Quaternion

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=np.float32)

rot_psi = lambda psi : np.array([
    [np.cos(psi),-np.sin(psi),0,0],
    [np.sin(psi), np.cos(psi),0,0],
    [0,0,1,0],
    [0,0,0,1],
], dtype=np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
], dtype=np.float32)


def pose_spherical(psi, theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = rot_psi(psi/180.*np.pi) @c2w
    c2w = np.array(
        [
            [0,0,-1,0],
            [1,0,0,0],
            [0,-1,0,0],
         [0,0,0,1]]) @ c2w
    return c2w

def project2D(points, K, R, T):
    x = points.reshape(-1,3)
def get_cam(c2w_, image = None):
    if isinstance(c2w_, torch.Tensor):
        c2w = c2w_.cpu().numpy()
    else:
        c2w = c2w_

    center = c2w[:3,3]
    x = c2w[:3,0]*0.1
    y = c2w[:3,1]*0.1
    z = c2w[:3,2]*0.1

    tz = trimesh.load_path(np.stack((center,center+z),axis=0))
    tz.colors = np.array([[255,0,0]])
    ty = trimesh.load_path(np.stack((center,center+y),axis=0))
    ty.colors = np.array([[0,255,0]])
    tx = trimesh.load_path(np.stack((center,center+x),axis=0))
    tx.colors = np.array([[0,0,255]])

    x0 = center-x-y
    x1 = center-x+y
    x2 = center+x+y
    x3 = center+x-y

    cam = np.array([
        (x0,x1),(x1,x2),(x2,x3),(x3,x0),
        (x0,center-z),
        (x1,center-z),
        (x2,center-z),
        (x3,center-z),
    ])
    
    colors = np.array([
        (255,0,0,64),
        # (0,255,0),(0,255,0),(0,255,0),(0,255,0),
    ])
    cam = trimesh.load_path(cam)
    colors = np.tile(colors,(cam.entities.shape[0],1))
    cam.colors = colors

    return cam

def slerp_interpolate(p0, p1, t):
    """Spherical linear interpolation between two points"""
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

def interpolate_camera_poses(R1, T1, R2, T2, num_frames):
    # Convert rotation matrices to quaternions
    slerp = Slerp([0,1], Rotation.from_matrix([R1,R2]))

    t = np.linspace(0,1,num_frames)
    Ri = slerp(t).as_matrix()
    Ti = t.reshape(-1,1)*T1[None,:] + (1-t).reshape(-1,1)*T2[None,:]
    
    return R_i, T_i

def linesToOpen3d(lines):
    if isinstance(lines, torch.Tensor):
        lines = lines.cpu().numpy()
    
    num_lines = lines.shape[0]
    points = lines.reshape(-1,3)
    edges = np.arange(num_lines*2).reshape(-1,2)

    lineset = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector2iVector(edges),
    )
    return lineset

def create_junctions(junctions, radius=0.01):
    N = junctions.shape[0]
    spheres = o3d.geometry.TriangleMesh()
    for i in range(N):
        obj = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        vertices = np.array(obj.vertices)
        colors = np.zeros_like(vertices)
        colors[:,-1] = 1
        obj.vertex_colors = o3d.utility.Vector3dVector(colors)
        obj.translate(junctions[i])
        spheres+=obj
    return spheres

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str,required=True,help='the path of the reconstructed wireframe model')
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--save-path', default=None, type=str)
    parser.add_argument('--cams', default=None, type=str, help='the path of the camera poses')
    parser.add_argument('--name', default='video', type=str)
    

    opt = parser.parse_args()

    if opt.save:
            opt.save = os.path.join(os.path.dirname(opt.data),'..',opt.name)
    else:
        opt.save = None

    if opt.save_path is not None:
        opt.save = opt.save_path

    if opt.save:
        os.makedirs(opt.save,exist_ok=True)
    data = np.load(opt.data,allow_pickle=True)

    lines3d = data['lines3d']

    if len(lines3d.shape) == 1:
        lines3d = np.concatenate(lines3d,axis=0)
    
    junctions = np.unique(lines3d.reshape(-1,3),axis=0)

    dis1 = np.linalg.norm(junctions[:,None]-lines3d[None,:,0],axis=-1)
    dis2 = np.linalg.norm(junctions[:,None]-lines3d[None,:,1],axis=-1)
    id1 = np.argmin(dis1,axis=0)
    id2 = np.argmin(dis2,axis=0)
    edges = np.stack((id1,id2),axis=1)

    lineset_obj = o3d.geometry.LineSet(o3d.utility.Vector3dVector(junctions),o3d.utility.Vector2iVector(edges))

    junctions_obj = create_junctions(junctions)
    

    camera = np.load(opt.cams,allow_pickle=True)

    K = camera['intrinsics']

    poses = camera['extrinsics']

    cam_objs = [o3d.geometry.LineSet.create_camera_visualization(512,512,Ki,np.linalg.inv(pi),0.1) for Ki,pi in zip(K,poses)]
    cam_obj_sum = cam_objs[0]
    for i in range(1,len(cam_objs)):
        cam_obj_sum += cam_objs[i]

    
    o3d.io.write_line_set(os.path.join(opt.save,'lineset.ply'),lineset_obj)
    o3d.io.write_line_set(os.path.join(opt.save,'cameras.ply'),cam_obj_sum)
    o3d.io.write_triangle_mesh(os.path.join(opt.save,'junctions.ply'),junctions_obj)
    # import pdb; pdb.set_trace()

    # o3d.visualization.draw_geometries([lineset_obj]+junctions_obj+cam_objs)
    # o3d.visualization.draw_geometries(cam_objs)
    # print(lines3d.shape)
    # WireframeVisualizer(lines3d,opt.save, None, rx=rx,ry=ry,rz=rz,t=t, points3d_all=None, show_endpoints=opt.show_points,
    # line_width=opt.line_width,
    # camera_path=opt.cams)

    # opt = visualizer


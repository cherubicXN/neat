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

def Visualizer(points, render_dir = None, cam_dir = None, rx=0, ry=0,rz=0,t=0):

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from collections import deque

    if render_dir is not None:
        os.makedirs(render_dir,exist_ok=True)
    
    Visualizer.view_cnt = 0
    Visualizer.render_dir = render_dir
    Visualizer.camera_path = []
    Visualizer.image_path = []
    Visualizer.vis = o3d.visualization.VisualizerWithKeyCallback()
    Visualizer.points = points
    Visualizer.rot_psi = rx
    Visualizer.rot_theta = ry
    Visualizer.rot_phi = rz
    Visualizer.t = t
    project_points = {}

    def load_view(vis):
        ctr = vis.get_view_control()
        glb = Visualizer
        if len(glb.external_cams) > 0:
            path = glb.external_cams.popleft()
            glb.external_cams.append(path)
            print('loading viewpoint from {}'.format(path))
        else:
            print('there is no precomputed viewpoints available')
            return False
        cam = o3d.io.read_pinhole_camera_parameters(path)
        ctr.convert_from_pinhole_camera_parameters(cam)

        return False
    
    def adjust_viewpoint(vis, rx,ry,rz,t):
        ctr = vis.get_view_control()
        glb = Visualizer

        cam = ctr.convert_to_pinhole_camera_parameters()
        glb.rot_psi += rx
        glb.rot_theta += ry
        glb.rot_phi += rz
        glb.t += t

        print('(rot_x,rot_y,rot_z,t) = ({:.3f},{:.3f},{:.3f},{:.3f})'.format(
            glb.rot_psi,
            glb.rot_theta,
            glb.rot_phi,
            glb.t,
        ))

        extrinsic = pose_spherical(glb.rot_psi, glb.rot_theta, glb.rot_phi, glb.t)
        extrinsic = np.linalg.inv(extrinsic)

        cam.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(cam)
        ctr.set_lookat(np.zeros(3))
        return False

    def rotate(vis):
        ctr = vis.get_view_control()
        # ctr.rotate(10,0)

        cam = ctr.convert_to_pinhole_camera_parameters()
        glb = Visualizer

        # extrinsic = pose_spherical(glb.rot_phi, glb.rot_theta, 3)
        extrinsic = pose_spherical(glb.rot_psi, glb.rot_theta, glb.rot_phi, glb.t)
        extrinsic = np.linalg.inv(extrinsic)
        cam.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(cam)
        ctr.set_lookat(np.zeros(3))
        glb.rot_theta+=5
        glb.rot_theta = glb.rot_theta%360

        if glb.rot_theta in project_points:
            print('finished')
            return False
        
        param = ctr.convert_to_pinhole_camera_parameters()
        height, width = param.intrinsic.height, param.intrinsic.width
        glb.width = width
        glb.height=height
        K = param.intrinsic.intrinsic_matrix
        R = param.extrinsic[:3,:3]
        T = param.extrinsic[:3,3:]

        x = points.copy()
        x2d = K@(R@x.transpose()+T)
        x2d = x2d[:2]/x2d[2:]
        x2d = x2d.transpose()
        
        project_points[glb.rot_theta] = x2d
        return False

    vis = Visualizer.vis
    vis.create_window(height=512,width=512, left=0, top=0, visible=True, window_name='Point Cloud Visualizer')

    render_option = vis.get_render_option()
    vis.add_geometry(pcd)

    adjust_viewpoint(vis,0,0,0,0)
    render_option.line_width = 3
    vis.register_key_callback(ord('R'), rotate)
    vis.register_key_callback(ord('W'), lambda x: adjust_viewpoint(x,5,0,0,0))
    vis.register_key_callback(ord('E'), lambda x: adjust_viewpoint(x,-5,0,0,0))
    vis.register_key_callback(ord('S'), lambda x: adjust_viewpoint(x,0,5,0,0))
    vis.register_key_callback(ord('D'), lambda x: adjust_viewpoint(x,0,-5,0,0))
    vis.register_key_callback(ord('X'), lambda x: adjust_viewpoint(x,0,0,5,0))
    vis.register_key_callback(ord('C'), lambda x: adjust_viewpoint(x,0,0,-5,0))
    vis.register_key_callback(ord('A'), lambda x: adjust_viewpoint(x,0,0,0,0.1))
    vis.register_key_callback(ord('Z'), lambda x: adjust_viewpoint(x,0,0,0,-0.1))
    # vis.register_key_callback(ord('H'), print_pose)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

    if render_dir is None:
        return 

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(height=512,width=512, left=0, top=0, visible=True, window_name='Wireframe Visualizer')
    render_option = vis.get_render_option()
    vis.add_geometry(pcd)
    adjust_viewpoint(vis,0,0,0,0)
    vis.register_animation_callback(rotate)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

    keys = sorted(project_points.keys())

    width = Visualizer.width
    height = Visualizer.height

    # fig = plt.figure()

    import os.path as osp
    from tqdm import tqdm 
    os.makedirs(render_dir,exist_ok=True)
    for i,key in enumerate(tqdm(keys)):
        points2d = project_points[key]
        fig = plt.figure()
        fig.set_size_inches(width/height,1,forward=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.xlim([-0.5, width-0.5])
        plt.ylim([height-0.5, -0.5])
        # plt.plot([lines2d[:,0,0],lines2d[:,1,0]],[lines2d[:,0,1],lines2d[:,1,1]],'-',color='black',linewidth=0.05)
        colors = np.stack((points2d[:,0]/width,points2d[:,1]/height,np.ones_like(points2d[:,0])),axis=-1)
        colors = colors/np.linalg.norm(colors,axis=-1,keepdims=True)
        colors = np.abs(colors)
        plt.scatter(points2d[:,0],points2d[:,1],color = colors, s=0.2,edgecolors='none',zorder=5)
        path = osp.join(render_dir,'{:04d}.png'.format(i))
        plt.savefig(path,dpi=width)
        plt.close(fig)
    
    rendered_images = []
    for i in range(len(keys)):
        path = osp.join(render_dir,'{:04d}.png'.format(i))
        image = cv2.imread(path)
        rendered_images.append(image)

    output_size = rendered_images[0].shape[:2]
    output_size = (output_size[1],output_size[0])
    out = cv2.VideoWriter(render_dir+'.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30 , output_size)
    for im in rendered_images:
        out.write(im)
    out.release()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/pointclouds/000000.ply')
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--pose', default=None, type=str, choices=['dtu'])

    opt = parser.parse_args()

    if opt.save:
        opt.save = opt.path.rstrip('.txt')+'_record'
    else:
        opt.save = None

    if opt.pose == 'dtu':
        rx = -155
        ry = 0
        rz = -25
        t  = 6
    else:
        rx = ry = rz = 0
        t = 6

    points3d = []
    with open(opt.path) as f:
        for line in f:
            if line[0] == '#':
                continue
            data = line.split()
            xyz = np.array([float(x) for x in data[1:4]])
            points3d.append(xyz)
    
    points3d = np.array(points3d)

    Visualizer(points3d, opt.save, None, rx=rx, ry=ry, rz=rz, t=t)
    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points3d))
    # o3d.visualization.draw_geometries([pcd])
    # import pdb; pdb.set_trace()
    # pcd = o3d.io.read_point_cloud(args.path)
    # points = np.asarray(pcd.points)
    # print(points.shape)

    # o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
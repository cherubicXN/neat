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

def WireframeVisualizer(pcd, render_dir = None, cam_dir = None, rx=0, ry=0,rz=0,t=0, points3d_all = None, show_endpoints = False):
    import matplotlib.pyplot as plt
    from collections import deque

    if render_dir is not None:
        os.makedirs(render_dir,exist_ok=True)
    
    WireframeVisualizer.view_cnt = 0
    WireframeVisualizer.render_dir = render_dir
    WireframeVisualizer.camera_path = []
    WireframeVisualizer.image_path = []
    WireframeVisualizer.vis = o3d.visualization.VisualizerWithKeyCallback()
    WireframeVisualizer.rot_psi = rx
    WireframeVisualizer.rot_theta = ry
    WireframeVisualizer.rot_phi = rz
    WireframeVisualizer.t = t
    WireframeVisualizer.done = False

    if render_dir is not None:
        WireframeVisualizer.view_cnt = len(os.listdir(render_dir))
    min_w = 10000
    min_h = 10000
    max_w = 0
    max_h = 0

    project_lines = {}

    if cam_dir is not None:
        cam_files = sorted(glob.glob(os.path.join(cam_dir,'*.json')))
        WireframeVisualizer.external_cams = deque(cam_files)

    def load_view(vis):
        ctr = vis.get_view_control()
        glb = WireframeVisualizer
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
    
    def capture_image(vis):
        ctr = vis.get_view_control()
        glb = WireframeVisualizer
        param = ctr.convert_to_pinhole_camera_parameters()
        glb.view_cnt +=1
        image = vis.capture_screen_float_buffer()
        image = np.asarray(image)*255
        image = np.asarray(image,dtype=np.uint8)

        height, width = param.intrinsic.height, param.intrinsic.width
        K = param.intrinsic.intrinsic_matrix
        R = param.extrinsic[:3,:3]
        T = param.extrinsic[:3,3:]

        x3d = np.array(pcd.points)
        x2d = K@R@x3d.T + K@T
        x2d = x2d/x2d[2:,:]
        
        fig = plt.figure()
        fig.set_size_inches(width/height,1,forward=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        for i in range(x2d.shape[1]):
            plt.scatter(x2d[0,i],x2d[1,i],s=0.75,edgecolors='none',zorder=5)
        # plt.scatter(x2d[0],x2d[1],s=2.0,edgecolors='none',zorder=5)
        plt.xlim([-0.5, width-0.5])
        plt.ylim([height-0.5, -0.5])
        plt.savefig(os.path.join(glb.render_dir,'{:06d}.png'.format(glb.view_cnt)),dpi=height)
        
        plt.close()
        return False
 
    def adjust_viewpoint(vis, rx,ry,rz,t):
        ctr = vis.get_view_control()
        glb = WireframeVisualizer

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
        glb = WireframeVisualizer

        # extrinsic = pose_spherical(glb.rot_phi, glb.rot_theta, 3)
        extrinsic = pose_spherical(glb.rot_psi, glb.rot_theta, glb.rot_phi, glb.t)
        extrinsic = np.linalg.inv(extrinsic)
        cam.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(cam)
        ctr.set_lookat(np.zeros(3))
        glb.rot_theta+=5
        glb.rot_theta = glb.rot_theta%360

        if glb.rot_theta in project_lines and not glb.done:
            print('finished')
            glb.done = True
            return False
        
        param = ctr.convert_to_pinhole_camera_parameters()
        height, width = param.intrinsic.height, param.intrinsic.width
        glb.width = width
        glb.height=height
        K = param.intrinsic.intrinsic_matrix
        R = param.extrinsic[:3,:3]
        T = param.extrinsic[:3,3:]

        x = lines.reshape(-1,3)
        x2d = K@(R@x.transpose()+T)
        x2d = x2d[:2]/x2d[2:]
        x2d = x2d.transpose()
        lines2d = x2d.reshape(-1,2,2)
        project_lines[glb.rot_theta] = lines2d
        return False
    

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0,0,0])
    vis = WireframeVisualizer.vis
    vis.create_window(height=512,width=512, left=0, top=0, visible=True, window_name='Wireframe Visualizer')
    # vis.create_window()
    render_option = vis.get_render_option()
    vis.add_geometry(pcd)

    if points3d_all is not None:
        vis.add_geometry(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points3d_all))
        )

    # adjust_psi(vis, sign=0)
    adjust_viewpoint(vis,0,0,0,0)
    # vis.add_geometry(mesh_frame)
    render_option.line_width = 3
    vis.register_key_callback(ord('P'), capture_image)
    # vis.register_key_callback(ord('L'), load_view)
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

    
    # print save path
    # print('Saving to {}'.format(render_dir))
        # plt.savefig('{}/{:04d}.png')
    
    # o3d.visualization.draw_geometries_with_key_callbacks([lineset],key_to_call_back)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str,required=True,help='the path of the reconstructed wireframe model')
    # parser.add_argument('--imgdir', type=None,)
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--pose', default=None, type=str, choices=['dtu','scan'])
    parser.add_argument('--threshold', default=None, type=float)
    

    opt = parser.parse_args()

    if opt.save:
        opt.save = os.path.join(os.path.dirname(opt.data),'..','junctions_record')
        if not os.path.exists(opt.save):
            os.makedirs(opt.save)
    else:
        opt.save = None

    if opt.pose == 'dtu':
        rx = -155
        ry = 0
        rz = -25
        t  = 3
    elif opt.pose == 'scan':
        rx = 0
        ry = 170
        rz = -45
        t = 3
    else:
        rx = ry = rz = 0
        t = 3

    points = torch.load(opt.data).cpu().numpy()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    
    
    # cam = np.load('../data/DTU/scan24/cameras.npz')
    # import pdb; pdb.set_trace()
    # lineset_o3d = linesToOpen3d(lines3d)
    WireframeVisualizer(pcd, opt.save, None, rx=rx,ry=ry,rz=rz,t=t)
    

    # opt = visualizer


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

def WireframeVisualizer(mesh, render_dir = None, cam_dir = None, rx=0, ry=0,rz=0,t=0):
#, points3d_all = None, show_endpoints = False):
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
    WireframeVisualizer.stop = False
    WireframeVisualizer.done = False


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
        
    
    def capture_image(vis):
        # view_cnt += 1
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

        if glb.render_dir is not None:
            img_path = os.path.join(glb.render_dir,'image_{:04d}.png'.format(glb.view_cnt))
            cam_path = os.path.join(glb.render_dir,'cam_{:04d}.json'.format(glb.view_cnt))

            glb.image_path.append(img_path)
            glb.camera_path.append(cam_path)

            fig = plt.figure()
            fig.set_size_inches(width/height,1,forward=False)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(image)
            plt.savefig(img_path,dpi=600)
            # plt.show()
            # plt.xlim([-0.5, width-0.5])
            # plt.ylim([height-0.5, -0.5])
            # plt.plot([lines2d[:,0,0],lines2d[:,1,0]],[lines2d[:,0,1],lines2d[:,1,1]],'-',color='black',linewidth=0.05)
            # # plt.plot(
            # #     [lines_display[:,0],lines_display[:,2]],
            # #     [lines_display[:,1],lines_display[:,3]],
            # #     'r-',
            # #     linewidth = 0.5 if args.saveto else None
            # # )
            # if show_endpoints:
            #     plt.scatter(lines2d[:,0,0],lines2d[:,0,1],color='b',s=1.2,edgecolors='none',zorder=5)
            #     plt.scatter(lines2d[:,1,0],lines2d[:,1,1],color='b',s=1.2,edgecolors='none',zorder=5)
            # # cv2.imwrite(img_path,image)
            # plt.savefig(img_path,dpi=600)
            # glb.stop = True
            # o3d.io.write_pinhole_camera_parameters(cam_path,param)
            # print('saving the rendered image into {}'.format(img_path))
            # print('saving the rendering viewpoint into {}'.format(cam_path))
        else:
            print('the rendering path is None')
        return False

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0,0,0])
    vis = WireframeVisualizer.vis
    vis.create_window(height=512,width=512, left=0, top=0, visible=True, window_name='Wireframe Visualizer')
    # vis.create_window()
    render_option = vis.get_render_option()
    vis.add_geometry(mesh)
    adjust_viewpoint(vis,0,0,0,0)
    # vis.add_geometry(mesh_frame)
    render_option.line_width = 3
    vis.register_key_callback(ord('P'), capture_image)
    # vis.register_key_callback(ord('L'), load_view)
    # vis.register_key_callback(ord('R'), rotate)
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

    # adjust_psi(vis, sign=0)
   
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
    parser.add_argument('--show-points', default=False, action='store_true')
    parser.add_argument('--threshold', default=None, type=float)
    

    opt = parser.parse_args()

    if opt.save:
        if opt.threshold is not None:
            opt.save = opt.data.rstrip('.npz')+'_record_{}'.format(opt.threshold)
        else:
            opt.save = opt.data.rstrip('.npz')+'_record'
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

    mesh = o3d.io.read_triangle_mesh(opt.data)
    center = mesh.get_center()
    max_bound = np.max(mesh.get_max_bound() - mesh.get_min_bound())
    mesh.translate(-center)
    mesh.scale(1/max_bound*2,center*0)
    # import pdb; pdb.set_trace()
    
    WireframeVisualizer(mesh,opt.save, None, rx=rx,ry=ry,rz=rz,t=t)

    # opt = visualizer


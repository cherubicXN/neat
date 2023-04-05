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

def WireframeVisualizer(lines, 
    render_dir = None, cam_dir = None, rx=0, ry=0,rz=0,t=0, points3d_all = None, show_endpoints = False, line_width=0.03, camera_path = None):
    lineset = linesToOpen3d(lines)
    import matplotlib.pyplot as plt
    from collections import deque

    if render_dir is not None:
        os.makedirs(render_dir,exist_ok=True)
    
    WireframeVisualizer.view_cnt = 0
    WireframeVisualizer.render_dir = render_dir
    WireframeVisualizer.camera_path = []
    WireframeVisualizer.image_path = []
    WireframeVisualizer.vis = o3d.visualization.VisualizerWithKeyCallback()
    WireframeVisualizer.lines = lines
    WireframeVisualizer.rot_psi = rx
    WireframeVisualizer.rot_theta = ry
    WireframeVisualizer.rot_phi = rz
    WireframeVisualizer.t = t
    WireframeVisualizer.stop = False
    WireframeVisualizer.done = False
    WireframeVisualizer.saved_views = []
    WireframeVisualizer.slerp_t = 0
    WireframeVisualizer.dt = 0.01


    min_w = 10000
    min_h = 10000
    max_w = 0
    max_h = 0

    project_lines = {}

    if camera_path is not None:
        import json 
        with open(camera_path,'r') as f:
            cam_data = json.load(f)
        WireframeVisualizer.saved_views = [np.array(x,dtype=np.float32) for x in cam_data]
        WireframeVisualizer.slerp_fn = Slerp([0,1], Rotation.from_matrix([x[:3,:3] for x in WireframeVisualizer.saved_views]))

    def save_view(vis):
        ctr = vis.get_view_control()
        glb = WireframeVisualizer

        cam = ctr.convert_to_pinhole_camera_parameters()
        K = cam.intrinsic.intrinsic_matrix
        RT = cam.extrinsic
        glb.saved_views.append(RT)
        if len(glb.saved_views) > 2:
            glb.saved_views.pop(0)

        print('{} views saved'.format(len(glb.saved_views)))
        # ctr.convert_from_pinhole_camera_parameters(cam)

        if len(glb.saved_views) == 2:
            glb.slerp_fn = Slerp([0,1], Rotation.from_matrix([x[:3,:3] for x in glb.saved_views]))
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
    
    def slerp(vis):
        ctr = vis.get_view_control()
        # K = WireframeVisualizer.saved_views[0][0]
        start = WireframeVisualizer.saved_views[0]
        end = WireframeVisualizer.saved_views[1]
        R1 = start[:3,:3]
        R2 = end[:3,:3]
        Rot = Rotation.from_matrix([R1,R2])#.as_quat()
        

        Ri = WireframeVisualizer.slerp_fn(WireframeVisualizer.slerp_t).as_matrix()
        Ti = WireframeVisualizer.slerp_t*start[:3,3] + (1-WireframeVisualizer.slerp_t)*end[:3,3]
        extrinsic = np.eye(4)
        extrinsic[:3,:3] = Ri
        extrinsic[:3,3] = Ti
        
        cam = ctr.convert_to_pinhole_camera_parameters()
        cam.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(cam)
        
        if WireframeVisualizer.slerp_t + WireframeVisualizer.dt > 1:
            # WireframeVisualizer.slerp_t = 1
            WireframeVisualizer.dt *= -1
        elif WireframeVisualizer.slerp_t + WireframeVisualizer.dt < 0:
            # WireframeVisualizer.slerp_t = 0
            WireframeVisualizer.dt *= -1
        # else:
        WireframeVisualizer.slerp_t += WireframeVisualizer.dt

        param = ctr.convert_to_pinhole_camera_parameters()
        height, width = param.intrinsic.height, param.intrinsic.width
        WireframeVisualizer.width = width
        WireframeVisualizer.height=height

        K = param.intrinsic.intrinsic_matrix
        x = lines.reshape(-1,3)
        x2d = K@(Ri@x.transpose()+Ti[:,None])        
        x2d = x2d[:2]/x2d[2:]
        x2d = x2d.transpose()
        lines2d = x2d.reshape(-1,2,2)
        index = int(WireframeVisualizer.slerp_t/abs(WireframeVisualizer.dt))
        project_lines[index] = lines2d
        
        
        if index in project_lines and not WireframeVisualizer.done:
            print('finished')
            WireframeVisualizer.done = True
            return False
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
    vis.add_geometry(lineset)
    if points3d_all is not None:
        vis.add_geometry(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points3d_all))
        )

    if camera_path is None:
        adjust_viewpoint(vis,0,0,0,0)
        # vis.add_geometry(mesh_frame)
        render_option.line_width = 3
        vis.register_key_callback(ord('T'), save_view)
        vis.register_key_callback(ord('J'), slerp)
        vis.register_key_callback(ord('W'), lambda x: adjust_viewpoint(x,5,0,0,0))
        vis.register_key_callback(ord('E'), lambda x: adjust_viewpoint(x,-5,0,0,0))
        vis.register_key_callback(ord('S'), lambda x: adjust_viewpoint(x,0,5,0,0))
        vis.register_key_callback(ord('D'), lambda x: adjust_viewpoint(x,0,-5,0,0))
        vis.register_key_callback(ord('X'), lambda x: adjust_viewpoint(x,0,0,5,0))
        vis.register_key_callback(ord('C'), lambda x: adjust_viewpoint(x,0,0,-5,0))
        vis.register_key_callback(ord('A'), lambda x: adjust_viewpoint(x,0,0,0,0.1))
        vis.register_key_callback(ord('Z'), lambda x: adjust_viewpoint(x,0,0,0,-0.1))
        
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        vis.destroy_window()

    # if render_dir is None or WireframeVisualizer.stop:
    #     return 
        
    vis = o3d.visualization.VisualizerWithKeyCallback()
    # vis.create_window()
    vis.create_window(height=512,width=512, left=0, top=0, visible=True, window_name='Wireframe Visualizer')
    render_option = vis.get_render_option()
    vis.add_geometry(lineset)
    
    # adjust_psi(vis, sign=0)
    adjust_viewpoint(vis,0,0,0,0)
    # vis.register_animation_callback(lambda x: adjust_viewpoint(x, 0, 5, 0, 0))
    vis.register_animation_callback(slerp)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

    keys = sorted(project_lines.keys())

    width = WireframeVisualizer.width
    height = WireframeVisualizer.height

    # fig = plt.figure()

    import os.path as osp
    from tqdm import tqdm 
    os.makedirs(render_dir,exist_ok=True)

    keys = keys + keys[::-1][1:]
    for i,key in enumerate(tqdm(keys)):
        lines2d = project_lines[key]
        # lines2d[:,0,:] -= [xmin,ymin]
        # lines2d[:,1,:] -= [xmin,ymin]

        fig = plt.figure()
        fig.set_size_inches(width/height,1,forward=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.xlim([-0.5, width-0.5])
        plt.ylim([height-0.5, -0.5])
        # plt.plot([lines2d[:,0,0],lines2d[:,1,0]],[lines2d[:,0,1],lines2d[:,1,1]],'-',color='black',linewidth=line_width)
        if show_endpoints:
            plt.scatter(lines2d[:,0,0],lines2d[:,0,1],color='b',s=0.2,edgecolors='none',zorder=5)
            plt.scatter(lines2d[:,1,0],lines2d[:,1,1],color='b',s=0.2,edgecolors='none',zorder=5)
            # plt.plot(lines2d[:,0,0],lines2d[:,0,1],'o',color='black',markersize=0.1)
            # plt.plot(lines2d[:,1,0],lines2d[:,1,1],'o',color='black',markersize=0.1)
        path = osp.join(render_dir,'{:04d}.pdf'.format(i))
        plt.savefig(path,dpi=width)
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

    print('Done! Saved to {}'.format(render_dir+'.mp4'))
    import imageio
    imageio.mimsave(render_dir+'.gif', [cv2.cvtColor(img,cv2.COLOR_BGR2RGB) for img in rendered_images], fps=30)

    import json
    with open(osp.join(render_dir,'cam.json'),'wb') as f:
        saved_views = WireframeVisualizer.saved_views

        cam_data = [k.tolist() for k in saved_views]
        f.write(json.dumps(cam_data).encode('utf-8'))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str,required=True,help='the path of the reconstructed wireframe model')
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--save-path', default=None, type=str)
    parser.add_argument('--cams', default=None, type=str, help='the path of the camera poses')
    parser.add_argument('--pose', default=None, type=str, choices=['dtu','scan'])
    parser.add_argument('--show-points', default=False, action='store_true')
    parser.add_argument('--line-width', default=0.03, type=float)
    parser.add_argument('--name', default='video', type=str)
    

    opt = parser.parse_args()

    if opt.save:
            opt.save = os.path.join(os.path.dirname(opt.data),'..',opt.name)
    else:
        opt.save = None

    if opt.save_path is not None:
        opt.save = os.path.join(opt.save_path,opt.name)

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


    data = np.load(opt.data,allow_pickle=True)

    lines3d = data['lines3d']

    if len(lines3d.shape) == 1:
        lines3d = np.concatenate(lines3d,axis=0)

    # if opt.show_points and 'points3d_all' in data:
    #     points3d = np.concatenate(data['points3d_all'])
    # el       points3d = None
        # else:
            
    

    print(lines3d.shape)
    WireframeVisualizer(lines3d,opt.save, None, rx=rx,ry=ry,rz=rz,t=t, points3d_all=None, show_endpoints=opt.show_points,
    line_width=opt.line_width,
    camera_path=opt.cams)

    # opt = visualizer


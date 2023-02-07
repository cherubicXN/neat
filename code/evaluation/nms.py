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

def WireframeVisualizer(lineset, render_dir = None, cam_dir = None):
    import matplotlib.pyplot as plt
    from collections import deque

    if render_dir is not None:
        os.makedirs(render_dir,exist_ok=True)
    
    WireframeVisualizer.view_cnt = 0
    WireframeVisualizer.render_dir = render_dir
    WireframeVisualizer.camera_path = []
    WireframeVisualizer.image_path = []

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
        # view_cnt += 1
        ctr = vis.get_view_control()
        glb = WireframeVisualizer
        param = ctr.convert_to_pinhole_camera_parameters()
        glb.view_cnt +=1
        image = vis.capture_screen_float_buffer()
        image = np.asarray(image)*255
        image = np.asarray(image,dtype=np.uint8)
        if glb.render_dir is not None:
            img_path = os.path.join(glb.render_dir,'image_{:04d}.png'.format(glb.view_cnt))
            cam_path = os.path.join(glb.render_dir,'cam_{:04d}.json'.format(glb.view_cnt))

            glb.image_path.append(img_path)
            glb.camera_path.append(cam_path)

            cv2.imwrite(img_path,image)
            o3d.io.write_pinhole_camera_parameters(cam_path,param)
            print('saving the rendered image into {}'.format(img_path))
            print('saving the rendering viewpoint into {}'.format(cam_path))
        else:
            print('the rendering path is None')
        return False

    key_to_call_back = {}
    key_to_call_back[ord('S')] = capture_image
    key_to_call_back[ord('L')] = load_view

    print(key_to_call_back)
    o3d.visualization.draw_geometries_with_key_callbacks([lineset],key_to_call_back)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str,required=True,help='the path of the reconstructed wireframe model')
    # parser.add_argument('--imgdir', type=None,)
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--load-views', type=str, default=None)

    opt = parser.parse_args()

    if opt.save:
        opt.save = opt.data.rstrip('.npz')+'_record_nms'
    else:
        opt.save = None



    data = np.load(opt.data,allow_pickle=True)

    lines3d = data['lines3d']

    lines3d = np.concatenate(lines3d,axis=0)

    bbox_min = lines3d.reshape(-1,3).min(axis=0)
    bbox_max = lines3d.reshape(-1,3).max(axis=0)

    xx = torch.linspace(bbox_min[0],bbox_max[0], 512)
    yy = torch.linspace(bbox_min[1],bbox_max[1], 512)
    zz = torch.linspace(bbox_min[2],bbox_max[2], 512)

    xm,ym,zm = torch.meshgrid(xx,yy,zz)
    xyz_grid = torch.stack((xm,ym,zm),dim=-1)
    delta_xyz = (bbox_max-bbox_min)/511

    points = lines3d.reshape(-1,3)
    points_long = (points-bbox_min[None])/delta_xyz
    points_long = np.array(points_long.round(),dtype=np.int64)
    points_long = torch.tensor(points_long)

    grid = torch.zeros((512,512,512))
    points_id_uni, points_cnt_uni = points_long.unique(dim=0,return_counts=True)

    grid[points_id_uni[:,0],points_id_uni[:,1],points_id_uni[:,2]] = points_cnt_uni.float()
    max_pool_res = torch.nn.functional.max_pool3d(grid[None],3,padding=1,stride=1)
    temp = (max_pool_res==grid).float()*(max_pool_res>0)
    idx = temp[0].nonzero()

    points_uni = torch.stack(
        (xx[idx[:,0]], yy[idx[:,1]],zz[idx[:,2]]),dim=-1
    )


    dis1 = torch.sum((points_uni[:,None] - lines3d[None,:,0])**2,dim=-1)
    dis2 = torch.sum((points_uni[:,None] - lines3d[None,:,1])**2,dim=-1)
    cost1, idx1 = dis1.min(dim=0)
    cost2, idx2 = dis2.min(dim=0)
    cost = torch.max(cost1,cost2)
    idx_pair = torch.stack((idx1,idx2),dim=-1)
    # idx_pair_valid = idx_pair[cost<delta_xyz.min()**2]
    idx_pair_valid = idx_pair[cost<10000]

    lines_optimized = points_uni[idx_pair_valid]
    lines_opt_set = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points_uni),
        o3d.utility.Vector2iVector(idx_pair_valid))


    WireframeVisualizer(lines_opt_set,opt.save, opt.load_views)

    # opt = visualizer


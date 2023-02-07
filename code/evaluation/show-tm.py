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

def WireframeVisualizer(lines, render_dir = None, cam_dir = None):
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
        lines = glb.lines

        height, width = param.intrinsic.height, param.intrinsic.width
        K = param.intrinsic.intrinsic_matrix
        R = param.extrinsic[:3,:3]
        T = param.extrinsic[:3,3:]

        x = lines.reshape(-1,3)
        x2d = K@(R@x.transpose()+T)
        x2d = x2d[:2]/x2d[2:]
        x2d = x2d.transpose()
        lines2d = x2d.reshape(-1,2,2)

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
            plt.xlim([-0.5, width-0.5])
            plt.ylim([height-0.5, -0.5])
            plt.plot([lines2d[:,0,0],lines2d[:,1,0]],[lines2d[:,0,1],lines2d[:,1,1]],'-',color='black',linewidth=0.05)
            # plt.plot(
            #     [lines_display[:,0],lines_display[:,2]],
            #     [lines_display[:,1],lines_display[:,3]],
            #     'r-',
            #     linewidth = 0.5 if args.saveto else None
            # )
            # plt.scatter(lines2d[:,0,0],lines2d[:,0,1],color='b',s=1.2,edgecolors='none',zorder=5)
            # plt.scatter(lines2d[:,1,0],lines2d[:,1,1],color='b',s=1.2,edgecolors='none',zorder=5)
            # cv2.imwrite(img_path,image)
            plt.savefig(img_path,dpi=600)
            o3d.io.write_pinhole_camera_parameters(cam_path,param)
            print('saving the rendered image into {}'.format(img_path))
            print('saving the rendering viewpoint into {}'.format(cam_path))
        else:
            print('the rendering path is None')
        return False

    def increase_line_width(vis):
        vis.get_render_option().line_width += 1
        vis.poll_events()
        vis.update_renderer()
        print(vis.get_render_option().line_width)
        return False
    def decrease_line_width(vis):
        vis.get_render_option().line_width -= 1
        vis.poll_events()
        vis.update_renderer()
        # vis.update_geometry()
        print(vis.get_render_option().line_width)
        return False

    # def load_render_option(vis):
    #     import pdb; pdb.set_trace()
    # vis = o3d.visualization.VisualizerWithKeyCallback()
    # mat = o3d.visualization.rendering.MaterialRecord() 
    # mat.shader = 'standard'
    # mat.line_width = 2

    # o3d.visualization.draw({
    #     'name': 'lines',
    #     'geometry': lineset,
    #     'material': mat
    # })
    # import pdb; pdb.set_trace()

    vis = WireframeVisualizer.vis
    # vis.create_window(height=512,width=512)
    vis.create_window()
    render_option = vis.get_render_option()
    vis.add_geometry(lineset)
    render_option.line_width = 3
    vis.register_key_callback(ord('S'), capture_image)
    vis.register_key_callback(ord('L'), load_view)
    vis.register_key_callback(ord('I'), increase_line_width)
    vis.register_key_callback(ord('D'), decrease_line_width)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()
    # o3d.visualization.draw_geometries_with_key_callbacks([lineset],key_to_call_back)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str,required=True,help='the path of the reconstructed wireframe model')
    # parser.add_argument('--imgdir', type=None,)
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--load-views', type=str, default=None)

    opt = parser.parse_args()

    if opt.save:
        opt.save = opt.data.rstrip('.npz')+'_record'
    else:
        opt.save = None



    data = np.load(opt.data,allow_pickle=True)

    lines3d = data['lines3d']

    lines3d = np.concatenate(lines3d,axis=0)
    
    # lineset_o3d = linesToOpen3d(lines3d)

    # WireframeVisualizer(lines3d,opt.save, opt.load_views)

    def callback(scene):
        print('hahahah')
    linepath = trimesh.load_path(lines3d)
    scene = trimesh.Scene([linepath])
    import pdb; pdb.set_trace()
    scene.show(callback=callback)

    # opt = visualizer


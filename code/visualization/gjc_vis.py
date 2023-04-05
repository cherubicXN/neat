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
from pathlib import Path
from scipy.optimize import linear_sum_assignment

import taichi as ti

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Camera:
    def __init__(self, window, up):
        self._window = window
        self._camera_pos = np.array((0.4, 0.5, 2.0))
        self._lookat_pos = np.array((0.0, 0.0, 0.0))
        self._up = np_normalize(np.array(up))
        self._last_mouse_pos = None

    @property
    def mouse_exclusive_owner(self):
        return True

    def update_camera(self):
        res = self._update_by_wasd()
        res = self._update_by_mouse() or res
        return res

    def _update_by_mouse(self):
        win = self._window
        if not self.mouse_exclusive_owner or not win.is_pressed(ti.ui.LMB):
            self._last_mouse_pos = None
            return False
        mouse_pos = np.array(win.get_cursor_pos())
        if self._last_mouse_pos is None:
            self._last_mouse_pos = mouse_pos
            return False
        # Makes camera rotation feels right
        dx, dy = self._last_mouse_pos - mouse_pos
        self._last_mouse_pos = mouse_pos

        out_dir = self._lookat_pos - self._camera_pos
        leftdir = self._compute_left_dir(np_normalize(out_dir))

        scale = 3
        rotx = np_rotate_matrix(self._up, dx * scale)
        roty = np_rotate_matrix(leftdir, dy * scale)

        out_dir_homo = np.array(list(out_dir) + [0.0])
        new_out_dir = np.matmul(np.matmul(roty, rotx), out_dir_homo)[:3]
        self._lookat_pos = self._camera_pos + new_out_dir

        return True

    def _update_by_wasd(self):
        win = self._window
        tgtdir = self.target_dir
        leftdir = self._compute_left_dir(tgtdir)
        lut = [
            ('w', tgtdir),
            ('a', leftdir),
            ('s', -tgtdir),
            ('d', -leftdir),
            ('e', [0, -1, 0]),
            ('q', [0, 1, 0]),
        ]
        dir = np.array([0.0, 0.0, 0.0])
        pressed = False
        for key, d in lut:
            if win.is_pressed(key):
                pressed = True
                dir += np.array(d)
        if not pressed:
            return False
        dir *= 0.05
        self._lookat_pos += dir
        self._camera_pos += dir
        return True

    @property
    def position(self):
        return self._camera_pos

    @property
    def look_at(self):
        return self._lookat_pos

    @property
    def target_dir(self):
        return np_normalize(self.look_at - self.position)

    def _compute_left_dir(self, tgtdir):
        cos = np.dot(self._up, tgtdir)
        if abs(cos) > 0.999:
            return np.array([-1.0, 0.0, 0.0])
        return np.cross(self._up, tgtdir)

class Scene:
    def __init__(self):
        ti.init(arch=ti.cuda)
        self.window = ti.ui.Window('GJC', (800, 600),vsync=True)
        self.camera = Camera(self.window, [0, 1, 0])
    
    def render(self, mesh, color):
        self.window.get_canvas().clear(color)
        self.camera.update_camera()
        self.window.set_viewport((0, 0, self.window.res[0], self.window.res[1]))
        self.window.set_camera(self.camera.position, self.camera.look_at, self.camera.up)
        self.window.show(mesh)
def sweep_ckpt(expdir, checkpoint):
    """
    Given an experiment directory and a checkpoint name, find the checkpoint file.
    If the checkpoint name is a timestamp, return the timestamp.
    If the checkpoint name is a name, return the timestamp corresponding to the latest checkpoint with that name.
    """
    expdir = Path(expdir)
    # Find all ckpt files with the same name
    ckpt_candidates = [x for x in expdir.glob('**/ModelParameters/{}.pth'.format(checkpoint))]
    if len(ckpt_candidates)> 1:
        # If there are more than one, raise an error
        msg = ['multiple timestamps containing the checkpoint {}: '.format(checkpoint)] + [x.relative_to(expdir).parts[0] for x in ckpt_candidates]
        msg = '\n\t- '.join(msg)
            
        raise RuntimeError(msg)

    # Get the ckpt file
    if len(ckpt_candidates) == 0:
        raise RuntimeError('No checkpoint matching {} found'.format(checkpoint))
        
    candidate = ckpt_candidates[0]
    # Get the timestamp from the ckpt file
    timestamp = candidate.relative_to(expdir).parts[0]
    
    return timestamp

def visualize_junctions(points3d, ax=None):
    if ax is None:
        fig = plt.figure("SparseMap", figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
        plt.tight_layout()
    ax.clear()
    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], s=1, c='r')
    return ax
def get_global_junctions(**kwargs):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']

    expname = conf.get_string('train.expname') + kwargs['expname']
    scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else conf.get_int('dataset.scan_id', default=-1)
    if scan_id != -1:
        expname = expname + '/{0}'.format(scan_id)

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    timestamp = kwargs['timestamp']
    if timestamp is None:
        timestamp = sweep_ckpt(expdir, kwargs['checkpoint'])

    # evaldir = os.path.join('../', evals_folder_name, expname)
    evaldir = os.path.join(expdir, timestamp )
    # utils.mkdir_ifnotexists(evaldir)
    os.makedirs(evaldir,exist_ok=True)


    conf_model = conf.get_config('model')
    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf_model)
    if torch.cuda.is_available():
        model.cuda()

    junction_dir = os.path.join(expdir, timestamp, 'junctions')

    ckpt_juncs = os.listdir(junction_dir)
    checkpoints = [int(x.split('.')[0]) for x in ckpt_juncs]

    checkpoints.sort()

    global_junctions = []

    model.eval()

    import json

    
    for ckpt in tqdm(checkpoints):
        # print('Loading checkpoint: {}'.format(ckpt))
        checkpoint_path = os.path.join(junction_dir, str(ckpt) + ".pth")

        saved_junctions = torch.load(checkpoint_path)
        global_junctions.append(saved_junctions)
        
    #     model.load_state_dict(saved_model_state['model_state_dict'])
    #     epoch = saved_model_state['epoch']
    #     junctions3d = model.ffn(model.latents).detach()
    #     global_junctions.append(junctions3d)

    global_junctions = torch.stack(global_junctions, dim=0)
    # global_junctions = global_junctions.cpu().numpy()

    # bb_min = np.min(global_junctions, axis=(0,1))
    # bb_max = np.max(global_junctions, axis=(0,1))

    # plt.figure('global junctions')
    # plt.title('global junctions')
    # plt.axis('off')

    # ax = None
    # for i in range(global_junctions.shape[0]):
    # # i = -1
    #     ax = visualize_junctions(global_junctions[i], ax=ax)
    #     plt.pause(1.0)
    # plt.show()
    ti.init(arch=ti.cuda)

    scene = ti.ui.Scene()

    window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(5, 2, 2)
    camera.lookat(0, 0, 0)
    # camera.up(0, 1, 0)
    camera.projection_mode(ti.ui.ProjectionMode(0))


    points = ti.Vector.field(3, dtype=ti.f32, shape=(global_junctions.shape[1],))
    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in points:
            points[i] = [i for j in ti.static(range(3))]
            

    init_points_pos(points)
    import time
    i = 0
    video_manager = ti.tools.VideoManager(os.path.join(expdir, timestamp))

    junctions_interpolated = global_junctions
    # for i in range(global_junctions.shape[0]-1):
    #     tspan = torch.linspace(0, 1,25)
    #     for t in tspan[1:]:
    #         cur_points = global_junctions[i] * (1-t) + global_junctions[i+1] * t
    #         junctions_interpolated.append(cur_points.cpu().numpy())
    # junctions_interpolated = np.stack(junctions_interpolated, axis=0)
    

    # while window.running:
    #     camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.LMB)
    #     scene.set_camera(camera)
        
    #     scene.ambient_light((0.8, 0.8, 0.8))
    #     scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

    #     cur_points = junctions_interpolated[-1]
    #     points.from_numpy(cur_points)
    #     scene.particles(points, color = (0.68, 0.26, 0.19), radius = 0.01)
    #     # time.sleep(0.02)
            
    #     # Draw 3d-lines in the scene

    #     canvas.scene(scene)
    #     camera.lookat(0, 0, 0)
    #     # img = window.get_image_buffer_as_numpy()
    #     # video_manager.write_frame(img)
    #     # i = (i + 1) % junctions_interpolated.shape[0]
    #     window.show()
    i = 0

    rendered_images = []
    while window.running:
        camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.LMB)
        # camera.lookat(0, 0, 0)
        scene.set_camera(camera)
        # import pdb; pdb.set_trace()
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        cur_points = junctions_interpolated[i].numpy()
        points.from_numpy(cur_points)
        scene.particles(points, color = (0.68, 0.26, 0.19), radius = 0.01)
        time.sleep(0.02)
            
        # Draw 3d-lines in the scene

        canvas.scene(scene)
        img = window.get_image_buffer_as_numpy()
        # video_manager.write_frame(img)
        rendered_images.append(img)
        window.show()

        if i < junctions_interpolated.shape[0] - 1:
            i += 1
    # for img in rendered_images:
    #     video_manager.write_frame(img)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--evals_folder', type=str, default='evals', help='The evaluation folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    # parser.add_argument('--timestamp', required=True, type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--timestamp', type=str,required=True, help='The experiemnt timestamp to test.')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')

    opt = parser.parse_args()

    if opt.gpu == 'auto':
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
    
    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)
    get_global_junctions(conf=opt.conf,
        expname=opt.expname,
        exps_folder_name=opt.exps_folder,
        evals_folder_name=opt.evals_folder,
        timestamp=opt.timestamp,
        scan_id=opt.scan_id,
    )

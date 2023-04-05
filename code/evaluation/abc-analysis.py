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
# import utils.plots as plt
import matplotlib.pyplot as plt
from utils import rend_util
from collections import defaultdict
import trimesh
from pathlib import Path
from scipy.optimize import linear_sum_assignment

import hashlib
import base64
import open3d as o3d



def project2D(K,R,T, points3d):
    shape = points3d.shape 
    assert shape[-1] == 3
    X = points3d.reshape(-1,3)
    
    x = K@(R@X.t()+T)
    x = x.t()
    # sign = x[:,-1:]>=0
    # x = x/x[:,-1:]
    denominator = x[:,-1:]
    sign = torch.where(denominator>=0, torch.ones_like(denominator), -torch.ones_like(denominator))
    eps = torch.where(denominator.abs()<1e-8, torch.ones_like(denominator)*1e-8, torch.zeros_like(denominator))
    x = x/(denominator+eps*sign)
    x = x.reshape(*shape)[...,:2]
    return x

def ray_casting_check(scene, points2d, points3d, intrinsics, pose, tol = 1e-4):
    ray_dirs, cam_loc = rend_util.get_camera_params(points2d[None].cuda(), pose.cuda(), intrinsics.cuda())
    ray_dirs = ray_dirs[0].cpu().numpy()
    cam_loc = cam_loc[0].cpu().numpy()
    cam_loc = cam_loc.reshape(1,3).repeat(points2d.shape[0],0)
    o3d_rays = np.concatenate([cam_loc, ray_dirs],1)
    rays = o3d.core.Tensor(o3d_rays, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays)
    t_cast = ans['t_hit'].numpy()
    junctions3d_cast = cam_loc + ray_dirs*t_cast[:,None]
    cast_valid = np.linalg.norm(junctions3d_cast - points3d.numpy(),axis=-1)<tol

    return cast_valid

def wireframe_recon(**kwargs):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    conf_path = kwargs['conf']
    conf = ConfigFactory.parse_file(kwargs['conf'])
    # assert os.path.basename(conf_path) == 'runconf.conf'

    root = os.path.join(*conf_path.split('/')[:-1])

    dataset_conf = conf.get_config('dataset')
    dataset_conf['distance_threshold'] = float(1)

    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)

    eval_dataset.distance = 1

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )

    dataset_root = dataset_conf['data_dir']
    import json
    with open(os.path.join('../data',dataset_root, 'lines.json'), 'r') as f:
        wireframe_gt = json.load(f)
    with open(os.path.join('../data',dataset_root, 'offset_scale.txt'), 'r') as f:
        offset_scale = np.array([float(x) for x in f.read().split()])
    scale_mat = np.array([[1/float(offset_scale[-1]),0,0,-float(offset_scale[0])],
                        [0,1/float(offset_scale[-1]),0,-float(offset_scale[1])],
                        [0,0,1/float(offset_scale[-1]),-float(offset_scale[2])],
                        [0,0,0,1]])
    inv_scale = np.linalg.inv(scale_mat)

    mesh = o3d.io.read_triangle_mesh(os.path.join('../data',dataset_root, 'mesh.obj'))
    mesh = mesh.transform(inv_scale)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    wireframe_junctions = np.array(wireframe_gt['junctions'])
    wireframe_junctions = (inv_scale[:3,:3]@wireframe_junctions.T + inv_scale[:3,3:]).T

    # wireframe_junctions = (wireframe_junctions-offset_scale[:3])*offset_scale[-1]
    wireframe_edges = np.array(wireframe_gt['lines'])
    wireframe_lines = wireframe_junctions[wireframe_edges]
    wireframe_junctions = torch.tensor(wireframe_junctions,dtype=torch.float32)
    wireframe_lines = torch.tensor(wireframe_lines,dtype=torch.float32)

    width, height = eval_dataset.img_res

    hit_count = 0
    hit_count_lines = 0

    wireframe_junctions_hit = torch.zeros_like(wireframe_junctions[:,0])
    wireframe_lines_hit = torch.zeros_like(wireframe_lines[:,0,0])

    for indices, model_input, ground_truth in tqdm(eval_dataloader):
        K = model_input["intrinsics"][0]
        proj_mat = model_input["pose"].inverse()
        R = proj_mat[0, :3, :3]
        t = proj_mat[0, :3, 3:]

        rgb = ground_truth['rgb'].reshape(height,width,-1)
        junctions2d_gt = project2D(K,R,t,wireframe_junctions)
        # plt.imshow(rgb)
        # plt.plot(junctions2d_gt[:,0], junctions2d_gt[:,1], 'r.')
        # plt.show()
        # import pdb; pdb.set_trace()
        is_valid = (junctions2d_gt[:,0]>=0)&(junctions2d_gt[:,0]<width)&(junctions2d_gt[:,1]>=0)&(junctions2d_gt[:,1]<height)
        
        cast_valid = ray_casting_check(scene,junctions2d_gt,wireframe_junctions,model_input['intrinsics'],model_input['pose'])

        is_valid *= cast_valid
        # junctions2d_gt = junctions2d_gt[is_valid]


        wireframe_hawp = model_input['wireframe'][0]
        junctions_pred = wireframe_hawp.vertices
        
        jdist = torch.cdist(junctions_pred, junctions2d_gt)
        assign = linear_sum_assignment(jdist.cpu().numpy())
        matched_dis = jdist[assign[0], assign[1]]
        hitted_junctions = (matched_dis<20)*(is_valid[assign[1]])
        wireframe_junctions_hit[assign[1][hitted_junctions]] +=1
        hitted_rate = hitted_junctions.sum()/is_valid.sum().clamp_min(1)
        hit_count += hitted_rate

        lines2d_gt = project2D(K,R,t,wireframe_lines).reshape(-1,4)
        is_in = (lines2d_gt[:,0]>=0)&(lines2d_gt[:,0]<width)&(lines2d_gt[:,1]>=0)&(lines2d_gt[:,1]<height)&(lines2d_gt[:,2]>=0)&(lines2d_gt[:,2]<width)&(lines2d_gt[:,3]>=0)&(lines2d_gt[:,3]<height)
        cast_valid_a = ray_casting_check(scene,lines2d_gt[:,:2],wireframe_lines[:,0],model_input['intrinsics'],model_input['pose'],tol=0.1)
        cast_valid_b = ray_casting_check(scene,lines2d_gt[:,2:],wireframe_lines[:,1],model_input['intrinsics'],model_input['pose'],tol=0.1)

        is_in *= cast_valid_a*cast_valid_b

        # lines2d_gt = lines2d_gt[is_in]

        lines2d_dt = wireframe_hawp.line_segments(0.05)[:,:-1]

        ldist1 = torch.norm(lines2d_dt[:,None,:2]-lines2d_gt[:,:2],dim=-1) + torch.norm(lines2d_dt[:,None,2:]-lines2d_gt[:,2:],dim=-1)
        ldist2 = torch.norm(lines2d_dt[:,None,:2]-lines2d_gt[:,2:],dim=-1) + torch.norm(lines2d_dt[:,None,2:]-lines2d_gt[:,:2],dim=-1)

        ldist = torch.min(ldist1,ldist2)*0.5

        assign = linear_sum_assignment(ldist.cpu().numpy())
        matched_dis = ldist[assign[0], assign[1]]
        hitted_lines = (matched_dis<20)*(is_in[assign[1]])
        wireframe_lines_hit[assign[1][hitted_lines]] +=1
        hitted_rate = hitted_lines.sum()/is_in.sum().clamp_min(1)
        # hitted_lines = ldist.min(dim=0)[0]<5

        # hitted_rate_lines = hitted_lines.sum()/(hitted_lines.numel()+1e-8)
        hit_count_lines += hitted_rate
        
        # print(hitted_rate)
    # import pdb; pdb.set_trace()
    print(wireframe_junctions.shape[0])
    print(wireframe_lines.shape[0])
    print((wireframe_junctions_hit>0).sum())
    print(hit_count/len(eval_dataloader))
    print(hit_count_lines/len(eval_dataloader))
    print((wireframe_lines_hit>=0).sum())
    np.savez(
        os.path.join('../data',dataset_root,'wireframe_junctions_hit.npz'),
        lines3d=wireframe_lines[wireframe_lines_hit>=0].numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    # parser.add_argument('--timestamp', required=True, type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--chunksize', default=2048, type=int, help='the chunksize for rendering')
    parser.add_argument('--reproj-dis', default=10, type=int, help='the distance threshold of 2D line segments')
    parser.add_argument('--ckdist', default=100, type=float, help='the distance threshold for visibility checking')
    parser.add_argument('--ckview', default=5, type=int, help='the number of views for visibility checking')

    parser.add_argument('--overwrite', default=False, action='store_true', help='overwrite the existing results')
    # parser.add_argument('--score-th', default=0.05, type=float, help='the score threshold of 2D line segments')

    opt = parser.parse_args()

    if opt.gpu == 'auto':
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
    
    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)
    wireframe_recon(conf=opt.conf,
        checkpoint=opt.checkpoint,
        chunksize=opt.chunksize,
        distance=opt.reproj_dis,
        overwrite=opt.overwrite,
        ckdist = opt.ckdist,
        ckview = opt.ckview
    )

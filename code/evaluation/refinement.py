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

import matplotlib.pyplot as plt


def draw_lines(lines, *args, **kwargs):
    if isinstance(lines, torch.Tensor):
        lines = lines.cpu().numpy()
    
    return plt.plot([lines[:,0,0],lines[:,1,0]],[lines[:,0,1],lines[:,1,1]],*args,**kwargs)


def wireframe_recon(**kwargs):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']

    expname = conf.get_string('train.expname') + kwargs['expname']
    scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else conf.get_int('dataset.scan_id', default=-1)
    if scan_id != -1:
        expname = expname + '/{0}'.format(scan_id)

    timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    evaldir = os.path.join('../', evals_folder_name, expname)
    utils.mkdir_ifnotexists(evaldir)

    dataset_conf = conf.get_config('dataset')
    if scan_id != -1:
        dataset_conf['scan_id'] = scan_id
    
    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)


    conf_model = conf.get_config('model')
    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf_model)
    if torch.cuda.is_available():
        model.cuda()

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    checkpoint_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")

    print('Checkpoint: {}'.format(checkpoint_path))
    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))

    model.load_state_dict(saved_model_state['model_state_dict'])
    epoch = saved_model_state['epoch']

    print('evaluating...')

    model.eval()

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )
    chunksize = kwargs['chunksize']

    sdf_threshold = kwargs['sdf_threshold']

    lines3d_all = []

    maskdirs = os.path.join(evaldir,'masks')
    utils.mkdir_ifnotexists(maskdirs)

    data = np.load(kwargs['data'],allow_pickle=True)
    lines3d_init = np.concatenate(data['lines3d'])
    lines3d_init = torch.tensor(lines3d_init, device='cuda')
    points3d_all = [torch.tensor(l) for l in data['points3d_all']]
    num_lines = lines3d_init.shape[0]
    visibility = torch.zeros((num_lines, len(eval_dataloader)),device='cuda')
    
    tspan = torch.linspace(0,1,16,device='cuda').reshape(1,-1,1)
    lines3d_diff = lines3d_init[:,1:] - lines3d_init[:,:1]
    points = lines3d_init[:,:1] + tspan*lines3d_diff
    sdf_vals = model.implicit_network.get_sdf_vals(points.reshape(-1,3)).flatten().reshape(points.shape[:-1]).detach()
    sdf_vals = sdf_vals.abs()
    sdf_vals_max = sdf_vals.max(dim=-1)[0]

    scores = data['scores']

    is_valid = (sdf_vals_max<0.01).cpu().numpy()*(scores<0.01)
    # lines3d_init = lines3d_init[sdf_vals_max<kwargs['sdf_threshold']]
    lines3d_init = lines3d_init[is_valid]
    # for indices, model_input, ground_truth in tqdm(eval_dataloader):    
    #     rgb = ground_truth['rgb'].reshape(*eval_dataset.img_res,3)
    #     wireframe = model_input['wireframe'][0]
    #     lsd = wireframe.line_segments(0.05).numpy()
    #     plt.imshow(rgb)
    #     plt.plot([lsd[:,0],lsd[:,2]],[lsd[:,1],lsd[:,3]],'r-')
    #     import pdb; pdb.set_trace()
    height, width = eval_dataset.img_res
    for epoch in range(1):
        for indices, model_input, ground_truth in tqdm(eval_dataloader):    
            image = ground_truth['rgb'].reshape(height,width,3)
            mask = model_input['mask']
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input['uv'] = model_input['uv'][:,mask[0]]
            # randidx = torch.randperm(model_input['uv'].shape[1])
            # model_input['uv'] = model_input['uv'][:,randidx]
            model_input['pose'] = model_input['pose'].cuda()
            lines = model_input['lines'][0].cuda()
            labels = model_input['labels'][0]

            proj_mat = model_input['pose'][0].inverse()[:3]
            R = proj_mat[:,:3]
            T = proj_mat[:,3:]

            lines2d = model.project2D(model_input['intrinsics'][0,:3,:3],R,T,lines3d_init)


            is_possible = (lines2d[...,0]>=0).all(dim=-1)*(lines2d[...,1]>=0).all(dim=-1)*(lines2d[...,0]<=width).all(dim=-1)*(lines2d[...,1]<=height).all(dim=-1)

            lines2d_gt = model_input['wireframe'][0].line_segments(0.05).cuda()
            lines2d_gt = lines2d_gt[:,:-1].reshape(-1,2,2)

            dis1 = torch.sum((lines2d_gt[None] - lines2d[:,None])**2,dim=-1).sum(dim=-1)
            dis2 = torch.sum((lines2d_gt[None,:,[1,0]] - lines2d[:,None,])**2,dim=-1).sum(dim=-1)

            mindis = torch.min(dis1,dis2)
            mindis, mindix = mindis.min(dim=1)
            mindis1 = dis1[torch.arange(lines3d_init.shape[0]),mindix]
            # mindis2 = dis2[torch.arange(lines3d_init.shape[0]),mindix]
            is_possible = is_possible * (mindis<10)
            if torch.sum(is_possible) == 0:
                continue
            is_reverse = (mindis!=mindis1)*is_possible

            used_idx = is_possible.nonzero().flatten()
            # import pdb; pdb.set_trace()
            points3d_vis = [points3d_all[u] for u in used_idx]
            points3d_flt = torch.cat(points3d_vis).cuda()
            points2d_flt = model.project2D(model_input['intrinsics'][0,:3,:3],R,T,points3d_flt)
            lines3d_wait = lines3d_init[used_idx]
            points3d_wait = []
            lines3d_wait[is_reverse[used_idx]] = lines3d_wait[is_reverse[used_idx]][:,[1,0]]
            minidx_wait = mindix[used_idx]

            match_set = minidx_wait.unique()

            lines3d_updated_cur = []
            used_idx = []
            for it in match_set:
                sel_idx = torch.nonzero((minidx_wait==it)).flatten()
                points3d_sel = [points3d_vis[s] for s in sel_idx]
                # pcd = trimesh.points.PointCloud(torch.cat(points3d_sel))
                # lcd = trimesh.load_path(lines3d_wait[sel_idx].cpu())
                # trimesh.Scene([pcd,lcd]).show()
                # import pdb; pdb.set_trace()
                lines3d_avg = lines3d_wait[sel_idx].mean(dim=0)
                lines3d_updated_cur.append(lines3d_avg)
                used_idx.append(sel_idx)
            lines3d_updated_cur = torch.stack(lines3d_updated_cur,dim=0)
            

            lines_unused = lines3d_init[~is_possible]

            lines3d_init = torch.cat((lines_unused,lines3d_updated_cur))

        # center_sdf, center_features, center_gradients = model.implicit_network.get_outputs(0.5*(lines3d_init[:,0]+lines3d_init[:,1]))

        
    # temp = lines3d_init[center_sdf.flatten().abs()<1e-3]
    
    # trimesh.load_path(lines3d_init.cpu()).show()
    # import pdb; pdb.set_trace()
        
    lines3d_all = lines3d_init.cpu().numpy()

    wireframe_dir = os.path.join(evaldir,'wireframes')
    utils.mkdir_ifnotexists(wireframe_dir)

    line_path = os.path.join(wireframe_dir,'{}-ref.npz'.format(kwargs['checkpoint']))
    print(line_path)
    np.savez(line_path,lines3d=lines3d_all)

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--evals_folder', type=str, default='exps', help='The evaluation folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--timestamp', required=True, type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--chunksize', default=2048, type=int, help='the chunksize for rendering')
    parser.add_argument('--sdf-threshold', default=1e-3, type=float, help='the sdf threshold')
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--preview', default=0, type=int )

    opt = parser.parse_args()

    if opt.gpu == 'auto':
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
    
    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)
    wireframe_recon(conf=opt.conf,
        expname=opt.expname,
        exps_folder_name=opt.exps_folder,
        evals_folder_name=opt.evals_folder,
        timestamp=opt.timestamp,
        checkpoint=opt.checkpoint,
        scan_id=opt.scan_id,
        resolution=opt.resolution,
        chunksize=opt.chunksize,
        sdf_threshold=opt.sdf_threshold,
        preview = opt.preview,
        data = opt.data,
    )

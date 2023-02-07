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
def wireframe_recon(**kwargs):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']

    expname = conf.get_string('train.expname') + kwargs['expname']
    scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else conf.get_int('dataset.scan_id', default=-1)
    if scan_id != -1:
        expname = expname + '_{0}'.format(scan_id)

    timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    evaldir = os.path.join('../', evals_folder_name, expname)
    utils.mkdir_ifnotexists(evaldir)

    dataset_conf = conf.get_config('dataset')
    dataset_conf['distance_threshold'] = 1.0
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

    eval_dataset.distance = 1
    eval_dataset.score_threshold = 0.05
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
    
    lines3d_init = np.load(kwargs['data'],allow_pickle=True)['lines3d']

    if lines3d_init.dtype == object:
        lines3d_init = np.concatenate(lines3d_init)
    lines3d_init = torch.tensor(lines3d_init).cuda()
    
    is_keep = torch.ones(lines3d_init.shape[0],device='cuda')
    counts = torch.zeros(lines3d_init.shape[0],device='cuda')
    scores = torch.zeros(lines3d_init.shape[0],device='cuda')

    for indices, model_input, ground_truth in tqdm(eval_dataloader):    
        mask = model_input['mask']
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input['uv'] = model_input['uv'][:,mask[0]]
        model_input['pose'] = model_input['pose'].cuda()

        proj = model_input['pose'][0].inverse()

        K = model_input["intrinsics"][0,:3,:3]
        R = proj[:3,:3]
        T = proj[:3,3:]

        lines2d_gt = model_input['lines_uniq'][0].cuda()[:,:-1]
        scors_gt = model_input['lines_uniq'][0].cuda()[:,-1]

        lines2d_init = model.project2D(K,R,T,lines3d_init).reshape(-1,4)

        dis1 = torch.sum((lines2d_gt[:,None]-lines2d_init[None,:])**2,dim=-1)
        dis2 = torch.sum((lines2d_gt[:,None]-lines2d_init[None,:,[2,3,0,1]])**2,dim=-1)

        dis = torch.min(dis1,dis2)

        match_cost, match_idx = dis.min(dim=0)

        is_available = match_cost<10

        label_set = match_idx[is_available].unique()

        lines_fused = []
        for i, label in enumerate(label_set):
            cur = is_available*(match_idx==label)
            scores[cur] += scors_gt[i]
            counts[cur] += 1
            # weight = torch.softmax(-match_cost[cur],dim=0)
            # is_keep[cur] = False

            # lines_ref = torch.sum(lines3d_init[cur]*weight[:,None,None],dim=0)
            # lines_fused.append(lines_ref)
        # lines_fused = torch.stack(lines_fused)

        # lines3d_init = torch.cat([lines_fused,lines3d_init[is_keep>0]])
    scores = scores/counts.clamp(min=1)
   
    lines3d_all = lines3d_init[scores>0.5].cpu().numpy()

    wireframe_dir = os.path.join(evaldir,'wireframes')
    utils.mkdir_ifnotexists(wireframe_dir)

    basename = os.path.basename(kwargs['data'])[:-4]
    line_path = os.path.join(wireframe_dir,'{}-fused.npz'.format(basename))

    np.savez(line_path,lines3d=lines3d_all)
    print('save the reconstructed wireframes to {}'.format(line_path))
    print('python evaluation/show.py --data {}'.format(line_path))

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--evals_folder', type=str, default='evals', help='The evaluation folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--timestamp', required=True, type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--chunksize', default=2048, type=int, help='the chunksize for rendering')
    parser.add_argument('--sdf-threshold', default=0.25, type=float, help='the sdf threshold')
    parser.add_argument('--preview', default=0, type=int )
    parser.add_argument('--data', type=str, required=True)

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
        data = opt.data
    )

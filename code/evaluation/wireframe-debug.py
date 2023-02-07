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
# import utils.plots as 
import matplotlib.pyplot as plt
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

    eval_dataset.distance = 5
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
    
    for indices, model_input, ground_truth in tqdm(eval_dataloader):    
        mask = model_input['mask']
        uv_all = model_input['uv'][0].cuda()
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        # model_input["uv"] = model_input["uv"].cuda()
        # model_input['uv'] = model_input['uv'][:,mask[0]]
        # randidx = torch.randperm(model_input['uv'].shape[1])
        # model_input['uv'] = model_input['uv'][:,randidx]
        model_input['pose'] = model_input['pose'].cuda()

        lines = model_input['lines'][0].cuda()
        labels = model_input['labels'][0]

        label_set = labels[mask[0]].unique()
        rgb = ground_truth['rgb'].reshape(*eval_dataset.img_res,-1)

        lines3d_by_view = []
        for label_item in label_set:
            label_indices = torch.nonzero(mask[0]*(labels==label_item)).flatten()

            model_input['uv'] = uv_all[label_indices][None]

            out = model(model_input)
            lines2d_pred = out['lines2d'].detach().reshape(-1,4)
            lines2d_gt = lines[label_indices]
            lines3d_pred = out['lines3d'].detach()
            
            dis1 = torch.sum((lines2d_pred-lines2d_gt[:,[0,1,2,3]])**2,axis=-1)
            dis2 = torch.sum((lines2d_pred-lines2d_gt[:,[2,3,0,1]])**2,axis=-1)

            line_dis = torch.minimum(
                dis1,dis2
            )
            is_learned = line_dis<10
            is_swap = dis1>dis2
            if is_swap.sum()>0:
                lines3d_pred[is_swap] = lines3d_pred[is_swap][:,[1,0]]
            if is_learned.sum() == 0:
                continue
            
            lines3d_valid = lines3d_pred[is_learned]
            weights = line_dis[is_learned]
            weights = torch.softmax(-weights,dim=0)

            lines3d_mean = torch.sum(lines3d_valid*weights[:,None,None],dim=0)
            lines3d_by_view.append(lines3d_mean)

        if len(lines3d_by_view) == 0:
            continue
        lines3d_all.append(torch.stack(lines3d_by_view))

        # trimesh.load_path(torch.cat(lines3d_all).cpu()).show()
    import pdb; pdb.set_trace()
        
    
   


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
        preview = opt.preview
    )

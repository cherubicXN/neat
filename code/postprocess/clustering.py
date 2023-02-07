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

def sweep_ckpt(expdir, key):
    timestamps = os.listdir(expdir)
    best_t = None
    max_epochs = 0
    for t in timestamps:
        checkpoints = os.listdir(os.path.join(expdir,t,'checkpoints','ModelParameters'))
        is_in = any(key in ckpt for ckpt in checkpoints)
        
        epochs = [c[:-4] for c in checkpoints]
        epochs = [int(c) for c in epochs if c.isdigit()]
        max_ep = max(epochs)
        if max_ep > max_epochs:
            max_epochs = max_ep
            best_t = t
    return best_t
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

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    timestamp = kwargs['timestamp']
    if timestamp is None:
        timestamp = sweep_ckpt(expdir, kwargs['checkpoint'])

    # evaldir = os.path.join('../', evals_folder_name, expname)
    evaldir = os.path.join(expdir, timestamp )
    # utils.mkdir_ifnotexists(evaldir)
    os.makedirs(evaldir,exist_ok=True)

    dataset_conf = conf.get_config('dataset')
    dataset_conf['distance_threshold'] = float(kwargs['distance'])
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

    eval_dataset.distance = kwargs.get('distance',1)
    eval_dataset.score_threshold = kwargs.get('score',0.05)

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )

    wireframe_dir = os.path.join(evaldir,'wireframes')
    utils.mkdir_ifnotexists(wireframe_dir)

    line_path = os.path.join(wireframe_dir,'{}-d={}-s={}.npz'.format(kwargs['checkpoint'],kwargs['distance'],kwargs['score']))

    data = np.load(line_path,allow_pickle=True)

    lines3d = np.concatenate(data['lines3d'],axis=0)
    scores = data['scores']

    lines3d = lines3d[scores<0.01]

    juncs3d = lines3d.reshape(-1,3)
    juncs3d = torch.from_numpy(juncs3d).cuda()
    is_unused = torch.ones(juncs3d.shape[0],dtype=torch.bool)
    
    import open3d as o3d
    for indices, model_input, ground_truth in tqdm(eval_dataloader):    
        mask = model_input['mask']
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input['uv'] = model_input['uv'][:,mask[0]]
        model_input["uv_proj"] = model_input['uv_proj'][:,mask[0]]
        model_input['lines'] = model_input['lines'].cuda()
        # randidx = torch.randperm(model_input['uv'].shape[1])
        # model_input['uv'] = model_input['uv'][:,randidx]
        model_input['pose'] = model_input['pose'].cuda()
        
        juncs2d_gt = model_input['wireframe'][0].vertices.cuda()

        proj_mat = model_input['pose'][0].inverse()[:3]
        K = model_input["intrinsics"][0,:3,:3]
        R = proj_mat[:,:3]
        T = proj_mat[:,3:]

        juncs3d_to_2d = model.project2D(K,R,T,juncs3d)
        
        cdist = torch.sum((juncs3d_to_2d[:,None]-juncs2d_gt[None])**2,dim=-1)
        
        cost, assign = cdist.min(dim=1)
        is_near = cost<3

        idx_list = []
        averaged_juncs = []
        for label in assign.unique():
            is_cur = is_near*(assign==label)
            if is_cur.sum() == 0:
                continue
            j3d = juncs3d[is_cur]
            j3d_sdf = model.implicit_network.get_sdf_vals(j3d).abs()
            # j3d_score = torch.exp(-j3d_sdf).detach()
            # import pdb; pdb.set_trace()
            # j3d = torch.sum(j3d*j3d_score,dim=0)/torch.sum(j3d_score)
            j3d = j3d[j3d_sdf.flatten().argmin()]
            # import pdb; pdb.set_trace()
            # j3d = j3d.mean(dim=0)
            averaged_juncs.append(j3d)
            idx_list.append(is_cur.nonzero().flatten())
            # juncs3d[is_cur] = j3d
            
            # is_unused[is_cur] = False

        juncs3d = torch.cat((juncs3d[~is_near],torch.stack(averaged_juncs)))
        # o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.
    lines3d = torch.from_numpy(lines3d).cuda()

    cost1 = torch.sum((juncs3d[:,None]-lines3d[None,:,0])**2,dim=-1)
    cost2 = torch.sum((juncs3d[:,None]-lines3d[None,:,1])**2,dim=-1)

    idx1 = cost1.min(dim=0)[1]
    idx2 = cost2.min(dim=0)[1]

    idx_min = torch.min(idx1,idx2)
    idx_max = torch.max(idx1,idx2)

    idx_pair = torch.stack((idx_min,idx_max),dim=1)
    idx_pair = idx_pair.unique(dim=0)

    lines_adj = juncs3d[idx_pair]

    lines3d_all = np.array([lines_adj.cpu().numpy()],dtype=object)
    np.savez('temp.npz',lines3d=lines3d_all)
    import pdb; pdb.set_trace()
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--evals_folder', type=str, default='evals', help='The evaluation folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    # parser.add_argument('--timestamp', required=True, type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--timestamp', default=None, type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--chunksize', default=2048, type=int, help='the chunksize for rendering')
    parser.add_argument('--dis-th', default=1, type=int, help='the distance threshold of 2D line segments')
    parser.add_argument('--score-th', default=0.05, type=float, help='the score threshold of 2D line segments')
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
        distance=opt.dis_th,
        score=opt.score_th,
        preview = opt.preview
    )

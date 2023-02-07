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

def project_point_to_line(line_segs, points):
    # Compute the 1D coordinate of the points projected on the line
    dir_vec = (line_segs[:, 1] - line_segs[:, 0])[:, None]
    coords1d = (((points[None] - line_segs[:, None, 0]) * dir_vec).sum(axis=2)
                / np.linalg.norm(dir_vec, axis=2) ** 2)
    # coords1d is of shape (n_lines, n_points)
    
    # Compute the orthogonal distance of the points to each line
    projection = line_segs[:, None, 0] + coords1d[:, :, None] * dir_vec
    dist_to_line = np.linalg.norm(projection - points[None], axis=2)

    return coords1d, dist_to_line


# Given a list of segments parameterized by the 1D coordinate of the endpoints
# compute the overlap with the segment [0, 1]
def get_segment_overlap(seg_coord1d):
    seg_coord1d = np.sort(seg_coord1d, axis=-1)
    overlap = ((seg_coord1d[..., 1] > 0) * (seg_coord1d[..., 0] < 1)
               * (np.minimum(seg_coord1d[..., 1], 1)
                  - np.maximum(seg_coord1d[..., 0], 0)))
    return overlap


# Compute the symmetrical orthogonal line distance between two sets of lines
# and the average overlapping ratio of both lines.
# Enforce a high line distance for small overlaps.
# This is compatible for nD objects (e.g. both lines in 2D or 3D).
def get_sAP_line_distance(warped_ref_line_seg, target_line_seg):
    dist = (((warped_ref_line_seg[:, None, :, None]
              - target_line_seg[:, None]) ** 2).sum(-1)) ** 0.5
    dist = np.minimum(
        dist[:, :, 0, 0] + dist[:, :, 1, 1],
        dist[:, :, 0, 1] + dist[:, :, 1, 0]
    )
    return dist
def get_overlap_orth_line_dist(line_seg1, line_seg2, min_overlap=0.5):
    n_lines1, n_lines2 = len(line_seg1), len(line_seg2)

    # Compute the average orthogonal line distance
    coords_2_on_1, line_dists2 = project_point_to_line(
        line_seg1, line_seg2.reshape(n_lines2 * 2, -1))
    line_dists2 = line_dists2.reshape(n_lines1, n_lines2, 2).sum(axis=2)
    coords_1_on_2, line_dists1 = project_point_to_line(
        line_seg2, line_seg1.reshape(n_lines1 * 2, -1))
    line_dists1 = line_dists1.reshape(n_lines2, n_lines1, 2).sum(axis=2)
    line_dists = (line_dists2 + line_dists1.T) / 2

    # Compute the average overlapping ratio
    coords_2_on_1 = coords_2_on_1.reshape(n_lines1, n_lines2, 2)
    overlaps1 = get_segment_overlap(coords_2_on_1)
    coords_1_on_2 = coords_1_on_2.reshape(n_lines2, n_lines1, 2)
    overlaps2 = get_segment_overlap(coords_1_on_2).T
    overlaps = (overlaps1 + overlaps2) / 2

    # Enforce a max line distance for line segments with small overlap
    low_overlaps = overlaps < min_overlap
    line_dists[low_overlaps] = np.amax(line_dists)
    return line_dists

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

    line_path = kwargs['data'][:-4]+'-group.npz'
    # line_path = os.path.join(wireframe_dir,'{}-d={}-s={}-fuse.npz'.format(kwargs['checkpoint'],kwargs['distance'],kwargs['score']))

    chunksize = kwargs['chunksize']

    sdf_threshold = kwargs['sdf_threshold']

    # data = np.load(kwargs['data'],allow_pickle=True)
    data = torch.load(kwargs['data'])
    
    device = 'cuda'
    
    lines3d_all = []
    lines3d_id = []
    for key, lines3d in data.items():
        lines3d_all.append(lines3d)
        id = torch.tensor([key]*lines3d.shape[0])
        lines3d_id.append(id)
    lines3d_all = torch.cat(lines3d_all).to(device)
    lines3d_id = torch.cat(lines3d_id).to(device)
    visibility = torch.zeros((lines3d_all.shape[0],),device=device)
    visibility_score = torch.zeros((len(eval_dataloader),lines3d_all.shape[0]), device=device)
    visibility_labels = -torch.ones((len(eval_dataloader),lines3d_all.shape[0]),device=device, dtype=torch.long)
    for indices, model_input, ground_truth in tqdm(eval_dataloader):    
        lines2d_gt = model_input['wireframe'][0].line_segments(0.05)
        mask = model_input['mask']
        model_input["intrinsics"] = model_input["intrinsics"].to(device=device)#.cuda()
        model_input['pose'] = model_input['pose'].to(device=device)

        K = model_input["intrinsics"][0,:3,:3]
        proj_mat = model_input['pose'][0].inverse()[:3]
        R = proj_mat[:,:3]
        T = proj_mat[:,3:]

        lines2d_all = model.project2D(K,R,T,lines3d_all).reshape(-1,4)
        lines2d_gt = lines2d_gt.to(device=device)
        # dis = get_overlap_orth_line_dist(lines2d_all.reshape(-1,2,2).cpu().numpy(),lines2d_gt[:,:-1].reshape(-1,2,2).cpu().numpy())

        # import pdb; pdb.set_trace()
        dis1 = torch.sum((lines2d_all[:,None]-lines2d_gt[None,:,[0,1,2,3]])**2,dim=-1)
        dis2 = torch.sum((lines2d_all[:,None]-lines2d_gt[None,:,[2,3,0,1]])**2,dim=-1)
        dis = torch.min(dis1,dis2)
        # dis = torch.tensor(dis,device='cuda')
        mindis, minidx = dis.min(dim=1)
        visibility[mindis<5] += 1
        visibility_score[indices.item()] = mindis
        visibility_labels[indices.item()] = minidx
    
    visibility_dict = {}

    lines3d_final = []
    for id in lines3d_id.unique():
        if id == -1:
            continue
        idx = (id==lines3d_id).flatten()
        visibility_dict[id] = {
            'nvis': visibility[idx],
            'score': visibility_score[:,idx],
            'labels': visibility_labels[:,idx],
            'lines3d': lines3d_all[idx]
        }
        nvis = visibility_dict[id]['nvis']
        scores = visibility_dict[id]['score']
        labels = visibility_dict[id]['labels']
        lines3d = visibility_dict[id]['lines3d']

        visib_views = (scores<5).sum(dim=1)

        visib_count = (scores<5).sum(dim=0)
        line3d_most = lines3d[visib_count.argmax()]
        if visib_count.max() >1:
            lines3d_final.append(line3d_most)
        # for vid in range(scores.shape[0]):
        #     label = labels[vid]
        #     score = scores[vid]
        #     import pdb; pdb.set_trace()
    lines3d_final = torch.stack(lines3d_final,dim=0)

    # np.savez(line_path,lines3d=lines3d_all.cpu().numpy(),scores=scores_all.cpu().numpy())#,cameras=cameras,scores=scores_all,points3d_all=points3d_all)
    np.savez(line_path,lines3d=lines3d_final.cpu().numpy())
    print('save the reconstructed wireframes to {}'.format(line_path))
    print('python evaluation/show.py --data {}'.format(line_path))

    # num_lines = sum([l.shape[0] for l in lines3d_all])
    # print('Number of Total Lines: {num_lines}'.format(num_lines=num_lines))
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
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
        data=opt.data,
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

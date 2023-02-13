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



# def wireframe_recon(**kwargs):
def wireframe_recon(conf, 
                    expname, 
                    exps_folder_name, 
                    evals_folder_name, 
                    timestamp, 
                    checkpoint, 
                    scan_id, 
                    **kwargs):
    """
        wireframe_recon(conf=opt.conf,
        expname=opt.expname,
        exps_folder_name=opt.exps_folder,
        evals_folder_name=opt.evals_folder,
        timestamp=opt.timestamp,
        checkpoint=opt.checkpoint,
        scan_id=opt.scan_id,
        chunksize=opt.chunksize,
        distance=opt.dis_th,
        score=opt.score_th,
    )

    """
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    # conf = ConfigFactory.parse_file(kwargs['conf'])
    conf = ConfigFactory.parse_file(conf)
    # exps_folder_name = kwargs['exps_folder_name']
    # evals_folder_name = kwargs['evals_folder_name']

    expname = conf.get_string('train.expname') + expname
    if scan_id == -1:
        scan_id = conf.get_int('dataset.scan_id', default=-1)
    if scan_id != -1:
        expname = expname + '/{0}'.format(scan_id)

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    # timestamp = kwargs['timestamp']
    if timestamp is None:
        timestamp = sweep_ckpt(expdir,checkpoint)

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
    checkpoint_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(checkpoint) + ".pth")

    print('Checkpoint: {}'.format(checkpoint_path))
    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', checkpoint + ".pth"))

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

    line_path = os.path.join(wireframe_dir,'{}-v2-noref.npz'.format(checkpoint))

    chunksize = kwargs['chunksize']

    # sdf_threshold = kwargs['sdf_threshold']

    lines3d_all = []

    maskdirs = os.path.join(evaldir,'masks')
    utils.mkdir_ifnotexists(maskdirs)
    
    points3d_all = []
    scores_all = []
    
    global_junctions = model.ffn(model.latents).detach()
    gjc_dict = defaultdict(list)
    # trimesh.points.PointCloud(global_junctions.cpu().numpy()).show()

    glj_sdf, glj_feats, glj_grad = model.implicit_network.get_outputs(global_junctions)
    
    global_junctions = (global_junctions - glj_sdf*glj_grad).detach()
    # global_junctions = global_junctions[glj_sdf[:,0].abs()<0.01].detach()
    # ix, iy = torch.triu_indices(graph.shape[0], graph.shape[1], offset=1)
    # trimesh.points.PointCloud(global_junctions.cpu().numpy()).show()

    
    global_junctions_vis = torch.zeros((global_junctions.shape[0]))
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
        K = model_input["intrinsics"][0,:3,:3]
        proj_mat = model_input['pose'][0].inverse()[:3]
        R = proj_mat[:,:3]
        T = proj_mat[:,3:]

        junctions2d_gt = model_input['wireframe'][0].vertices.cuda()
        lines2d_gt = model_input['wireframe'][0].line_segments(0.05).cuda()[:,:-1]
        lines2d_gt = torch.cat((lines2d_gt,lines2d_gt[:,[2,3,0,1]]),dim=0)
        junctions2d = model.project2D(K,R,T,global_junctions)
        is_inside = (junctions2d[:,0]>=0)*(junctions2d[:,0]<eval_dataset.img_res[1])*(junctions2d[:,1]>=0)*(junctions2d[:,1]<eval_dataset.img_res[0])

        jcost = torch.cdist(junctions2d_gt,junctions2d[is_inside])

        jassign = linear_sum_assignment(jcost.cpu().numpy())
        jcost_assign = jcost[jassign[0],jassign[1]]
        source_idx = jassign[0][jcost_assign.cpu().numpy()<10]
        target_idx = is_inside.nonzero().flatten()[jassign[1]][jcost_assign.cpu().numpy()<10]
        global_junctions_vis[target_idx] += 1

    global_junctions = global_junctions[global_junctions_vis>0]
    ix, iy = torch.triu_indices(global_junctions.shape[0], global_junctions.shape[0], offset=1)
    graph = torch.zeros((global_junctions.shape[0],global_junctions.shape[0])).cuda()

    enumerated_indices = torch.stack((ix,iy),dim=-1)
    
    for split in tqdm(torch.split(enumerated_indices,1024)):
        lines3d = global_junctions[split]
        tspan = torch.linspace(0,1,32).reshape(1,-1,1).cuda()
        points3d = lines3d[:,:1]*tspan + lines3d[:,1:]*(1-tspan)
        points3d_flt = points3d.reshape(-1,3)

        torch.cuda.empty_cache()
        sdf_, feats_, grads_ = model.implicit_network.get_outputs(points3d_flt)
        sdf = sdf_.reshape(-1,32).detach()
        is_small = sdf.abs()<0.01
        cnt = is_small.sum(dim=-1)/32
        graph[split[:,0],split[:,1]] += (cnt>0.9)
        

    lines3d_wf = global_junctions[(graph.triu()>0).nonzero()]
    np.savez(line_path,lines3d=lines3d_wf.cpu().numpy())#scores=scores_all,cameras=cameras),#scores=scores_all,points3d_all=points3d_all)
    print('save the reconstructed wireframes to {}'.format(line_path))
    print('python evaluation/show.py --data {}'.format(line_path))

    # num_lines = sum([l.shape[0] for l in lines3d_all])
    # print('Number of Total Lines: {num_lines}'.format(num_lines=num_lines))
    

    # lines3d_all = np.concatenate(lines3d_all,axis=0)
    # lines3d_all = torch.stac

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--evals_folder', type=str, default='evals', help='The evaluation folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--timestamp', default=None, type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--chunksize', default=2048, type=int, help='the chunksize for rendering')
    parser.add_argument('--dis-th', default=1, type=int, help='the distance threshold of 2D line segments')
    parser.add_argument('--score-th', default=0.05, type=float, help='the score threshold of 2D line segments')

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
        chunksize=opt.chunksize,
        distance=opt.dis_th,
        score=opt.score_th,
    )

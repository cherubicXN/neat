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

def make_hash_sha256(o):
    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(o)).encode())
    return base64.b64encode(hasher.digest()).decode()

def make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k,make_hashable(v)) for k,v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))

    return o

@torch.no_grad()
def plt_lines(lines,*args, **kwargs):
    if isinstance(lines, torch.Tensor):
        lines = lines.cpu().numpy()
    return plt.plot([lines[:,0],lines[:,2]],[lines[:,1],lines[:,3]],*args, **kwargs)

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

def get_wireframe_from_lines_and_junctions(lines, junctions, *, 
                    rel_matching_distance_threshold = 0.01,
                    ):
    device = lines.device
    ep1, ep2 = lines[:, 0], lines[:, 1]
    cost1 = torch.cdist(ep1, junctions)
    cost2 = torch.cdist(ep2, junctions)
    mcost1, midx1 = cost1.min(dim=1)
    mcost2, midx2 = cost2.min(dim=1)
    is_matched = torch.max(mcost1,mcost2)<torch.norm(ep1-ep2,dim=-1)
    
    if rel_matching_distance_threshold>0:
        is_matched *= is_matched < rel_matching_distance_threshold

    graph = torch.zeros((junctions.shape[0], junctions.shape[0]), device=device)

    if is_matched.sum()>0:
        pair = torch.stack([torch.min(midx1,midx2),torch.max(midx1,midx2)],dim=1)[is_matched]
        graph[pair[:,0],pair[:,1]] = 1
        graph[pair[:,1],pair[:,0]] = 1

    lines3d_wf = junctions[graph.triu().nonzero()]
    return graph, lines3d_wf


def initial_recon(model, eval_dataloader, chunksize, *, 
                  line_dis_threshold = 10,
                  line_score_threshold = 0.01,
                  junc_match_threshold = 0.05,
                  **kwargs,
                ):

    DEBUG = kwargs.get('DEBUG', False)
    model.eval()

    lines3d_all = []
    points3d_all = []
    scores_all = []

    global_junctions = model.ffn(model.latents).detach()
    if kwargs.get('sdf_junction_refine',True):
        glj_sdf, glj_feats, glj_grad = model.implicit_network.get_outputs(global_junctions)
        is_valid = glj_sdf.abs()<0.05
        # glj_sdf = torch.where(is_valid, glj_sdf, torch.zeros_like(glj_sdf))
        global_junctions = (global_junctions - glj_sdf*glj_grad).detach()

        glj_sdf = model.implicit_network.get_sdf_vals(global_junctions).flatten()
        argsort = torch.argsort(glj_sdf)
        global_junctions = global_junctions[argsort]
        glj_sdf = glj_sdf[argsort]
        is_valid = glj_sdf.abs()<0.05
    
    # dist = torch.eye(global_junctions.shape[0], device=global_junctions.device) + torch.norm(global_junctions[:,None]-global_junctions[None],dim=-1)
    # is_valid = torch.ones_like(is_valid, dtype=torch.bool)
    # for i in range(global_junctions.shape[0]):
    #     if is_valid[i] == 0:
    #         continue
    #     if dist[i].min()<=0.02:
    #         idx = (dist[i]<=0.02).nonzero().flatten()
    #         is_valid[idx] = 0
    
    # global_junctions = global_junctions[is_valid]
    

    gjc_dict = defaultdict(list)
    # eval_dataloader.dataset.wireframes
    trimesh.points.PointCloud(global_junctions[is_valid].cpu().numpy()).show()
    graph = torch.zeros((global_junctions.shape[0], global_junctions.shape[0]), device=global_junctions.device)
    for indices, model_input, ground_truth in tqdm(eval_dataloader):
        if DEBUG and indices.item()>5:
            break
        mask = model_input['mask']
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input['uv'] = model_input['uv'][:,mask[0]]
        model_input["uv_proj"] = model_input['uv_proj'][:,mask[0]]
        model_input['lines'] = model_input['lines'].cuda()
        model_input['pose'] = model_input['pose'].cuda()

        lines = model_input['lines'][0].cuda()
        labels = model_input['labels'][0]
        split = utils.split_input(model_input, mask.sum().item(), n_pixels=chunksize,keys=['uv','uv_proj','lines'])
        split_label = torch.split(labels[mask[0]],chunksize)
        split_lines = torch.split(lines[mask[0]],chunksize)
        lines3d = []
        lines2d = []
        points3d = []
        for s, lb, lines_gt in zip(split,split_label,split_lines):
            torch.cuda.empty_cache()
            out = model(s)
            lines3d_ = out['lines3d'].detach()
            lines2d_ = out['lines2d'].detach().reshape(-1,4)
            
            lines3d.append(lines3d_)
            lines2d.append(lines2d_)
            
            points3d_ = out['l3d'].detach()
            points3d.append(points3d_)

            dis1 = torch.sum((lines2d_-lines_gt[:,:-1])**2,dim=-1)
            dis2 = torch.sum((lines2d_-lines_gt[:,[2,3,0,1]])**2,dim=-1)
            dis = torch.min(dis1,dis2)
            # trimesh.load_path(lines3d_[dis<10].cpu()).show()
            # print(dis.min().item())

        
        lines3d = torch.cat(lines3d)
        lines3d = torch.cat((lines3d,lines3d[:,[1,0]]),dim=0)
        lines2d = torch.cat(lines2d,dim=0)
        lines2d = torch.cat((lines2d,lines2d[:,[2,3,0,1]]),dim=0)
        if len(points3d)>0:
            points3d = torch.cat(points3d,dim=0)
            points3d = torch.cat([points3d,points3d])

        gt_lines = model_input['wireframe'][0].line_segments(0.01).cuda()[:,:-1]

        dis = torch.sum((lines2d[:,None]-gt_lines[None])**2,dim=-1)
        # import pdb; pdb.set_trace()

        mindis, minidx = dis.min(dim=1)

        labels = minidx[mindis<line_dis_threshold].unique()
        lines3d_valid = lines3d[mindis<line_dis_threshold]
        points3d_valid = points3d[mindis<line_dis_threshold]
        assignment = minidx[mindis<line_dis_threshold]
        lines3d = []
        points3d = []
        scores = []
        # import pdb; pdb.set_trace()
        for label in labels:
            idx = (assignment==label).nonzero().flatten()
            if idx.numel()==0:
                continue
            val = lines3d_valid[idx].mean(dim=0)
            lines3d.append(val)
            support_pts = points3d_valid[idx]
            support_dis = torch.norm(torch.cross(support_pts-val[:1],support_pts-val[1:]),dim=-1)/torch.norm(val[1]-val[0]).clamp_min(1e-6)
            points3d.append(
                support_pts[torch.randperm(support_pts.shape[0])[0]]
                )
            scores.append(support_dis.mean())
        
        if len(lines3d)>0:
            lines3d = torch.stack(lines3d,dim=0)
            points3d = torch.stack(points3d,dim=0)
            scores = torch.tensor(scores)
            endpoints = lines3d.reshape(-1,3)
            cdist = torch.cdist(global_junctions,endpoints)
            assign = linear_sum_assignment(cdist.cpu().numpy())
            for ai, aj in zip(*assign):
                if cdist[ai,aj]<junc_match_threshold:
                    gjc_dict[ai].append(endpoints[aj])
            
        # weight = torch.where(dis<100,torch.exp(-dis),torch.zeros_like(-dis))
        # weight = torch.nn.functional.normalize(weight,p=1,dim=0)
            points3d_all.append(points3d.cpu())
            lines3d_all.append(lines3d.cpu())
            scores_all.append(scores.cpu())
            print(len(gjc_dict.keys()),'<--',sum(l.shape[0] for l in lines3d_all))
            # graph_, lines3d_ajd = get_wireframe_from_lines_and_junctions(lines3d, global_junctions, rel_matching_distance_threshold=0.3)
            # graph += graph_
            
            # trimesh.load_path(global_junctions[graph.triu().nonzero()].cpu()).show()
            # import pdb; pdb.set_trace()

    
    lines3d_all = torch.cat(lines3d_all,dim=0)
    scores_all = torch.cat(scores_all,dim=0)
    lines3d_all = lines3d_all[scores_all<0.01]

    for key in gjc_dict.keys():
        gjc_dict[key] = torch.stack(gjc_dict[key],dim=0)

    def junction_dict_to_keys(jdict, threshold = 1):
        key_list = []
        for k,v in jdict.items():
            if v.shape[0]>threshold:
                key_list.append(k)
        return torch.tensor(key_list)

    # jlist = junction_dict_to_keys(gjc_dict, threshold=5)
    # _, temp = get_wireframe_from_lines_and_junctions(lines3d_all, global_junctions[jlist], rel_matching_distance_threshold=0.3)
    # import pdb; pdb.set_trace()
    junctions3d_initial = torch.stack([global_junctions[k] for k,v in gjc_dict.items() if v.shape[0]>2])
    junctions3d_refined = torch.stack([v.mean(dim=0) for v in gjc_dict.values() if v.shape[0]>2])

    graph_initial, lines3d_wfi = get_wireframe_from_lines_and_junctions(lines3d_all.cuda(), junctions3d_initial.cuda(), rel_matching_distance_threshold=0)
    graph_refined, lines3d_wfr = get_wireframe_from_lines_and_junctions(lines3d_all.cuda(), junctions3d_refined.cuda(), rel_matching_distance_threshold=0)

    result_dict = {
        'junctions3d_initial': junctions3d_initial,
        'junctions3d_refined': junctions3d_refined,
        'lines3d_all': lines3d_all,
        'graph_initial': graph_initial,
        'graph_refined': graph_refined,
        'lines3d_wfi': lines3d_wfi,
        'lines3d_wfr': lines3d_wfr,
    }
    return result_dict


def visibility_checking(lines3d_all, eval_dataloader, model, *, 
                        mindis_th = 25, min_visible_views=1, device='cuda'):

    visibility = torch.zeros((lines3d_all.shape[0],),device=device)
    lines3d_visibility = torch.zeros((lines3d_all.shape[0],len(eval_dataloader)),device=device, dtype=torch.bool)
    duplicated_graph = torch.zeros((lines3d_all.shape[0],lines3d_all.shape[0]),device=device, dtype=torch.bool)
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

        
        dis1 = torch.sum((lines2d_all[:,None]-lines2d_gt[None,:,[0,1,2,3]])**2,dim=-1)
        dis2 = torch.sum((lines2d_all[:,None]-lines2d_gt[None,:,[2,3,0,1]])**2,dim=-1)
        dis = torch.min(dis1,dis2)
        
        mindis, labels = dis.min(dim=1)
        lines3d_visibility[mindis<mindis_th,indices[0]] = True

    lines3d_visible = lines3d_all[lines3d_visibility.sum(dim=1)>=min_visible_views]
    # lines3d_visible = lines3d_all[visibility>=min_visible_views]
    return lines3d_visible

def wireframe_recon(**kwargs):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    conf_path = kwargs['conf']
    conf = ConfigFactory.parse_file(kwargs['conf'])
    assert os.path.basename(conf_path) == 'runconf.conf'

    root = os.path.join(*conf_path.split('/')[:-1])

    dataset_conf = conf.get_config('dataset')
    dataset_conf['distance_threshold'] = float(1)

    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)

    conf_model = conf.get_config('model')
    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf_model)
    if torch.cuda.is_available():
        model.cuda()

    old_checkpnts_dir = os.path.join(root, 'checkpoints')
    checkpoint_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")

    print('Checkpoint: {}'.format(checkpoint_path))
    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))

    model.load_state_dict(saved_model_state['model_state_dict'])
    epoch = saved_model_state['epoch']

    print('evaluating...')

    model.eval()

    eval_dataset.distance = 1

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )

    wireframe_dir = os.path.join(root,'wireframes')
    utils.mkdir_ifnotexists(wireframe_dir)

    pth_path = os.path.join(wireframe_dir,'{}-neat.pth'.format(kwargs['checkpoint']))

    if os.path.exists(pth_path):
        initial_recon_results = torch.load(pth_path)

        loaded_kwargs = initial_recon_results.get('kwargs',{})
    else:
        loaded_kwargs = {}
    
    sha256 = make_hash_sha256(kwargs)[:8]

    
    if loaded_kwargs != kwargs:
        initial_recon_results = initial_recon(
            model, 
            eval_dataloader, 
            kwargs['chunksize'],
            junc_match_threshold=0.02,
            DEBUG=False, 
            line_dis_threshold = kwargs['distance'], 
            device='cuda')
        initial_recon_results['kwargs'] = kwargs

    out_basename = '{}-{}'.format(kwargs['checkpoint'],sha256)
    line_path = os.path.join(wireframe_dir,'{}-wfr.npz'.format(out_basename))

    # 5 views for dtu24
    lines3d_wfi_checked = visibility_checking(initial_recon_results['lines3d_wfi'], eval_dataloader, model, mindis_th = kwargs['ckdist'], min_visible_views=kwargs['ckview'])
    lines3d_wfr_checked = visibility_checking(initial_recon_results['lines3d_wfr'], eval_dataloader, model, mindis_th = kwargs['ckdist'], min_visible_views=kwargs['ckview'])
    initial_recon_results['lines3d_wfi_checked'] = lines3d_wfi_checked
    initial_recon_results['lines3d_wfr_checked'] = lines3d_wfr_checked

    basename = os.path.join(wireframe_dir,'{}-{}.npz'.format(out_basename,'{}'))


    for key in ['all','wfi', 'wfr', 'wfi_checked', 'wfr_checked']:
        np.savez(basename.format(key), lines3d=initial_recon_results['lines3d_{}'.format(key)].cpu().numpy())
        print('python evaluation/show.py --data {}'.format(basename.format(key)))

    torch.save(initial_recon_results, os.path.join(wireframe_dir,'{}-neat.pth'.format(out_basename)))
    print('done')



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

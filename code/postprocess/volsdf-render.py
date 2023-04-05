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
import matplotlib.pyplot as plot
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

def get_wireframe_from_lines_and_junctions(lines, junctions, *, 
                    rel_matching_distance_threshold = 0.01,
                    ):
    device = lines.device
    ep1, ep2 = lines[:, 0], lines[:, 1]
    cost1 = torch.cdist(ep1, junctions)
    cost2 = torch.cdist(ep2, junctions)
    mcost1, midx1 = cost1.min(dim=1)
    mcost2, midx2 = cost2.min(dim=1)
    is_matched = torch.max(mcost1,mcost2)<torch.norm(ep1-ep2,dim=-1)*rel_matching_distance_threshold

    graph = torch.zeros((junctions.shape[0], junctions.shape[0]), device=device)
    pair = torch.stack([torch.min(midx1,midx2),torch.max(midx1,midx2)],dim=1)
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

    
    images = []
    for indices, model_input, ground_truth in tqdm(eval_dataloader):
        if DEBUG and indices.item()>5:
            break

        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()    
        model_input['pose'] = model_input['pose'].cuda()

        num_pixels = eval_dataloader.dataset.img_res[0]*eval_dataloader.dataset.img_res[1]
        split = utils.split_input(model_input, num_pixels, n_pixels=chunksize,keys=['uv'])
        
        rgb_values = []
        # for s, lb, lines_gt in zip(split,split_label,split_lines):
        for s in tqdm(split):
            torch.cuda.empty_cache()
            out = model(s)
            rgb_values.append(out['rgb_values'].detach().cpu())
        rgb_values = torch.cat(rgb_values,dim=0)
        rgb_values = rgb_values.reshape(*eval_dataloader.dataset.img_res,-1)
        
        plot.imshow(rgb_values)
        plot.show()
    import pdb; pdb.set_trace()
            # trimesh.load_path(lines3d_[dis<10].cpu()).show()
            # print(dis.min().item())


def wireframe_recon(**kwargs):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    conf_path = kwargs['conf']
    conf = ConfigFactory.parse_file(kwargs['conf'])
    assert os.path.basename(conf_path) == 'runconf.conf'

    root = os.path.join(*conf_path.split('/')[:-1])

    dataset_conf = conf.get_config('dataset')
    # dataset_conf['distance_threshold'] = float(kwargs['distance'])

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

    # eval_dataset.distance = kwargs.get('distance',1)
    # eval_dataset.score_threshold = kwargs.get('score',0.05)

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )
    scan_id = 1
    mesh = plt.get_surface_high_res_mesh(
        sdf=lambda x: model.implicit_network(x)[:,0],
        resolution = 128,
        grid_boundary = conf.get_list('plot.grid_boundary'),
        level=conf.get_int('plot.level', default=0),    
        take_components = type(scan_id) is not str
        )
    mesh.export('{}/{}.ply'.format(root,kwargs['checkpoint']))
    # import pdb; pdb.set_trace()
    wireframe_dir = os.path.join(root,'wireframes')
    utils.mkdir_ifnotexists(wireframe_dir)

    line_path = os.path.join(wireframe_dir,'{}-wfr.npz'.format(kwargs['checkpoint']))

    initial_recon_results = initial_recon(model, eval_dataloader, kwargs['chunksize'])
    lines3d_wfi_checked = visibility_checking(initial_recon_results['lines3d_wfi'], eval_dataloader, model, mindis_th = 25, min_visible_views=1)
    lines3d_wfr_checked = visibility_checking(initial_recon_results['lines3d_wfr'], eval_dataloader, model, mindis_th = 25, min_visible_views=1)
    initial_recon_results['lines3d_wfi_checked'] = lines3d_wfi_checked
    initial_recon_results['lines3d_wfr_checked'] = lines3d_wfr_checked

    basename = os.path.join(wireframe_dir,'{}-{}.npz'.format(kwargs['checkpoint'],'{}'))

    for key in ['wfi', 'wfr', 'wfi_checked', 'wfr_checked']:
        np.savez(basename.format(key), lines3d=initial_recon_results['lines3d_{}'.format(key)].cpu().numpy())
        print('python evaluation/show.py --data {}'.format(basename.format(key)))

    torch.save(initial_recon_results, os.path.join(wireframe_dir,'{}-neat.pth'.format(kwargs['checkpoint'])))
    print('done')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    # parser.add_argument('--timestamp', required=True, type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
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
        checkpoint=opt.checkpoint,
        chunksize=opt.chunksize,
        distance=opt.dis_th,
        score=opt.score_th,
    )

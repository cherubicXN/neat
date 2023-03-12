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
import matplotlib.pyplot as plt
def sweep_ckpt(expdir, checkpoint):
    expdir = Path(expdir)
    ckpt_candidates = [x for x in expdir.glob('**/ModelParameters/{}.pth'.format(checkpoint))]
    if len(ckpt_candidates)> 1:
        msg = ['multiple timestamps containing the checkpoint {}: '.format(checkpoint)] + [x.relative_to(expdir).parts[0] for x in ckpt_candidates]
            
        raise RuntimeError(msg)

    candidate = ckpt_candidates[0]
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

    dataset_conf['distance_threshold'] = 5
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

    # eval_dataset.distance = kwargs.get('distance',20)
    # eval_dataset.distance_threshold = 20
    eval_dataset.score_threshold = kwargs.get('score',0.05)

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )
                            

    wireframe_dir = os.path.join(evaldir,'wireframes')
    mask_dir = os.path.join(evaldir,'masks')

    utils.mkdir_ifnotexists(wireframe_dir)
    utils.mkdir_ifnotexists(mask_dir)

    # line_path = os.path.join(wireframe_dir,'{}-d={}-s={}-fuse.npz'.format(kwargs['checkpoint'],kwargs['distance'],kwargs['score']))

    chunksize = kwargs['chunksize']

    sdf_threshold = kwargs['sdf_threshold']

    # data = np.load(kwargs['data'],allow_pickle=True)
    # if len(data['lines3d'].shape) ==1:
    #     lines3d_all = np.concatenate(data['lines3d'],axis=0)
    # else:
    #     lines3d_all = data['lines3d']

    device = 'cuda'
    # lines3d_all = torch.tensor(lines3d_all,device=device)
    # junctions_all = lines3d_all.reshape(-1,3).unique(dim=0)

    # graph = torch.zeros((junctions_all.shape[0],junctions_all.shape[0]),device=device)

    # idx1 = torch.cdist(lines3d_all[:,0],junctions_all).min(dim=-1)[1]
    # idx2 = torch.cdist(lines3d_all[:,1],junctions_all).min(dim=-1)[1]
    # graph[idx1,idx2] = 1
    # graph[idx2,idx1] = 1
    
    # scores_all = torch.tensor(data['scores']).cuda()
    # visibility = torch.zeros((lines3d_all.shape[0],),device=device)
    # lines3d_all = []

    # maskdirs = os.path.join(evaldir,'masks')
    # utils.mkdir_ifnotexists(maskdirs)
    
    # points3d_all = []
    # scores_all = []
    # visibility_score = torch.zeros((lines3d_all.shape[0],),device=device)

    # lines3d_visibility = torch.zeros((lines3d_all.shape[0],len(eval_dataloader)),device=device, dtype=torch.bool)
    # junctions_visibility = torch.zeros((junctions_all.shape[0],len(eval_dataloader)),device=device, dtype=torch.bool)

    images = []
    K_list = []
    R_list = []
    T_list = []

    MR = []
    for indices, model_input, ground_truth in tqdm(eval_dataloader):    
        rgb = ground_truth['rgb'][0].reshape(*eval_dataset.img_res,3)

        lines2d_gt = model_input['wireframe'][0].line_segments(0.05).cpu().numpy()
        mask = model_input['mask']
        mask = mask[0].reshape(*eval_dataset.img_res)
        # plt.imshow(mask.cpu().numpy())
        # plt.show()
        import cv2
        rgb[mask==0] = 1
        # mask = np.array(mask,dtype=np.uint8)
        # mask = (1-mask)*255
        rgb = rgb.cpu().numpy()[...,[2,1,0]]
        width, height = rgb.shape[1], rgb.shape[0]
        rgb = np.array(rgb*255,dtype=np.uint8)
        fig = plt.figure()
        fig.set_size_inches(rgb.shape[1]/rgb.shape[0],1,forward=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.xlim([-0.5, width-0.5])
        plt.ylim([height-0.5, -0.5])

        plt.imshow(rgb[...,::-1])
        # plt.imshow(mask==1,cmap='gray')
        plt.scatter(lines2d_gt[:,0],lines2d_gt[:,1],color='b',s=0.2,edgecolors='none',zorder=0.5)
        plt.scatter(lines2d_gt[:,2],lines2d_gt[:,3],color='b',s=0.2,edgecolors='none',zorder=0.5)
        plt.plot([lines2d_gt[:,0],lines2d_gt[:,2]],[lines2d_gt[:,1],lines2d_gt[:,3]],'-',linewidth=0.1)
        # import pdb; pdb.set_trace()
        plt.savefig(os.path.join(mask_dir,'{:06d}.png'.format(indices.item())),dpi=height)
        plt.close('all')
        MR.append(1-mask.sum()/mask.numel())

    print(MR[0], MR[5], MR[8])
    print(sum(MR)/len(MR))
    
    
    
    
    # axes[1].imshow(images[rand_ind[1]])
    # j2d = model.project2D(K_list[rand_ind[1]],R_list[rand_ind[1]],T_list[rand_ind[1]],junctions_all).reshape(-1,2).cpu().numpy()
    # axes[1].plot(j2d[juncs_ind.cpu().numpy(),0],j2d[juncs_ind.cpu().numpy(),1],'r.')
    # axes[1].plot([j2d[ix,0],j2d[jx,0]],[j2d[ix,1],j2d[jx,1]],'r-')
    
    import pdb; pdb.set_trace()
    

    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    # parser.add_argument('--data', type=str, required=True)
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
        # data=opt.data,
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

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
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
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

def hungarian(pred_ins, gt_ins, valid_ins_num, ins_num):
    @torch.no_grad()
    def reorder(cost_matrix, valid_ins_num):
        valid_scores = cost_matrix[:valid_ins_num].detach()
        valid_scores[torch.isnan(valid_scores)] = 10
        
        valid_scores = valid_scores.cpu().numpy()
        # try:
        row_ind, col_ind = linear_sum_assignment(valid_scores)
        # except:
        #     import pdb; pdb.set_trace()

        unmapped = ins_num - valid_ins_num
        if unmapped > 0:
            unmapped_ind = np.array(list(set(range(ins_num)) - set(col_ind)))
            col_ind = np.concatenate([col_ind, unmapped_ind])
        return row_ind, col_ind
    # preprocess prediction and ground truth
    pred_ins = pred_ins.permute([1, 0])
    gt_ins = gt_ins.permute([1, 0])
    pred_ins = pred_ins[None, :, :]
    gt_ins = gt_ins[:, None, :]

    cost_ce = torch.mean(-gt_ins * torch.log(pred_ins + 1e-8) - (1 - gt_ins) * torch.log(1 - pred_ins + 1e-8), dim=-1)

    # get soft iou score between prediction and ground truth, don't need do mean operation
    TP = torch.sum(pred_ins * gt_ins, dim=-1)
    FP = torch.sum(pred_ins, dim=-1) - TP
    FN = torch.sum(gt_ins, dim=-1) - TP
    cost_siou = TP / (TP + FP + FN + 1e-6)
    cost_siou = 1.0 - cost_siou

    # final score
    cost_matrix = cost_ce + cost_siou
    # get final indies order
    order_row, order_col = reorder(cost_matrix, valid_ins_num)

    return cost_ce, cost_siou, order_row, order_col

def ins_eval(pred_ins, gt_ins, gt_ins_num, ins_num, mask=None):
    if mask is None:
        pred_label = torch.argmax(pred_ins, dim=-1)
        valid_pred_labels = torch.unique(pred_label)
    else:
        pred_label = torch.argmax(pred_ins, dim=-1)

        pred_label[mask == 0] = ins_num  # unlabeled index for prediction set as -1
        valid_pred_labels = torch.unique(pred_label)[:-1]

    valid_pred_num = len(valid_pred_labels)
    # prepare confidence masks and confidence scores
    pred_conf_mask = np.max(pred_ins.numpy(), axis=-1)

    pred_conf_list = []
    valid_pred_labels = valid_pred_labels.numpy().tolist()
    for label in valid_pred_labels:
        index = torch.where(pred_label == label)
        ssm = pred_conf_mask[index[0]]
        pred_obj_conf = np.median(ssm)
        pred_conf_list.append(pred_obj_conf)
    pred_conf_scores = torch.from_numpy(np.array(pred_conf_list))

    # change predicted labels to each signal object masks not existed padding as zero
    pred_ins = torch.zeros_like(gt_ins)
    pred_ins[..., :valid_pred_num] = F.one_hot(pred_label)[..., valid_pred_labels]
    cost_ce, cost_iou, order_row, order_col = hungarian(pred_ins.reshape((-1, ins_num)),
                                                        gt_ins.reshape((-1, ins_num)),
                                                        gt_ins_num, ins_num)

    import pdb; pdb.set_trace()
    valid_inds = order_col[:gt_ins_num]
    ious_metrics = 1 - cost_iou[order_row, valid_inds]

    # prepare confidence values
    confidence = torch.zeros(size=[gt_ins_num])
    for i, valid_ind in enumerate(valid_inds):
        if valid_ind < valid_pred_num:
            confidence[i] = pred_conf_scores[valid_ind]
        else:
            confidence[i] = 0

    # ap = calculate_ap(ious_metrics, gt_ins_num, confidence=confidence, function_select='integral')

    invalid_mask = valid_inds >= valid_pred_num
    valid_inds[invalid_mask] = 0
    valid_pred_labels = torch.from_numpy(np.array(valid_pred_labels))
    return_labels = valid_pred_labels[valid_inds].cpu().numpy()
    return_labels[invalid_mask] = -1

    return pred_label, return_labels

# def wireframe_recon(**kwargs):
def wireframe_recon(
    conf, exps_folder_name, evals_folder_name, expname, scan_id=-1, **kwargs):
  
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
    
    timestamp = kwargs['timestamp']
    if timestamp is None:
        timestamp =sweep_ckpt(expdir, kwargs['checkpoint'])
    

    evaldir = os.path.join(expdir, timestamp )
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

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )

    wireframe_dir = os.path.join(evaldir,'wireframes')
    utils.mkdir_ifnotexists(wireframe_dir)

    line_path = kwargs['data'][:-4]+f"-group-v2-{kwargs['reproj_th']}"
    if kwargs['comment'] is not None:
        line_path = line_path + '-{}.npz'.format(kwargs['comment'])
    else:
        line_path = line_path + '.npz'

    chunksize = kwargs['chunksize']

    data = np.load(kwargs['data'],allow_pickle=True)
    if len(data['lines3d'].shape) ==1:
        lines3d_all = np.concatenate(data['lines3d'],axis=0)
    else:
        lines3d_all = data['lines3d']

    if 'scores' in data and kwargs['dscore'] is not None:
        scores = data['scores']
        lines3d_all = lines3d_all[scores<kwargs['dscore']]
        line_path = line_path[:-4] + '-{}ds.npz'.format(kwargs['dscore'])

    device = 'cuda'
    reproj_th = kwargs['reproj_th']
    lines3d_all = torch.tensor(lines3d_all,device=device)
    # scores_all = torch.tensor(data['scores']).cuda()
    visibility = torch.zeros((lines3d_all.shape[0],),device=device)
    
    visibility_score = torch.zeros((lines3d_all.shape[0],),device=device)
    lines3d_global_id = -torch.ones((lines3d_all.shape[0],len(eval_dataloader)),device=device,dtype=torch.long)
    lines3d_pi_all_a = model.attraction_network.feature(lines3d_all.reshape(-1,6)).detach()
    lines3d_pi_all_b = model.attraction_network.feature(lines3d_all[:,[1,0]].reshape(-1,6)).detach()
    lines3d_pi_all = 0.5*(lines3d_pi_all_a + lines3d_pi_all_b)
    # lines3d_pi_all = lines3d_pi_all_a 
    # lines3d_pi_all = model.attraction_network.feature(lines3d_all.mean(dim=1)).detach()
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

        dis1 = torch.sum((lines2d_all[:,None]-lines2d_gt[None,:,[0,1,2,3]])**2,dim=-1)
        dis2 = torch.sum((lines2d_all[:,None]-lines2d_gt[None,:,[2,3,0,1]])**2,dim=-1)
        dis = torch.min(dis1,dis2)
        # dis = torch.tensor(dis,device='cuda')
        mindis,minidx = dis.min(dim=1)
        lines3d_sel = lines3d_all[mindis<reproj_th]
        lines3d_pi = lines3d_pi_all[mindis<reproj_th]
        labels_sel = minidx[mindis<reproj_th]
        valid_gt_labels = torch.unique(labels_sel)
        valid_gt_num = len(valid_gt_labels)
        gt_ins = torch.zeros(labels_sel.shape[0],1024)
        gt_ins[:,:valid_gt_num] = F.one_hot(labels_sel)[...,valid_gt_labels.long()]
        pred_label = lines3d_pi.max(dim=-1)[1]
        pred_label, pred_matched_order = ins_eval(lines3d_pi.cpu(), gt_ins, valid_gt_num, 1024)
        
        lines3d_global_id[mindis<reproj_th,indices.item()] = pred_label.cuda()
        visibility[mindis<reproj_th] += 1
        visibility_score[mindis<reproj_th] += torch.exp(-mindis)[mindis<reproj_th]
    
    lines3d_dict = defaultdict(list)
    score_dict = defaultdict(list)
    for line3d, line3d_id, score in zip(lines3d_all, lines3d_global_id,visibility_score):
        if (line3d_id==-1).all().item():
            continue

        unique_ids, counts = line3d_id[line3d_id!=-1].unique(return_counts=True)
        if counts.max()==1:
            continue
        gid = unique_ids[counts.argmax()].item()
        if unique_ids.numel() == 1:
            lines3d_dict[gid].append(line3d.cpu())
            score_dict[gid].append(score.item())
    lines3d_final = []
    import open3d as o3d

    ffn = model.attraction_network.feature
        # d_output = torch.ones_like()
    for key in lines3d_dict.keys():
        lines3d_dict[key] = torch.stack(lines3d_dict[key])
        lines3d_cur = lines3d_dict[key]
        
        centers = lines3d_cur.mean(dim=1)
        cdist = torch.norm(centers[None]-centers[:,None],dim=-1,p=2)
        # cdist_th = torch.min(cdist + torch.eye(cdist.shape[0]))*2
        cdist_th = cdist[~torch.eye(cdist.shape[0],dtype=torch.bool)].mean().clamp_max(0.01)
        # clusters = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers.cpu().numpy())).cluster_dbscan(0.01,1)
        clusters = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers.cpu().numpy())).cluster_dbscan(cdist_th,1)
        clusters = torch.tensor(clusters)
        # import pdb; pdb.set_trace()
        labels = clusters.unique(return_counts=True)[0]
        scores = torch.tensor(score_dict[key])
        for lb in labels:
            if lb == -1:
                continue
            lines3d_cls = lines3d_cur[clusters==lb]
            scores_cls = scores[clusters==lb]
            scores_sorted, argscore = scores_cls.sort(dim=-1,descending=True)
            head = lines3d_cls[argscore[:1]]
            tails = lines3d_cls[argscore]
            sign = torch.norm(head[:,0]-tails[:,0],p=2,dim=-1,keepdim=True)<torch.norm(head[:,0]-tails[:,1],p=2,dim=-1,keepdim=True)
            x1 = torch.where(sign,tails[:,0],tails[:,1])
            x2 = torch.where(sign,tails[:,1],tails[:,0])

            lines_adj = torch.stack((x1,x2),dim=1)
            lines_mean = torch.sum(scores_sorted.reshape(-1,1,1)*lines_adj,dim=0)/torch.sum(scores_sorted).clamp_min(1e-10)

            lines3d_final.append(lines_adj.mean(dim=0))
    lines3d_final = torch.stack(lines3d_final)#.cpu().numpy()

    # dist1 = torch.norm(lines3d_final[:,None].cuda() - lines3d_all[None],p=2,dim=-1).mean(dim=-1)
    # dist2 = torch.norm(lines3d_final[:,None].cuda() - lines3d_all[None,:,[1,0]],p=2,dim=-1).mean(dim=-1)
    # dist = torch.min(dist1,dist2)
    # lines3d_final = lines3d_all[dist.min(dim=1)[1]].cpu()
    # import pdb; pdb.set_trace()
    print('save the reconstructed wireframes to {}'.format(line_path))
    np.savez(line_path,lines3d=lines3d_final.cpu().numpy())
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--evals_folder', type=str, default='evals', help='The evaluation folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--timestamp', default=None, type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--chunksize', default=2048, type=int, help='the chunksize for rendering')
    parser.add_argument('--dis-th', default=1, type=int, help='the distance threshold of 2D line segments')
    parser.add_argument('--dscore', default=None, type=float, help='the score ')
    parser.add_argument('--reproj-th', default=5, type=float, help='the reprojection threshold of 3D line segments (in px)')
    parser.add_argument('--comment', default=None, type=str, help='the suffix of the output file')
    # parser.add_argument('--score-th', default=0.05, type=float, help='the score threshold of 2D line segments')
    
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
        distance=opt.dis_th,
        dscore = opt.dscore,
        preview = opt.preview,
        reproj_th = opt.reproj_th,
        comment = opt.comment
    )

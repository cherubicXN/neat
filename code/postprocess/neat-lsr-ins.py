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

import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
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

    line_path = os.path.join(wireframe_dir,'{}-d={}-ins.pth'.format(kwargs['checkpoint'],kwargs['distance']))

    chunksize = kwargs['chunksize']

    sdf_threshold = kwargs['sdf_threshold']

    # lines3d_all = []

    maskdirs = os.path.join(evaldir,'masks')
    utils.mkdir_ifnotexists(maskdirs)
    
    lines3d_dict = {}
    cnt = 0
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
        import cv2
        mask_im = mask.numpy().reshape(*eval_dataset.img_res)
        mask_im = np.array(mask_im,dtype=np.uint8)*255
        mask_path = os.path.join(maskdirs,'{:04d}.png'.format(indices.item()))
        cv2.imwrite(mask_path, mask_im)
        lines = model_input['lines'][0].cuda()
        labels = model_input['labels'][0]
        split = utils.split_input(model_input, mask.sum().item(), n_pixels=chunksize,keys=['uv','uv_proj','lines'])
        # import pdb; pdb.set_trace()
        split_label = torch.split(labels[mask[0]],chunksize)
        split_lines = torch.split(lines[mask[0]],chunksize)

        lines3d = []
        lines2d = []
        predins = []
        instgt = []
        # emb_by_dict = defaultdict(list)
        for s, lb, lines_gt in zip(tqdm(split),split_label,split_lines):
        # for s, lb, lines_gt in zip(split,split_label,split_lines):
            torch.cuda.empty_cache()
            out = model(s)
            lines3d_ = out['lines3d'].detach()
            lines2d_ = out['lines2d'].detach().reshape(-1,4)
            predins.append(out['ins'].detach())
            lines3d.append(lines3d_)
            lines2d.append(lines2d_)
            instgt.append(lb)
            # lines_gt = lines_gt[:,:-1]
        predins = torch.cat(predins)
        predins = torch.cat([predins,predins],dim=0)
        instgt = torch.cat(instgt)
        instgt = torch.cat([instgt,instgt])
        # gtins = F.one_hot(gtins,num_classes=1024)
        lines3d = torch.cat(lines3d)
        lines3d = torch.cat((lines3d,lines3d[:,[1,0]]),dim=0)
        lines2d = torch.cat(lines2d,dim=0)
        lines2d = torch.cat((lines2d,lines2d[:,[2,3,0,1]]),dim=0)

        gt_lines = model_input['wireframe'][0].line_segments(0.01).cuda()[:,:-1]

        dis = torch.sum((lines2d[:,None]-gt_lines[None])**2,dim=-1)

        mindis, minidx = dis.min(dim=1)

        if (mindis<10).sum() == 0:
            continue
        labels = minidx[mindis<10].unique()
        lines3d_valid = lines3d[mindis<10]
        predins_valid = predins[mindis<10]
        assignment = minidx[mindis<10]
        instgt_valid = instgt[mindis<10]
        # valid_gt_labels = torch.unique(instgt_valid)
        # valid_gt_num = len(valid_gt_labels)
        # gt_ins = F.one_hot(instgt_valid,num_classes=1024)
        # gt_ins = torch.zeros(instgt_valid.shape[0],1024)
        # gt_ins[:,:valid_gt_num] = F.one_hot(instgt_valid)[...,valid_gt_labels.long()]
        # gt_label_np = valid_gt_labels.cpu().numpy()
        # pred_label, pred_matched_order = ins_eval(predins_valid.cpu(), gt_ins, valid_gt_num, 1024)
        lines3d = []
        predins = []
        instgt = []
        for i, label in enumerate(labels):
            idx = (assignment==label).nonzero().flatten()
            if idx.numel() == 0:
                continue
            val = lines3d_valid[idx].mean(dim=0)
            lines3d.append(val)
            val = predins_valid[idx].mean(dim=0)
            predins.append(val)
            instgt.append(i)
            # val = instgt_valid[idx]
        lines3d = torch.stack(lines3d)
        predins = torch.stack(predins)
        instgt = torch.tensor(instgt, dtype=torch.long)
        valid_gt_labels = torch.unique(instgt)
        valid_gt_num = len(valid_gt_labels)

         # gt_ins = F.one_hot(instgt_valid,num_classes=1024)
        gt_ins = torch.zeros(instgt.shape[0],1024)
        gt_ins[:,:valid_gt_num] = F.one_hot(instgt)[...,valid_gt_labels.long()]
        gt_label_np = valid_gt_labels.cpu().numpy()
        pred_label, pred_matched_order = ins_eval(predins.cpu(), gt_ins, valid_gt_num, 1024)
        # pred_conf, pred_label = predins.max(dim=-1)

        # for lb in pred_matched_order.unique():
        for i, lb in enumerate(pred_matched_order):
            # idx = (lb==pred_label).nonzero().flatten()
            if lb.item() in lines3d_dict:
                lines3d_dict[lb.item()] = torch.cat((lines3d_dict[lb.item()],lines3d[i][None]))
            else:
                lines3d_dict[lb.item()] = lines3d[i][None]
            # lines3d_dict[lb.item()].append(lines3d_valid[idx])
        # trimesh.load_path(lines3d.cpu()).show()
        print('\n',len(lines3d_dict),'<-', len(lines3d))

    print(line_path)
    torch.save(lines3d_dict, line_path)

    # device = 'cuda'

    # for indices, model_input, ground_truth in tqdm(eval_dataloader):
    #     lines2d_gt = model_input['wireframe'][0].line_segments(0.05)
    #     mask = model_input['mask']
    #     model_input["intrinsics"] = model_input["intrinsics"].to(device=device)#.cuda()
    #     model_input['pose'] = model_input['pose'].to(device=device)

    #     K = model_input["intrinsics"][0,:3,:3]
    #     proj_mat = model_input['pose'][0].inverse()[:3]
    #     R = proj_mat[:,:3]
    #     T = proj_mat[:,3:]

    #     lines2d_gt = lines2d_gt.to(device=device)
    #     uncertain_keys = []
    #     for key, lines3d_all in lines3d_dict.items():
    #         if key == -1:
    #             continue
    #         lines2d_all = model.project2D(K,R,T,lines3d_all).reshape(-1,4)
    #         dis1 = torch.sum((lines2d_all[:,None]-lines2d_gt[None,:,[0,1,2,3]])**2,dim=-1)
    #         dis2 = torch.sum((lines2d_all[:,None]-lines2d_gt[None,:,[2,3,0,1]])**2,dim=-1)
    #         dis = torch.min(dis1,dis2)
    #         # dis = torch.tensor(dis,device='cuda')
    #         mindis, minidx = dis.min(dim=1)
    #         if (mindis<5).sum() == 0:
    #             continue
    #         if minidx[mindis<5].unique().numel() > 1:
    #             uncertain_keys.append(key)
    #         else:
    #             lines3d_dict[key] = lines3d_dict[key][mindis.argmin()][None]
        
    #     for key in uncertain_keys:
    #         lines3d_dict[-1] = torch.cat((lines3d_dict[-1],lines3d_dict.pop(key)))

    # # trimesh_lines = []
    # # for key, x in lines3d_dict.items():
    # #     tm = trimesh.load_path(x.cpu())
    # #     num_ent = len(tm.entities)
    # #     rand_rgb = np.stack([np.random.choice(255,3)]*num_ent)
    # #     tm.colors = rand_rgb
    # #     trimesh_lines.append(tm)
    
    # lines3d_all = []
    # for key, x in lines3d_dict.items():
    #     if key == -1:
    #         continue
    #     lines3d_all.append(x)

    # lines3d_all = torch.cat(lines3d_all,dim=0).cpu().numpy()
    # # lines3d_all = np.array([l.numpy() for l in lines3d_all],dtype=object)
    # # points3d_all = np.array([l.numpy() for l in points3d_all],dtype=object)
    # # scores_all = torch.stack(scores_all).cpu().numpy()

    # cameras = torch.cat([model_input['pose'] for indices, model_input, ground_truth in tqdm(eval_dataloader)],dim=0)
    # cameras = cameras.numpy()
   
    # # line_path = os.path.join(wireframe_dir,'{}-{:.0e}.npz'.format(kwargs['checkpoint'],sdf_threshold))

    # np.savez(line_path,lines3d=lines3d_all,cameras=cameras),#scores=scores_all,points3d_all=points3d_all)
    # print('save the reconstructed wireframes to {}'.format(line_path))
    # print('python evaluation/show.py --data {}'.format(line_path))

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

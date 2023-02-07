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

def lshow(lines2d, *args,**kwargs):
    import matplotlib.pyplot as plt
    # if image is not None:
        # plt.imshow(image.cpu().numpy())
    if isinstance(lines2d, torch.Tensor):
        l = lines2d.detach().cpu().numpy()
    else:
        l = lines2d
    return plt.plot([l[:,0],l[:,2]],[l[:,1],l[:,3]],*args,**kwargs)


def project_point_to_line(line_segs, points):
    dir_vec = line_segs[:,2:] - line_segs[:,:2]
    coords1d = torch.sum((points-line_segs[:,:2])*dir_vec,dim=-1)/(torch.norm(dir_vec,dim=-1)**2)

    projection = line_segs[:,:2] + coords1d[:,None]*dir_vec

    dist_to_line = torch.norm(projection-points,dim=-1)

    return coords1d, dist_to_line

def get_segment_overlap(seg_coord1d):
    seg_coord1d = torch.sort(seg_coord1d, dim=-1)[0]
    overlap = ((seg_coord1d[..., 1] > 0) * (seg_coord1d[..., 0] < 1)
               * (np.minimum(seg_coord1d[..., 1], 1)
                  - np.maximum(seg_coord1d[..., 0], 0)))
    return overlap

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


    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    timestamp = kwargs['timestamp']
    if timestamp is None:
        timestamp = sweep_ckpt(expdir, kwargs['checkpoint'])

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
    
    points3d_all = []
    scores_all = []

    bb_dict = np.load('../data/DTU/bbs.npz')
    grid_params = bb_dict[str(scan_id)]
    input_min = torch.tensor(grid_params[0]).float()
    input_max = torch.tensor(grid_params[1]).float()
    from utils.plots import get_grid
    grid = get_grid(None, 100, input_min=input_min, input_max=input_max, eps=0.0)

    points = grid['grid_points']
    z = []
    for i, pnts in enumerate(torch.split(points, 50000, dim=0)):
        z.append(model.implicit_network.get_sdf_vals(pnts).detach())
    z = torch.cat(z).flatten()

    points = points[z.abs().flatten()<1e-2]

    points_cnt = torch.zeros(points.shape[0],dtype=torch.float32)

    points_view = torch.zeros(
        (points.shape[0],len(eval_dataloader)),dtype=torch.long)-1

    lines_view = torch.zeros(
        (points.shape[0],len(eval_dataloader),2,3),dtype=torch.float)-1
    
    query_3d_all = []
    for indices, model_input, ground_truth in tqdm(eval_dataloader):    
        mask = model_input['mask']
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input['uv'] = model_input['uv']#[:,mask[0]]
        model_input["uv_proj"] = model_input['uv_proj']#[:,mask[0]]
        model_input['lines'] = model_input['lines'].cuda()
        # randidx = torch.randperm(model_input['uv'].shape[1])
        # model_input['uv'] = model_input['uv'][:,randidx]
        model_input['pose'] = model_input['pose'].cuda()
        uv = model_input['uv'][0].reshape(*eval_dataset.img_res,2)
        uv_proj = model_input['uv_proj'][0].reshape(*eval_dataset.img_res,2)
        line_map = model_input['lines'][0].reshape(*eval_dataset.img_res,5)

        K = model_input["intrinsics"][0,:3,:3]
        RT = model_input['pose'][0].inverse()
        pts2d = model.project2D(K,RT[:3,:3],RT[:3,3:],points)
        pts2dl = pts2d.round().long()
        
        is_in = (pts2dl[:,0]>=0)*(pts2dl[:,0]<=eval_dataset.img_res[1]-1)*(pts2dl[:,1]>=0)*(pts2dl[:,1]<=eval_dataset.img_res[0]-1)

        mask_im = mask.reshape(*eval_dataset.img_res)
        label_im = model_input['labels'].reshape(*eval_dataset.img_res)

        qx = pts2dl[:,0]
        qy = pts2dl[:,1]

        label_set = label_im[qy[is_in],qx[is_in]].unique()

        model_input['uv'] = uv[qy[is_in],qx[is_in]][None]
        model_input['uv_proj'] = uv_proj[qy[is_in],qx[is_in]][None]
        model_input['lines'] = line_map[qy[is_in],qx[is_in]][None]
        split = utils.split_input(model_input, is_in.sum().item(), n_pixels=chunksize,keys=['uv','uv_proj','lines'])

        lines3d = []
        lines2d = []
        query3d = []
        for sp in tqdm(split):
            torch.cuda.empty_cache()
            out = model(sp)
            lines3d.append(out['lines3d'].detach())
            lines2d.append(out['lines2d'].detach())
            query3d.append(out['l3d'].detach().cpu())
            
        lines3d = torch.cat(lines3d)
        lines2d = torch.cat(lines2d)
        query3d = torch.cat(query3d)
        lines2d = lines2d.reshape(-1,4)

        dis = torch.min(
            torch.sum((lines2d-line_map[qy[is_in],qx[is_in],:-1])**2,dim=-1),
            torch.sum((lines2d[:,[2,3,0,1]]-line_map[qy[is_in],qx[is_in],:-1])**2,dim=-1))

        lines2d_gt = line_map[qy[is_in],qx[is_in],:-1]

        t1d1, dis_orth_1 = project_point_to_line(lines2d_gt, lines2d[:,:2])
        t1d2, dis_orth_2 = project_point_to_line(lines2d_gt, lines2d[:,2:])
        overlap = get_segment_overlap(torch.stack((t1d1.cpu(),t1d2.cpu()),dim=-1)).cuda()

        is_orth = (torch.max(dis_orth_1,dis_orth_2))<1
        is_orth *= overlap>0.5
        # t1 = lines2d[dis<10].cpu().numpy()
        # t2 = line_map[qy[is_in],qx[is_in],:-1][dis<10].cpu().numpy()

        ang1 = (lines2d[:,:2]-lines2d[:,2:])/torch.norm((lines2d[:,:2]-lines2d[:,2:]),dim=-1,keepdim=True)
        ang2 = (lines2d_gt[:,:2]-lines2d_gt[:,2:])/torch.norm((lines2d_gt[:,:2]-lines2d_gt[:,2:]),dim=-1,keepdim=True)

        ang_dis = torch.sum(ang1*ang2,dim=-1).abs()
        # is_perfect = (dis.cpu()<10)#*(ang_dis.cpu()>0.999)
        is_perfect = is_orth.cpu()
        # print(torch.sum(dis.cpu()<10), is_perfect.sum())
        
        lines3d_all.append(lines3d[is_perfect])
        query_3d_all.append(query3d[is_perfect] )
        points_cnt[is_in] += mask_im[qy[is_in],qx[is_in]]*is_perfect
        points_view[is_in,indices.item()] = torch.where(is_perfect,label_im[qy[is_in],qx[is_in]],label_im[qy[is_in],qx[is_in]]*0-1)

        is_ok =  mask_im[qy[is_in],qx[is_in]]*is_perfect

        ind_ok = is_in.nonzero().flatten()[is_ok]

        lines_view[ind_ok,indices.item()] = lines3d[is_perfect].cpu()

        # if (indices.item()+1)%10 == 0:
        # if indices.item()>3:
            # temp = torch.cat(lines3d_all).cpu()
            # trimesh.load_path(temp).show()
            # break
    
    # lines3d_all = torch.cat(lines3d_all).cpu()

    is_multiview = points_cnt>1

    argsort = torch.argsort(points_cnt[is_multiview],descending=True)

    points_view = points_view[is_multiview][argsort]
    lines_view = lines_view[is_multiview][argsort]

    is_visited = torch.zeros(points_view.shape[0])


    lines_nms = []
    for i, pvi in enumerate(points_view):
        if is_visited[i]:
            continue
        lines_all_i = []
        flag = False
        for j in range(i+1,points_view.shape[0]):
            pvj = points_view[j]
            identical = (pvi==pvj)*(pvi>-1)
            identical_score = identical.sum()/(pvi>-1).sum().clamp_min(1)
            if identical_score == 0:
                continue
            lines_i = lines_view[i,identical]
            lines_j = lines_view[j,identical]
            lines_all_i.append(lines_i)
            lines_all_i.append(lines_j)
            if identical_score>0.75:
                is_visited[j] = 1
                flag = True
        if len(lines_all_i) == 0:
            continue
        if flag:
            is_visited[i] = True
        
        lines_nms.append(torch.cat(lines_all_i).mean(dim=0,keepdim=True))
        print(len(lines_nms))
    import pdb; pdb.set_trace()
    
    query_a = 0
    keys_a = (affinity[0]>0).nonzero().flatten()

    lines_sub = [lines_view[query_a,points_view[query_a]>-1]]

    import pdb; pdb.set_trace()

    # query_3d_all = torch.cat(query_3d_all)
    # query_sdf, feats, normals = model.implicit_network.get_outputs(query_3d_all.cuda())
    # query_3d_new = (query_3d_all.cuda()-query_sdf*normals).detach()
    # _, feats, normals = model.implicit_network.get_outputs(query_3d_new)
    # model.attraction_network(query_3d_new, normals, feats)
    
    # lines_candidates = lines_view[points_cnt>1]
    # weights = points_view[points_cnt>1]

    
    import pdb; pdb.set_trace()

    out = np.array([l.cpu().numpy() for l in lines3d_all],dtype=object)
    np.savez('temp.npz',lines3d=out)
    import pdb; pdb.set_trace()
    

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

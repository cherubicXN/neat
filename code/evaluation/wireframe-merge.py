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
from scipy.optimize import linear_sum_assignment

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
    # for indices, model_input, ground_truth in tqdm(eval_dataloader):    
    #     rgb = ground_truth['rgb'].reshape(*eval_dataset.img_res,3)
    #     wireframe = model_input['wireframe'][0]
    #     lsd = wireframe.line_segments(0.05).numpy()
    #     plt.imshow(rgb)
    #     plt.plot([lsd[:,0],lsd[:,2]],[lsd[:,1],lsd[:,3]],'r-')
    #     import pdb; pdb.set_trace()
    for num_views, (indices, model_input, ground_truth) in enumerate(tqdm(eval_dataloader)):    
        mask = model_input['mask']
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input['uv'] = model_input['uv'][:,mask[0]]
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
        split = utils.split_input(model_input, mask.sum().item(), n_pixels=chunksize)
        split_label = torch.split(labels[mask[0]],chunksize)
        split_lines = torch.split(lines[mask[0]],chunksize)

        lines3d = []
        lines3d_by_dict = defaultdict(list)

        # emb_by_dict = defaultdict(list)
        for s, lb, lines_gt in zip(tqdm(split),split_label,split_lines):
            torch.cuda.empty_cache()
            out = model(s)
            lines3d_ = out['lines3d'].detach()
            lines3d_length = torch.norm(lines3d_[:,0]-lines3d_[:,1],dim=-1)
            lines3d_aux = [o.detach() for o in out['lines3d-aux']]

            is_inlier = torch.ones(lines3d_.shape[0],dtype=torch.bool,device='cuda')
            for aux in lines3d_aux:
            # lines3d_aux = lines3d_aux[0]
                dis_3d = torch.min(
                    torch.norm(lines3d_-aux,dim=-1).max(dim=-1)[0],
                    torch.norm(lines3d_-aux[:,[1,0]],dim=-1).max(dim=-1)[0],
                )
                is_inlier = (dis_3d<(lines3d_length*0.5))*is_inlier

            lines2d_ = out['lines2d'].detach().reshape(-1,4)
            # for l__ in out['lines3d-aux']:
            #     lines3d_ += l__.detach()
            # lines3d_ /= 1+len(out['lines3d-aux'])
            # lines3d_ = (lines3d_ + out['lines3d-aux'][0].detach() + out['lines3d-aux'][1])/3.0
            # lines3d_ = out['lines3d-aux'][-1].detach()
            # lines2d_ = out['lines2d-aux'][-1].detach().reshape(-1,4)
            # import pdb; pdb.set_trace()
            lines_gt = lines_gt[:,:-1]
            tspan = torch.linspace(0,1,16,device='cuda').reshape(1,-1,1)
            lines3d_diff = lines3d_[:,1:] - lines3d_[:,:1]
            points = lines3d_[:,:1] + tspan*lines3d_diff
            sdf_vals = model.implicit_network.get_sdf_vals(points.reshape(-1,3)).flatten().reshape(points.shape[:-1]).detach()
            sdf_vals = sdf_vals.abs()
            sdf_vals_mean = sdf_vals.mean(dim=-1)
            mask_ = sdf_vals_mean<sdf_threshold
            mask_ *= is_inlier
            # sdf_vals_max = sdf_vals.max(dim=-1)[0]
            # embeddings.append(out['emb'].detach())
            # embeddings_ = out['emb'].detach()
            
            # scores = out['lines3d-score'].detach()
            # mask_ = out['lines3d-score'].detach().abs()<sdf_threshold
            if mask_.sum() == 0:
                continue
            lines3d_valid = lines3d_[mask_]
            lines2d_valid = lines2d_[mask_]
            lines2d_gt = lines_gt[mask_]
            labels_valid = lb[mask_]
            label_set = labels_valid.unique()
            for label_ in label_set:
                idx = (labels_valid==label_).nonzero().flatten()
                # print(idx.shape)
                lines3d_by_label = lines3d_valid[idx]
                lines2d_by_label = lines2d_valid[idx]
                # emb_by_label = embeddings_[idx]
                lines2d_gt_ = lines2d_gt[idx]
                dis1 = torch.sum((lines2d_by_label-lines2d_gt_)**2,dim=-1)
                dis2 = torch.sum((lines2d_by_label-lines2d_gt_[:,[2,3,0,1]])**2,dim=-1)
                dis = torch.min(dis1,dis2)
                is_swap = (dis==dis2)
                lines3d_by_label[is_swap] = lines3d_by_label[is_swap][:,[1,0]]
                is_correct = dis<10
                if is_correct.sum()==0:
                    continue
                
                lines3d_by_dict[label_.item()].append((lines3d_by_label[is_correct],dis[is_correct]))
                # emb_by_dict[label_.item()].append(emb_by_label[is_correct])
                # lines3d.append(lines3d_by_label)
        # for k in emb_by_dict.keys():
        #     emb_by_dict[k] = torch.cat(emb_by_dict[k]).mean(dim=0)
        # temp = torch.stack([v for v in emb_by_dict.values()],dim=0)

        for key, val in lines3d_by_dict.items():
            dis = torch.cat([v[1] for v in val])
            val = torch.cat([v[0] for v in val])

            # import pdb; pdb.set_trace()
            # val = torch.cat(val).cpu()
            if val.shape[0] == 1:
                lines3d.append(val[0])
                continue
            # import pdb; pdb.set_trace()
            # lines_kept = torch.sum(torch.softmax(-dis,dim=0)[:,None,None]*val,dim=0)
            lines_kept = val.mean(dim=0)
            lines3d.append(lines_kept)

        if len(lines3d)>0:
            lines3d = torch.stack(lines3d,dim=0).cpu()
            # trimesh.load_path(lines3d).show()

            if len(lines3d_all) == 0:
                lines3d_all = lines3d.clone()
            else:

                dis = torch.min(
                    torch.norm(lines3d_all[:,None]-lines3d[None],dim=-1).mean(dim=-1),
                    torch.norm(lines3d_all[:,None]-lines3d[None,:,[1,0]],dim=-1).mean(dim=-1),)
                md, mid = dis.min(dim=1)
                length = torch.norm(lines3d_all[:,0]-lines3d_all[:,1],dim=-1)
                is_exist = md<0.05*length
                is_new = torch.ones(lines3d.shape[0],dtype=torch.bool,device='cuda')
                is_new[mid[is_exist]] = False
                lines3d_all = torch.cat((lines3d_all,lines3d[is_new]))

                print(is_new.sum().item(), 'lines are newly added to yield ', lines3d_all.shape[0], 'line segments \n')
                # avg = 0.5*(lines3d[assign[1][is_exist]] + lines3d_all[assign[0][is_exist]])
                # lines3d_all[assign[0][is_exist]] = avg



                # import pdb; pdb.set_trace()
                # lines3d_all.append(lines3d)
        else:
            continue
            
        if kwargs['preview']>0 and num_views%kwargs['preview']== 0:
            trimesh.load_path(lines3d_all.cpu()).show()
    
   
    # lines3d_all = np.array([l.numpy() for l in lines3d_all],dtype=object)
    lines3d_all = lines3d_all.cpu()

    cameras = torch.cat([model_input['pose'] for indices, model_input, ground_truth in tqdm(eval_dataloader)],dim=0)
    cameras = cameras.numpy()
    wireframe_dir = os.path.join(evaldir,'wireframes')
    utils.mkdir_ifnotexists(wireframe_dir)

    line_path = os.path.join(wireframe_dir,'{}-{:.0e}.npz'.format(kwargs['checkpoint'],sdf_threshold))

    np.savez(line_path,lines3d=lines3d_all,cameras=cameras,)
    print('save the reconstructed wireframes to {}'.format(line_path))
    print('python evaluation/show.py --data {}'.format(line_path))

    
    

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
    parser.add_argument('--sdf-threshold', default=1e-3, type=float, help='the sdf threshold')
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

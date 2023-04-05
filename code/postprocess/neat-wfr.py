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

class WireframeRecon:
    def __init__(self, conf, expname, exps_folder_name, evals_folder_name, timestamp, checkpoint, scan_id, **kwargs):
        self.conf = ConfigFactory.parse_file(conf)
        self.expname = self.conf.get_string('train.expname') + expname

        self.scan_id = scan_id

        if self.scan_id == -1:
            self.scan_id = self.conf.get_int('train.scan_id', default=-1)
        if self.scan_id != -1:
            self.expname = os.path.join(self.expname,'{}'.format(scan_id))
        
        self.expdir = os.path.join(exps_folder_name, self.expname)
        if timestamp is None:
            self.timestamp = sweep_ckpt(self.expdir, checkpoint)
        else:
            self.timestamp = timestamp
        
        self.evaldir = os.path.join(self.expdir, self.timestamp)
        os.makedirs(self.evaldir, exist_ok=True)

        dataset_conf = self.conf.get_config('dataset')
        dataset_conf['distance_threshold'] = float(kwargs['distance'])
        if self.scan_id != -1:
            dataset_conf['scan_id'] = self.scan_id
        
        self.dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)

        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)
        if torch.cuda.is_available():
            self.model.cuda()
        
        old_checkpnts_dir = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        checkpoint_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(checkpoint) + ".pth")

        print('Checkpoint: {}'.format(checkpoint_path))
        saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', checkpoint + ".pth"))
        self.model.load_state_dict(saved_model_state['model_state_dict'])

        self.epoch = saved_model_state['epoch']

        self.model.eval()

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False, collate_fn=self.dataset.collate_fn)

        self.image_width = self.dataset.img_res[1]
        self.image_height = self.dataset.img_res[0]

        self.outdir = os.path.join(self.evaldir, 'wireframes')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def run(self, 
            sdf_refine = 1, 
            reproj_th = 10, 
            num_samples_per_line = 32,
            sample_sdf_th = 0.01,
            sample_consis_th = 0.9,
            min_num_visibility = 1,
        ):
        model = self.model
        global_junctions = self.model.ffn(self.model.latents).detach()
        gjc_dict = defaultdict(list)
        if sdf_refine > 0:
            for i in range(sdf_refine):
                glj_sdf, glj_feats, glj_grad = model.implicit_network.get_outputs(global_junctions)
                global_junctions = (global_junctions - glj_sdf*glj_grad).detach()
        
        global_junctions_vis = torch.zeros((global_junctions.shape[0]))

        for indices, model_input, ground_truth in tqdm(self.dataloader):
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input['pose'] = model_input['pose'].cuda()
            K = model_input["intrinsics"][0,:3,:3]
            proj_mat = model_input['pose'][0].inverse()[:3]
            R = proj_mat[:,:3]
            T = proj_mat[:,3:]

            junctions2d_gt = model_input['wireframe'][0].vertices.cuda()
            lines2d_gt = model_input['wireframe'][0].line_segments(0.05).cuda()[:,:-1]
            lines2d_gt = torch.cat((lines2d_gt,lines2d_gt[:,[2,3,0,1]]),dim=0)
            junctions2d = model.project2D(K,R,T,global_junctions)
            is_inside = (junctions2d[:,0]>=0)*(junctions2d[:,0]<self.image_width)*(junctions2d[:,1]>=0)*(junctions2d[:,1]<self.image_height)

            jcost = torch.cdist(junctions2d_gt,junctions2d[is_inside])

            jassign = linear_sum_assignment(jcost.cpu().numpy())
            jcost_assign = jcost[jassign[0],jassign[1]]
            source_idx = jassign[0][jcost_assign.cpu().numpy()<reproj_th]
            target_idx = is_inside.nonzero().flatten()[jassign[1]][jcost_assign.cpu().numpy()<reproj_th]
            global_junctions_vis[target_idx] += 1
        
        # global_junctions = global_junctions[global_junctions_vis>=min_num_visibility]

        ix, iy = torch.triu_indices(global_junctions.shape[0], global_junctions.shape[0], offset=1)
        graph = torch.zeros((global_junctions.shape[0],global_junctions.shape[0])).cuda()
        enumerated_indices = torch.stack((ix,iy),dim=-1)

        tspan = torch.linspace(0,1,num_samples_per_line).reshape(1,-1,1).cuda()
        for split in tqdm(torch.split(enumerated_indices,1024)):
            lines3d = global_junctions[split]
            
            points3d = lines3d[:,:1]*tspan + lines3d[:,1:]*(1-tspan)
            points3d_flt = points3d.reshape(-1,3)

            torch.cuda.empty_cache()
            sdf_, feats_, grads_ = model.implicit_network.get_outputs(points3d_flt)
            sdf = sdf_.reshape(-1,num_samples_per_line).detach()
            is_small = sdf.abs()<sample_sdf_th
            cnt = is_small.sum(dim=-1)/num_samples_per_line
            graph[split[:,0],split[:,1]] += (cnt>sample_consis_th).float()

        self.global_junctions = global_junctions
        self.wireframe_graph = graph

    def get_lines(self, min_num_visibility = 1):
        lines3d_all = self.global_junctions[(self.wireframe_graph>=min_num_visibility).nonzero()]
        return lines3d_all

    def grouping(self, min_num_visibility = 1, distance_threshold = 25):
        device = 'cuda'

        lines3d_all = self.get_lines(min_num_visibility)

        visibility = torch.zeros((lines3d_all.shape[0],),device=device)
        lines3d_visibility = torch.zeros((lines3d_all.shape[0],len(self.dataloader)),device=device, dtype=torch.bool)

        for indices, model_input, ground_truth in tqdm(self.dataloader):
            lines2d_gt = model_input['wireframe'][0].line_segments(0.05)
            model_input["intrinsics"] = model_input["intrinsics"].to(device=device)#.cuda()
            model_input['pose'] = model_input['pose'].to(device=device)
            K = model_input["intrinsics"][0,:3,:3]
            proj_mat = model_input['pose'][0].inverse()[:3]
            R = proj_mat[:,:3]
            T = proj_mat[:,3:]

            lines2d_all = self.model.project2D(K,R,T,lines3d_all).reshape(-1,4)
            lines2d_gt = lines2d_gt.to(device=device)
            dis1 = torch.sum((lines2d_all[:,None]-lines2d_gt[None,:,[0,1,2,3]])**2,dim=-1)
            dis2 = torch.sum((lines2d_all[:,None]-lines2d_gt[None,:,[2,3,0,1]])**2,dim=-1)
            dis = torch.min(dis1,dis2)
            # dis = torch.tensor(dis,device='cuda')
            # rgb = ground_truth['rgb'][0].reshape(*eval_dataset.img_res,3)
            
            mindis = dis.min(dim=1)[0]
            visibility[mindis<distance_threshold] += 1
            lines3d_visibility[mindis<distance_threshold,indices[0]] = True
        
        lines3d_visible = lines3d_all[visibility>=min_num_visibility]
        return lines3d_all, lines3d_visible

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='../exps', help='The experiments folder name.')
    parser.add_argument('--evals_folder', type=str, default='../evals', help='The evaluation folder name.')
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
    
    recon = WireframeRecon(conf=opt.conf,
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

    options = {
        'default': {
            'run': {
                'sdf_refine': 1,
                'reproj_th': 25,
                'num_samples_per_line': 32,
                'sample_sdf_th': 0.01,
                'sample_consis_th': 0.5,
                'min_num_visibility': 1,
            },
            'grouping': {
                'min_num_visibility': 1,
                'distance_threshold': 25,
            }
        },
        # 'no_sdf_refine': {
        #      'run': {
        #         'sdf_refine': 0,
        #         'reproj_th': 10,
        #         'num_samples_per_line': 32,
        #         'sample_sdf_th': 0.01,
        #         'sample_consis_th': 0.9,
        #         'min_num_visibility': 1,
        #     },
        #     'grouping': {
        #         'min_num_visibility': 1,
        #         'distance_threshold': 25,
        #     }
        # },
        
    }

    for run_name, option in options.items():
        recon.run(**option['run'])
        lines3d_all, lines3d_visible = recon.grouping(**option['grouping'])

        outpath = os.path.join(recon.outdir, 'lines3d_{}_all.npz'.format(run_name))
        np.savez(outpath,lines3d = lines3d_all.cpu().numpy())
        outpath = os.path.join(recon.outdir, 'lines3d_{}_vis.npz'.format(run_name))
        np.savez(outpath,lines3d = lines3d_visible.cpu().numpy())

        print('results of {} saved to {}'.format(run_name, outpath))
        
    # recon.run(sdf_refine=True, reproj_th=10, num_samples_per_line=32, sample_sdf_th=0.01, sample_consis_th=0.9)
    # lines3d_all, lines3d_visible = recon.grouping(min_num_visibility=1, distance_threshold=25)


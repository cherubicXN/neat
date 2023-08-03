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
import utils.plots as plots
from utils import rend_util
import matplotlib.pyplot as plt 

from collections import defaultdict


def run(**kwargs):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']
    eval_rendering = kwargs['eval_rendering']

    expname = conf.get_string('train.expname') + kwargs['expname']
    scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else conf.get_int('dataset.scan_id', default=-1)
    if scan_id != -1:
        expname = expname + '_{0}'.format(scan_id)

    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            # self.timestamp = sorted(timestamps)[-1]
            timestamp = None
            for t in sorted(timestamps):
                if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname, t, 'checkpoints',
                                               'ModelParameters', str(kwargs['checkpoint']) + ".pth")):
                    timestamp = t
            if timestamp is None:
                print('NO GOOD TIMSTAMP')
                exit()
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    evaldir = os.path.join('../', evals_folder_name, expname)
    outdir = os.path.join('../', evals_folder_name, expname,'outputs')
    utils.mkdir_ifnotexists(evaldir)
    utils.mkdir_ifnotexists(outdir)

    dataset_conf = conf.get_config('dataset')
    if kwargs['scan_id'] != -1:
        dataset_conf['scan_id'] = kwargs['scan_id']
    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)

    conf_model = conf.get_config('model')
    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf_model)
    if torch.cuda.is_available():
        model.cuda()


    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )
    
    total_pixels = eval_dataset.total_pixels
    img_res = eval_dataset.img_res
    split_n_pixels = conf.get_int('train.split_n_pixels', 10000)

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
    model.load_state_dict(saved_model_state["model_state_dict"])
    epoch = saved_model_state['epoch']

    import trimesh
    lines_3d = []
    for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
        # model_input["intrinsics"] = model_input["intrinsics"].cuda()
        # model_input["uv"] = model_input["uv"].cuda()
        # model_input['pose'] = model_input['pose'].cuda()

        wireframe = model_input['wireframe']

        patch = np.meshgrid(
            np.linspace(-1, 1,3),
            np.linspace(-1, 1,3),
            indexing='xy')
        patch = np.stack(patch,axis=-1).reshape(-1,2)
        patch = torch.from_numpy(patch).float()

        junctions = wireframe[0].vertices.clone()
        # points = (junctions[:,None]+patch[None])
        # points_flt = points.reshape(1,-1,2)
        points = junctions.reshape(1,-1,2)

        inp = {
            'intrinsics': model_input['intrinsics'].cuda(),
            'pose': model_input['pose'].cuda(),
            'uv': points.cuda(),
        }

        out = defaultdict(list)
        split = utils.split_input(inp, points.shape[1], n_pixels=split_n_pixels)
        for s in tqdm(split):
            torch.cuda.empty_cache()
            out_ = model(s)
            for key, val in out_.items():
                out[key].append(val.detach())

        for key in out.keys():
            out[key] = torch.cat(out[key],dim=0)
        # depth = out['depth'].reshape(points.shape[:-1])
        depth = out['depth']
        # mind, argd = depth.cpu().min(dim=-1)
        # junctions_adjusted = points[torch.arange(points.shape[0]),argd.cpu()]
        mask_d = depth<3.0

        edges = wireframe[0].edges[wireframe[0].weights>0.5]
        L2d = junctions[edges].reshape(-1,4).numpy()

        mask_L = mask_d[edges].all(dim=-1).cpu().numpy()
        # plt.imshow(ground_truth['rgb'].reshape(*img_res,-1))
        # plt.plot([L2d[:,0],L2d[:,2]],[L2d[:,1],L2d[:,3]],'r-')
        # plt.show()

        # X = out['xyz'].reshape(*points.shape[:-1],3)
        rgb = ground_truth['rgb'].reshape(*img_res,-1)

        pose = model_input['pose']
        K = model_input['intrinsics'][0,:3,:3].numpy()
        R = pose[0].inverse().cpu().numpy()[:3,:3]
        T = pose[0].inverse().cpu().numpy()[:3,3]

        X = out['xyz'].cpu().numpy()

        plt.imshow(rgb)
        x_rep = K@(R@X.transpose()+T[:,None])
        x_rep = (x_rep/x_rep[-1:]).transpose()

        plt.plot(x_rep[:,0],x_rep[:,1],'r.')
        plt.show()
        import pdb; pdb.set_trace()
        # X = X[torch.arange(points.shape[0]),argd.cpu()].cpu().numpy()
        L = X[edges][mask_L]
        
        lines_3d.append(trimesh.load_path(L))

        # X = out['xyz'].detach().cpu().numpy()

        # L = trimesh.load_path(X[edges.numpy()])
        # L2d = junctions[edges].reshape(-1,4).numpy()
        # lines_3d.append(L)

        # inp_image = {
        #     'intrinsics': model_input['intrinsics'].cuda(),
        #     'pose': model_input['pose'].cuda(),
        #     'uv': model_input['uv'].cuda()
        # }

        # split = utils.split_input(inp_image, total_pixels, n_pixels=split_n_pixels)
        # res = defaultdict(list)
        # for s in tqdm(split):
        #     torch.cuda.empty_cache()
        #     out = model(s)
        #     for key, val in out.items():
        #         res[key].append(val.detach())
        
        # for key in res.keys():
        #     res[key] = torch.cat(res[key])

    
    trimesh.Scene(lines_3d).show()
    import pdb; pdb.set_trace()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser = argparse.ArgumentParser()

    parser.add_argument('--conf', type=str, default='./confs/dtu.conf')
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--evals_folder', type=str, default='evals', help='The evaluation folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--eval_rendering', default=False, action="store_true", help='If set, evaluate rendering quality.')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    run(
        conf=opt.conf,
        expname=opt.expname,
        exps_folder_name=opt.exps_folder,
        evals_folder_name=opt.evals_folder,
        timestamp=opt.timestamp,
        checkpoint=opt.checkpoint,
        scan_id=opt.scan_id,
        resolution=opt.resolution,
        eval_rendering=opt.eval_rendering,
    )
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

def evaluate(**kwargs):
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
    # else:
        # scan_id = conf.get_string('dataset.object', default='')

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
    utils.mkdir_ifnotexists(evaldir)

    dataset_conf = conf.get_config('dataset')
    if kwargs['scan_id'] != -1:
        dataset_conf['scan_id'] = kwargs['scan_id']
    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)

    conf_model = conf.get_config('model')
    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf_model)
    if torch.cuda.is_available():
        model.cuda()

    # settings for camera optimization
    scale_mat = eval_dataset.get_scale_mat()

    if eval_rendering:
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

    ####################################################################################################################
    print("evaluating...")

    model.eval()

    if eval_rendering:
        images_dir = '{0}/rendering_{1}'.format(evaldir, epoch)
        utils.mkdir_ifnotexists(images_dir)

        psnrs = []
        for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input['pose'] = model_input['pose'].cuda()

            split = utils.split_input(model_input, total_pixels, n_pixels=split_n_pixels)
            res = []
            for s in tqdm(split):
                torch.cuda.empty_cache()
                out = model(s)
                res.append({
                    'rgb_values': out['rgb_values'].detach(),
                })

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, total_pixels, batch_size)
            rgb_eval = model_outputs['rgb_values']
            rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)

            rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
            img.save('{0}/eval_{1}.png'.format(images_dir,'%03d' % indices[0]))

            psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                      ground_truth['rgb'].cuda().reshape(-1, 3)).item()
            psnrs.append(psnr)


        psnrs = np.array(psnrs).astype(np.float64)
        print("RENDERING EVALUATION {2}: psnr mean = {0} ; psnr std = {1}".format("%.2f" % psnrs.mean(), "%.2f" % psnrs.std(), scan_id))
        psnrs = np.concatenate([psnrs, psnrs.mean()[None], psnrs.std()[None]])
        pd.DataFrame(psnrs).to_csv('{0}/psnr_{1}.csv'.format(evaldir, epoch))
        
    with torch.no_grad():
        
        if scan_id < 24: # Blended MVS
            mesh = plt.get_surface_high_res_mesh(
                sdf=lambda x: model.implicit_network(x)[:, 0],
                resolution=kwargs['resolution'],
                grid_boundary=conf.get_list('plot.grid_boundary'),
                level=conf.get_int('plot.level', default=0),
                take_components = type(scan_id) is not str
            )
        else: # DTU
            bb_dict = np.load('../data/DTU/bbs.npz')
            grid_params = bb_dict[str(scan_id)]
            mesh = plt.get_surface_by_grid(
                grid_params=grid_params,
                sdf=lambda x: model.implicit_network(x)[:, 0],
                resolution=kwargs['resolution'],
                level=conf.get_int('plot.level', default=0),
                higher_res=True
            )
        # Transform to world coordinates
        mesh.apply_transform(scale_mat)

        # Taking the biggest connected component
        components = mesh.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float32)
        mesh_clean = components[areas.argmax()]

        mesh_folder = '{0}/{1}'.format(evaldir, epoch)
        utils.mkdir_ifnotexists(mesh_folder)
        mesh_clean.export('{0}/scan{1}.ply'.format(mesh_folder, scan_id), 'ply')
    
    mesh_path = '{0}/scan{1}.ply'.format(mesh_folder, scan_id)

    cmd_line = 'python evaluation/eval-dtu.py --data {} --scan {} --mode mesh --dataset_dir ~/datasets/DTU'.format(mesh_path,opt.scan_id)

    os.system(cmd_line)

    



if __name__ == '__main__':

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

    evaluate(conf=opt.conf,
             expname=opt.expname,
             exps_folder_name=opt.exps_folder,
             evals_folder_name=opt.evals_folder,
             timestamp=opt.timestamp,
             checkpoint=opt.checkpoint,
             scan_id=opt.scan_id,
             resolution=opt.resolution,
             eval_rendering=opt.eval_rendering,
             )

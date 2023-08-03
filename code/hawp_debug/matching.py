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
import trimesh
import cv2
import kornia

def get_cam(c2w_):
    if isinstance(c2w_, torch.Tensor):
        c2w = c2w_.cpu().numpy()
    else:
        c2w = c2w_

    center = c2w[:3,3]
    x = c2w[:3,0]*0.1
    y = c2w[:3,1]*0.1
    z = c2w[:3,2]*0.1

    tz = trimesh.load_path(np.stack((center,center+z),axis=0))
    tz.colors = np.array([[255,0,0]])
    ty = trimesh.load_path(np.stack((center,center+y),axis=0))
    ty.colors = np.array([[0,255,0]])
    tx = trimesh.load_path(np.stack((center,center+x),axis=0))
    tx.colors = np.array([[0,0,255]])

    cam = trimesh.Scene([tx,ty,tz])

    
    return cam

def rgb2gray(tensor):
    assert isinstance(tensor,torch.Tensor)
    device = tensor.device
    nparray = np.array(tensor.cpu()*255,dtype=np.uint8)
    gray = cv2.cvtColor(nparray,cv2.COLOR_RGB2GRAY)
    gray = np.array(gray,dtype=np.float32)/255.0
    return torch.tensor(gray,device=device)

def patch_sample(grayim, points, w, h):
    device = grayim.device
    xm,ym = torch.meshgrid(torch.arange(-16,16),torch.arange(-16,16),indexing='xy')
    xm = xm.reshape(1,-1).to(device)
    ym = ym.reshape(1,-1).to(device)
    
    xl = points.long()[:,None,0] + xm
    yl = points.long()[:,None,1] + ym
    
    mask = (xl>=0)*(xl<w)*(yl>=0)*(yl<h)
    
    xl = xl.clamp(0,w-1)
    yl = yl.clamp(0,h-1)
    
    patches = grayim[yl,xl]
#     return torch.stack((xl,yl),dim=-1)
#     print(xl)
    return patches.reshape(-1,32,32)

def twoView3D(points0, points1, desc0, desc1, K0, K1, P0w, P1w, img0, img1, HyNet):
    device = points0.device
    I = torch.eye(3)[None].to(device)
    O = torch.zeros((1,3,1)).to(device)
    
    R, T = kornia.geometry.epipolar.relative_camera_motion(P0w[:,:3,:3],P0w[:,:3,3:],P1w[:,:3,:3],P1w[:,:3,3:])
    E = kornia.geometry.epipolar.essential_from_Rt(I,O,R,T)
    F = kornia.geometry.epipolar.fundamental_from_essential(E, K0, K1)

    # gray0 = rgb2gray(img0)
    # gray1 = rgb2gray(img1)

    # patches0 = patch_sample(gray0, points0, img0.shape[1], img0.shape[0])
    # patches1 = patch_sample(gray1, points1, img1.shape[1], img1.shape[0])

    # with torch.no_grad():
        # desc0 = HyNet(patches0[:,None])
        # desc1 = HyNet(patches1[:,None])

    ddis = torch.sum((desc0[:,None]-desc1[None])**2,dim=-1)

    mdis, midx = ddis.topk(k=2,largest=False,dim=1)

    dmask = (mdis[:,0]<mdis[:,1]*0.75)
    
    x0 = points0[dmask]
    x1 = points1[midx[dmask,0]]

    epiline0 = kornia.geometry.epipolar.compute_correspond_epilines(x0[None], F)[0]
    epiline1 = kornia.geometry.epipolar.compute_correspond_epilines(x1[None], F.transpose(1,2))[0]

    emask = kornia.geometry.epipolar.symmetrical_epipolar_distance(x0[None], x1[None], F, squared=False)<5
    x0e = x0[emask[0]]
    x1e = x1[emask[0]]

    if emask.sum() == 0:
        return None
    X = kornia.geometry.epipolar.triangulate_points(
        K0@torch.cat((I,O),dim=-1),
        K1@torch.cat((R,T),dim=-1),
        x0e[None],
        x1e[None]
    )[0]

    return (X, x0e, x1e)

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

    # if kwargs['timestamp'] == 'latest':
    #     if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname)):
    #         timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], expname))
    #         if (len(timestamps)) == 0:
    #             print('WRONG EXP FOLDER')
    #             exit()
    #         # self.timestamp = sorted(timestamps)[-1]
    #         timestamp = None
    #         for t in sorted(timestamps):
    #             if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname, t, 'checkpoints',
    #                                            'ModelParameters', str(kwargs['checkpoint']) + ".pth")):
    #                 timestamp = t
    #         if timestamp is None:
    #             print('NO GOOD TIMSTAMP')
    #             exit()
    #     else:
    #         print('WRONG EXP FOLDER')
    #         exit()
    # else:
    #     timestamp = kwargs['timestamp']

    # utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    # expdir = os.path.join('../', exps_folder_name, expname)
    evaldir = os.path.join('../', evals_folder_name, expname)
    # outdir = os.path.join('../', evals_folder_name, expname,'outputs')
    utils.mkdir_ifnotexists(evaldir)
    # utils.mkdir_ifnotexists(outdir)

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

    # old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    # saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
    # model.load_state_dict(saved_model_state["model_state_dict"])
    # epoch = saved_model_state['epoch']

    # lines_3d = []
    wireframes = []

    poses = []
    K = []
    rgb = []

    cameras = []

    descriptors = []

    HyNet = kornia.feature.HyNet(pretrained=True)
    HyNet = HyNet.cuda()
    for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
        # model_input["intrinsics"] = model_input["intrinsics"].cuda()
        # model_input["uv"] = model_input["uv"].cuda()
        # model_input['pose'] = model_input['pose'].cuda()
        rgb.append(ground_truth['rgb'].reshape(*img_res,-1))
        wireframe = model_input['wireframe']

        wireframes.append(wireframe[0])
        poses.append(model_input['pose'])
        K.append(model_input['intrinsics'])

        cameras.append(get_cam(model_input['pose'][0]))

        gray = rgb2gray(rgb[-1])
        patches = patch_sample(gray.cuda(), wireframe[0].vertices.cuda(), img_res[1], img_res[0])
        with torch.no_grad():
            desc = HyNet(patches[:,None])
        descriptors.append(desc.cpu())

    projections = [p.inverse()[:,:3] for p in poses]

    # mesh = trimesh.load_mesh('/home/xn/repo/hawp2vision/volsdf/exps/dtu_24/2022_09_08_14_12_26/plots/surface_2000.ply')
    points3d = [] 
    lines3d = []
    HyNet = HyNet.cuda()
    for i in tqdm(range(len(wireframes))):
    # for i in range(0,1):
        max_p = i+1
        max_v = 0
        lines3d_i = []
        points3d_i = []
        for j in range(0,len(wireframes)):
            if j == i:
                continue
        # for j in range(i+1,2):
            out = twoView3D(wireframes[i].vertices.cuda(),
                          wireframes[j].vertices.cuda(), 
                          descriptors[i].cuda(),
                          descriptors[j].cuda(),
                          K[i][:,:3,:3].cuda(), 
                          K[j][:,:3,:3].cuda(), 
                          projections[i][:,:3,:].cuda(), 
                          projections[j][:,:3,:].cuda(), 
                          rgb[i].cuda(), 
                          rgb[j].cuda(), 
                          HyNet)
            if out is None:
                continue
            X, x0, x1 = out
            if X.shape[0]<5:
                continue
            x0 = x0.cpu()
            x1 = x1.cpu()
            X = X.cpu()
            Xw = (poses[i][0,:3,:3]@X.t()+poses[i][0,:3,3:]).t()
            mask = Xw.abs().max(dim=-1)[0]<3
            if mask.sum() == 0:
                continue
            x0 = x0[mask]
            x1 = x1[mask]
            X = X[mask]
            Xw = Xw[mask]
            dis0 = torch.sum((x0[:,None] - wireframes[i].vertices[None])**2,dim=-1)
            dis1 = torch.sum((x1[:,None] - wireframes[j].vertices[None])**2,dim=-1)

            idx0 = (dis0<1e-3).nonzero()
            idx1 = (dis1<1e-3).nonzero()
            map0 = torch.zeros(wireframes[i].vertices.shape[0],dtype=torch.long)-1
            map1 = torch.zeros(wireframes[j].vertices.shape[0],dtype=torch.long)-1

            map0[idx0[:,-1]] = idx0[:,0]
            map1[idx1[:,-1]] = idx1[:,0]

            edge0 = wireframes[i].edges[wireframes[i].weights>0.1]
        
            edge1 = wireframes[j].edges[wireframes[j].weights>0.1]

            edge_mask0 = (map0[edge0]>=0).all(dim=-1)
            edge_mask1 = (map1[edge1]>=0).all(dim=-1)

            edge0_valid = edge0[edge_mask0]
            edge1_valid = edge1[edge_mask1]

            graph = torch.zeros(Xw.shape[0],Xw.shape[0])

            edge0_matched = map0[edge0_valid]
            edge1_matched = map1[edge1_valid]

            graph[edge0_matched[:,0],edge0_matched[:,1]] = 1
            graph[edge0_matched[:,1],edge0_matched[:,0]] = 1
            graph[edge1_matched[:,0],edge1_matched[:,1]] += 1
            graph[edge1_matched[:,1],edge1_matched[:,0]] += 1
            lines = Xw[(graph.triu()>1).nonzero()]
            lines2d = x0[(graph.triu()>1).nonzero()].numpy()
            # plt.imshow(rgb[i])
            # plt.plot([lines2d[:,0,0],lines2d[:,1,0]],[lines2d[:,0,1],lines2d[:,1,1]],'r-')
            # plt.show()
            # import pdb; pdb.set_trace()
            points3d_i.append(Xw) 
            lines3d_i.append(lines)

        if len(lines3d_i)==0:
            continue
        num_lines = torch.tensor([x.shape[0] for x in lines3d_i])

        maxv, maxi = num_lines.max(dim=0)

        lines3d.append(lines3d_i[maxi])
        points3d.append(points3d_i[maxi])
        # lines3d = [lines3d[max_p]]
    
    # points3d_all = torch.cat(points3d)
    # mask = (points3d_all.abs()<3).all(dim=-1)
    # tm_points = trimesh.points.PointCloud(points3d_all[mask].numpy())
    # scene = trimesh.Scene([tm_points,cameras,mesh])
    lines3d = np.array([l.numpy() for l in lines3d ],dtype=object)
    wireframe_dir = os.path.join(evaldir,'wireframes')
    os.makedirs(wireframe_dir,exist_ok=True)
    line_path = os.path.join(wireframe_dir,'matching.npz')
    print(line_path)
    np.savez(line_path,lines3d=lines3d)
def verifyByDensity(lines3d_all, model):
    lines_lambda = torch.linspace(0, 1,64).reshape(1,-1,1).repeat(lines3d_all.shape[0],1,1)
    lines_length = torch.sum((lines3d_all[:,0]-lines3d_all[:,1])**2,dim=-1,keepdim=True)
    lines_points =  (lines3d_all[:,:1]-lines3d_all[:,1:])*lines_lambda + lines3d_all[:,:1]

    sdf = model.implicit_network.get_sdf_vals(lines_points.reshape(-1,3).cuda()).reshape(lines_points.shape[:-1]).detach()
    sigma = model.density.density_func(sdf).detach()
    free_energy = 1/63*sigma
    scores = torch.exp(-sigma).mean(dim=-1)

    return scores


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
    parser.add_argument('--dest', default=None, type=str, help='destination to save the 3d lines')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    print(opt)
    
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
        dest=opt.dest,
    )
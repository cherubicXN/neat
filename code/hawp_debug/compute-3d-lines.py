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


def twoView3D(points0, points1, 
              desc0, desc1, 
              K0, K1, P0w, P1w, 
              img0, img1, HyNet, 
              match_ratio = 0.75):
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

    dmask = (mdis[:,0]<mdis[:,1]*match_ratio)
    
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

    x0r = (K0[0]@X.t()).t()
    x0r = x0r[:,:2]/x0r[:,2:]
    x1r = (K1[0]@(R[0]@X.t()+T[0])).t()
    x1r = x1r[:,:2]/x1r[:,2:]

    return (X, x0e, x1e, x0r, x1r)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--conf',type=str, required=True)
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--dest', type=str, required=True)

    opt = parser.parse_args()


    conf = ConfigFactory().parse_file(opt.conf)

    scan_id = opt.scan_id if opt.scan_id!=-1 else conf.get_int('dataset.scan_id',default=-1)
    dataset_conf = conf.get_config('dataset')

    if scan_id != -1:
        dataset_conf['scan_id'] = scan_id
    
    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )
    
    total_pixels = eval_dataset.total_pixels
    img_res = eval_dataset.img_res


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

    points3d = [] 
    points2d  = []
    edges = []
    view_pairs = []
    
    HyNet = HyNet.cuda()

    num_views = len(wireframes)

    for i in tqdm(range(num_views)):
        for j in range(num_views):
            if j == i:
                continue
            
            if (i,j) in view_pairs or (j,i) in view_pairs:
                continue

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
                        HyNet,
                        match_ratio=0.75
                        )
            
            if out is None:
                continue
            X, x0, x1, x0r, x1r = out
            if X.shape[0]<5:
                continue

            x0 = x0.cpu()
            x1 = x1.cpu()
            x0r = x0r.cpu()
            x1r = x1r.cpu()
            X = X.cpu()
            Xw = (poses[i][0,:3,:3]@X.t()+poses[i][0,:3,3:]).t()
            mask = Xw.abs().max(dim=-1)[0]<3
            
            if mask.sum() == 0:
                continue

            x0 = x0[mask]
            x1 = x1[mask]
            x0r = x0r[mask]
            x1r = x1r[mask]

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

            edge0 = wireframes[i].edges[wireframes[i].weights>0.05]
        
            edge1 = wireframes[j].edges[wireframes[j].weights>0.05]

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
            lines2d = x0r[(graph.triu()>1).nonzero()].numpy()

            if lines.shape[0]<10:
                continue
            view_pairs.append((i,j))

            points3d.append(Xw.numpy())
            points2d.append((x0r.numpy(),x1r.numpy()))
            edges.append((graph.triu()>1).nonzero().numpy())
     
    view_pairs = np.array(view_pairs)
    edges = np.array(edges,dtype=object)
    points2d = np.array(points2d,dtype=object)
    points3d = np.array(points3d,dtype=object)
    
    np.savez(opt.dest,
        view_pairs = view_pairs,
        edges = edges,
        junctions_2D = points2d,
        junctions_3D = points3d,
    )


    # for id, (vi,vj) in enumerate(view_pairs):

    #     points_i = points2d[id][0]
    #     points_j = points2d[id][1]

    #     edge_ids = edges[id]

    #     lines2d_i = points_i[edge_ids]

    #     lines2d_j = points_j[edge_ids]

        # fig, axes = plt.subplots(1,2,figsize=(18,12))
        # axes[0].imshow(rgb[vi])
        # axes[0].plot(
        #     [lines2d_i[:,0,0],lines2d_i[:,1,0]],
        #     [lines2d_i[:,0,1],lines2d_i[:,1,1]],
        #     'r-'
        #     )

        # axes[1].imshow(rgb[vj])
        # axes[1].plot(
        #     [lines2d_j[:,0,0],lines2d_j[:,1,0]],
        #     [lines2d_j[:,0,1],lines2d_j[:,1,1]],
        #     'r-'
        #     )

        # plt.show()
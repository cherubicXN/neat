import sys
sys.path.append('../code')
import argparse
import GPUtil
import os
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import scipy.io as sio
import sklearn.neighbors as skln
from scipy.io import loadmat

def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

def evaluate(pcd_pred, mesh_trgt, threshold=.05, down_sample=.02):
    # pcd_pred = o3d.geometry.PointCloud(pcd_pred.vertices)
    pcd_trgt = o3d.geometry.PointCloud(mesh_trgt.vertices)

    if down_sample:
        # pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)
    dist1 = nn_correspondance(verts_pred, verts_trgt)
    dist2 = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
    }
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,required=True,help='the path of the reconstructed wireframe model')
    parser.add_argument('--scan', type=str, required=True)
    # parser.add_argument('--cam', type=str,required=True,help='the path of cam')
    parser.add_argument('--threshold', type=float, default=1., help='dist to surface threshold')
    parser.add_argument('--dataset_dir', type=str, default='/home/xn/repo/hawp2vision/volsdf/data/scannet')
    parser.add_argument('--downsample', type=float, default=0.02)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=20)

    opt = parser.parse_args()

    # camera = np.load(opt.cam)
    if opt.scan == '0084_00':
        scale= 0.44963
        offset= np.array([1.23815, 2.57319, 1.38001]).reshape(1, 3)
    elif opt.scan == '0616_00':
        scale = 0.38626
        offset = np.array([2.84253, 2.14299, 1.38729]).reshape(1, 3)
    else:
        raise NotImplementedError

    data = np.load(opt.data,allow_pickle=True)
    lines3d = data['lines3d']
    if lines3d.dtype == object:
        lines3d = np.concatenate(lines3d,axis=0)
    t = np.linspace(0, 1, 32).reshape(1,-1,1)
    lines3d =  (lines3d[:,:1]*t)+(lines3d[:,1:]*(1-t))

    lines3d = np.concatenate(lines3d,axis=0).reshape(-1, 3)
    N = lines3d.shape[0]
    lines3d = lines3d / scale 
    lines3d = lines3d + offset
    
    # np.savetxt('/home/xn/repo/hawp2vision/volsdf/code/groundtruth/tmp.txt', lines3d)

    data_pcd = lines3d
    data_pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lines3d))

    mesh_gt = o3d.io.read_triangle_mesh(f'{opt.dataset_dir}/{opt.scan}/gt.obj')

    res = evaluate(data_pcd_o3d, mesh_gt, down_sample=opt.downsample)

    for k, v in res.items():
        print(f'{k:7s}: {v:1.3f}')


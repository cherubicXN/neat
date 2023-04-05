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


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='the path of the reconstructed wireframe model')
    parser.add_argument('--stl', type=str, required=True, help='the path of stl')
    parser.add_argument('--cam', type=str, required=True, help='the path of cam')
    parser.add_argument('--score', type=float, default=None)
    parser.add_argument('--threshold', type=float, default=1., help='dist to surface threshold')
    parser.add_argument('--dataset_dir', type=str, default='/home/xn/datasets/DTU')
    parser.add_argument('--downsample_density', type=float, default=0.2)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=20)
    parser.add_argument('--noscale', default=False, action='store_true')

    opt = parser.parse_args()
    thresh = opt.downsample_density

    camera = np.load(opt.cam)
    global_scale_mat = camera['scale_mat_0']

    if opt.noscale:
        global_scale_mat = np.eye(4)

    data = np.load(opt.data, allow_pickle=True)

    lines3d = data['lines3d']
    if lines3d.dtype == object:
        lines3d = np.concatenate(lines3d, axis=0)
    if opt.score is not None:
        scores = data['scores']
        lines3d = lines3d[scores < opt.score]
    t = np.linspace(0, 1, 32).reshape(1, -1, 1)

    lines3d = (lines3d[:, :1] * t) + (lines3d[:, 1:] * (1 - t))

    lines3d = np.concatenate(lines3d, axis=0).reshape(-1, 3)
    N = lines3d.shape[0]

    lines3d = global_scale_mat @ (np.concatenate([lines3d, np.ones([N, 1])], axis=-1).T)  # 4, N
    lines3d = lines3d[:3].transpose(1, 0)  # N, 3

    data_pcd = lines3d
    data_pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lines3d))

    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    # data_down = data_pcd[mask]
    print("Note: use all line pts")
    data_down = data_pcd

    stl_pcd = o3d.io.read_point_cloud(f'{opt.stl}')
    stl = np.asarray(stl_pcd.points)
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_down, n_neighbors=1, return_distance=True)
    max_dist = opt.max_dist
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    nn_engine.fit(data_down)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    print('ACC: {}\t COMP: {}'.format(mean_d2s, mean_s2d))


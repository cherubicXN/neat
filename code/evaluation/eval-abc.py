import trimesh
import numpy as np
import torch
import argparse
import os
import os.path as osp
import json
from scipy.optimize import linear_sum_assignment
import open3d as o3d
from sklearn.neighbors import KDTree
import scipy.io as sio
import sklearn.neighbors as skln
from scipy.io import loadmat
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,required=True,help='the path of the reconstructed wireframe model')
    parser.add_argument('--scan', type=str,required=True,help='the path of the scan dir')

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_args()

    data = torch.load(opt.data)

    with open(osp.join(opt.scan,'lines.json')) as f:
        wireframe_gt = json.load(f)
    junctions_gt = np.array(wireframe_gt['junctions'])

    with open(osp.join(opt.scan,'offset_scale.txt')) as f:
        offset_scale = f.read().split()

    scale_mat = np.array([[1/float(offset_scale[-1]),0,0,-float(offset_scale[0])],
                          [0,1/float(offset_scale[-1]),0,-float(offset_scale[1])],
                          [0,0,1/float(offset_scale[-1]),-float(offset_scale[2])],
                          [0,0,0,1]])


    junctions_pred = data['junctions3d_initial'].cpu().numpy()
    junctions_pred_scaled = (junctions_pred@scale_mat[:3,:3].T)+scale_mat[:3,3]
     
    cdist = np.linalg.norm(junctions_pred_scaled[:,None]-junctions_gt[None],axis=-1)
    global_scale = scale_mat[0,0]

    assign = linear_sum_assignment(cdist)
    cost = cdist[assign]

    # nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=0.05*global_scale, algorithm='kd_tree', n_jobs=-1)
    # nn_engine.fit(junctions_gt)
    # dist_d2s, idx_d2s = nn_engine.kneighbors(junctions_pred_scaled)
    # nn_engine.fit(junctions_pred_scaled)
    # dist_s2d, idx_s2d = nn_engine.kneighbors(junctions_gt)
    # # import pdb; pdb.set_trace()
    thresholds = [0.01, 0.02, 0.05]

    junctions_precision = []
    junctions_recall = []
    for threshold in thresholds:
        num_correct = (cost<threshold*global_scale).sum()
        recall = num_correct/junctions_gt.shape[0]
        precision = num_correct/junctions_pred.shape[0]
        # f1 = 2*precision*recall/(precision+recall)
        junctions_precision.append(precision)
        junctions_recall.append(recall)
        # print('threshold: ', threshold)
        # print('recall: ', recall)
        # print('precision: ', precision)
        # print('f1: ', f1)
        # print('num correct: ', num_correct)
        # print('num gt: ', junctions_gt.shape[0])
        # print('num pred: ', junctions_pred.shape[0])
        # print('')

    edges = np.array(wireframe_gt['lines'])
    lines3d_gt = junctions_gt[edges]

    lines3d_pred = data['lines3d_wfi_checked'].cpu().numpy()
    temp = lines3d_pred.reshape(-1,3)

    lines3d_pred_scaled = (temp@scale_mat[:3,:3].T)+scale_mat[:3,3]
    lines3d_pred_scaled = lines3d_pred_scaled.reshape(-1,2,3)



    cmat1 = np.linalg.norm(lines3d_pred_scaled[:,None,:]-lines3d_gt[None,:,:],axis=-1).mean(axis=-1)
    cmat2 = np.linalg.norm(lines3d_pred_scaled[:,None,:]-lines3d_gt[None,:,[1,0]],axis=-1).mean(axis=-1)
    cdist = np.minimum(cmat1,cmat2)
    assign = linear_sum_assignment(cdist)
    cost = cdist[assign]

    thresholds = [0.01, 0.02, 0.05]
    lines_precision = []
    lines_recall = []

    # import pdb; pdb.set_trace()

    for threshold in thresholds:
        num_correct = (cost<threshold*global_scale).sum()
        recall = num_correct/lines3d_gt.shape[0]
        precision = num_correct/lines3d_pred.shape[0]
        lines_precision.append(precision)
        lines_recall.append(recall)
        # f1 = 2*precision*recall/(precision+recall)
        # print('threshold: ', threshold)
        # print('recall: ', recall)
        # print('precision: ', precision)
        # print('f1: ', f1)
        # print('num correct: ', num_correct)
        # print('num gt: ', junctions_gt.shape[0])
        # print('num pred: ', junctions_pred.shape[0])
        # print('')

    latex_junctions = []
    # for jp, jr in zip(junctions_precision, junctions_recall):
    for jp in junctions_precision + junctions_recall:
        latex_junctions.append('{:.3f}'.format(jp))
    latex_junctions = ' & '.join(latex_junctions)
    print(latex_junctions)

    latex_lines = []
    # for lp, lr in zip(lines_precision, lines_recall):
    for lp in lines_precision + lines_recall:
        latex_lines.append('{:.3f}'.format(lp))
    latex_lines = ' & '.join(latex_lines)
    print(latex_lines)
    # p
    import pdb; pdb.set_trace()
if __name__ == "__main__":
    main()

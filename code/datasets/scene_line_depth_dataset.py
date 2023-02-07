import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util
from pathlib import Path
from sslib import WireframeGraph

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 lines_npz,
                 scan_id=0,
                 line_detector = 'hawp',
                 distance_threshold = 5.0,
                 ):

        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))
        # lines3d = np.concatenate(np.load(lines_npz,allow_pickle=True)['lines3d'])
        lines3d = np.load(lines_npz,allow_pickle=True)['lines3d']
        lines3d = [torch.tensor(t) for t in lines3d]
        self.lines3d = torch.cat(lines3d)
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        image_dir = '{0}/image'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = []
        self.wireframes = []
        self.lines = []
        self.distance = distance_threshold
        self.score_threshold = 0.05

        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            hawp_path = Path(self.instance_dir)/line_detector/Path(path).with_suffix('.json').name

            wireframe = WireframeGraph.load_json(hawp_path)
            assert wireframe.frame_height == img_res[0] and wireframe.frame_width ==img_res[1]
            self.wireframes.append(wireframe)
            self.lines.append(wireframe.line_segments(self.score_threshold))

    def __len__(self):
        return self.n_images

    def project2D(self, K,R,T, points3d):
        shape = points3d.shape 
        assert shape[-1] == 3
        X = points3d.reshape(-1,3)
        
        x = K@(R@X.t()+T)
        x = x.t()
        x = x/x[:,-1:]
        x = x.reshape(*shape)[...,:2]
        return x
    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]

            K = self.intrinsics_all[idx][:3,:3]
            R = self.pose_all[idx].inverse()[:3,:3]
            T = self.pose_all[idx].inverse()[:3,3:]
            lines_p = self.project2D(self.intrinsics_all[idx][:3,:3], self.pose_all[idx].inverse()[:3,:3], self.pose_all[idx].inverse()[:3,3:], self.lines3d)
            lines_p = lines_p.reshape(-1,4)

            dis1 = torch.sum((lines_p[:,None] - self.lines[idx][None,:,:-1])**2,dim=-1)
            dis2 = torch.sum((lines_p[:,None] - self.lines[idx][None,:,[2,3,0,1]])**2,dim=-1)
            dis = torch.min(dis1,dis2)
            mindis, minidx = dis.min(dim=0)
            is_avail = mindis<10

            lines3d_sel = self.lines3d[minidx[is_avail]]
            lines3d_cur =  (R@lines3d_sel.reshape(-1,3).t()+T).t().reshape(-1,2,3)
            weight = self.lines[idx][is_avail,-1]
            t = torch.linspace(0,1,32).reshape(1,-1,1)
            pts3d = lines3d_cur[:,:1]*t + lines3d_cur[:,1:]*(1-t)
            weight = weight[:,None].repeat(1,32)
            pts3d = pts3d.reshape(-1,3)

            pts2d = self.project2D(K, torch.eye(3), torch.zeros_like(T), pts3d)
            depth = pts3d[:,-1]
            dw = weight.reshape(-1)
            
            randperm = torch.randperm(pts2d.shape[0])[:len(self.sampling_idx)]
            # sample['pts'] = pts.reshape(-1,3)
            sample['pts'] = pts2d[randperm]
            sample['z'] = depth[randperm]
            ground_truth['depth'] = depth[randperm]
            ground_truth['depth.w'] = dw[randperm]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

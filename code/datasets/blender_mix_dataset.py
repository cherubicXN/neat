import os
import os.path as osp
import torch
import numpy as np
import cv2

import utils.general as utils
from utils import rend_util
from utils import hawp_util

from pathlib import Path
class BlenderDataset(torch.utils.data.Dataset):
    def __init__(self,
        data_dir,
        img_res,
        reverse_coordinate = False):

        self.instance_dir = osp.join('../data', data_dir)

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        image_dir = '{}/images'.format(self.instance_dir)
        image_paths = [f for f in sorted(utils.glob_imgs(image_dir)) if not 'mask' in f]

        self.n_images = len(image_paths)
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)

        camera_dict = np.load(self.cam_file)

        self.intrinsics_all = camera_dict['intrinsics']
        self.intrinsics_all = torch.from_numpy(self.intrinsics_all).float()
        self.pose_all = camera_dict['extrinsics']
        self.pose_all = torch.from_numpy(self.pose_all).float()

        self.rgb_images = []
        # self.masks = []
        # self.hats = []
        self.distance = 10
        self.wireframes = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            # hat_path = Path(self.instance_dir)/'hats'/Path(path).with_suffix('.npz').name
            # hats = np.load(hat_path)
            # self.hats.append(torch.from_numpy(hats['hat'].reshape(-1,4)).float())
            # self.masks.append(torch.from_numpy(hats['mask'].reshape(-1)))

            hawp_path = hat_path = Path(self.instance_dir)/'hawp'/Path(path).with_suffix('.json').name
            wireframe = hawp_util.WireframeGraph.load_json(hawp_path)
            self.wireframes.append(wireframe)

            self.rgb_images.append(torch.from_numpy(rgb).float())

        if reverse_coordinate:
            self.normalization = torch.diag(torch.tensor([1,-1,-1,1])).float()
        else:
            self.normalization = torch.diag(torch.tensor([1,1,1,1])).float()
    def __len__(self):
        return self.n_images

    def compute_point_line_attraction(self, points, lines):
        lines = lines.cuda()
        points = points.cuda()
        dx = lines[:,2:4] - lines[:,:2]
        norm2 = torch.sum(dx**2,dim=-1)
        du = lines[None,:,:2] - points[:,None,:2]
        dv = lines[None,:,2:4] - points[:,None,:2]
        t = - torch.sum(du*dx[None],dim=-1)/norm2[None]
        p_att = lines[None,:,:2] + t.unsqueeze(-1).clamp(min=0,max=1)*dx[None] 
        dis_att = torch.norm(p_att - points[:,None],dim=-1)
        dis_min,idx_min = dis_att.min(dim=1)

        t_range = torch.arange(points.shape[0])

        mask = (dis_min<=self.distance)*(t[t_range,idx_min]<=1)*(t[t_range,idx_min]>=0)
        # mask = (dis_min<=1)*(t[t_range,idx_min]<=1)*(t[t_range,idx_min]>=0)
        
        labels = -torch.ones_like(mask,dtype=torch.long)
        labels[mask] = idx_min[mask]

        return mask.cpu(), labels.cpu()

    def __getitem__(self, idx):       
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        lines = self.wireframes[idx].line_segments(0.05)
        mask, labels = self.compute_point_line_attraction(uv, lines)

        R = self.pose_all[idx][:3,:3]
        # normalization = torch.diag(torch.tensor([1,1,1,1])).float()
        normalization = self.normalization
        # pose = torch.zeros_like(self.pose_all[idx])
        # pose = self.pose_all[idx].clone()
        # pose[:3,:3] = R.t()
        # t = c = -R't t = -Rc
        edges = self.wireframes[idx].edges
        ew = self.wireframes[idx].weights
        sample = {
            "uv": uv,
            "juncs2d": self.wireframes[idx].vertices,
            "edges": edges[ew>0.5],
            "weights": ew[ew>0.5],
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]@normalization,
            "wireframe": self.wireframes[idx],
            "mask": mask,
            "labels": labels,
            "lines": lines[labels],
            "lines_uniq": lines,
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
            # "hat": self.hats[idx][:,:1]
        }

        if self.sampling_idx is not None:
            num_pixels = len(self.sampling_idx)

            # sampling_idx = self.masks[idx].reshape(-1).nonzero().flatten().numpy()
            sampling_idx_pos = mask.nonzero().flatten()
            sampling_idx_pos = np.random.choice(sampling_idx_pos,num_pixels//2)
            sampling_idx_neg = (~mask).nonzero().flatten()
            sampling_idx_neg = np.random.choice(sampling_idx_neg,num_pixels//2)
            sampling_idx = np.concatenate((sampling_idx_pos, sampling_idx_neg))

            ground_truth["rgb"] = self.rgb_images[idx][sampling_idx, :]
            ground_truth['lines2d'] = lines[labels[sampling_idx]]
            ground_truth['labels'] =  (labels[sampling_idx]!=-1)
            # ground_truth["hat"] = self.hats[idx][sampling_idx, :]
            sample["uv"] = uv[sampling_idx, :]
            sample['labels'] = labels[sampling_idx]
            # ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            # sample["uv"] = uv[self.sampling_idx, :]

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
                    if isinstance(entry[0][k], torch.Tensor):
                        ret[k] = torch.stack([obj[k] for obj in entry])
                    else:
                        ret[k] = [obj[k] for obj in entry]
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
        return np.eye(4)
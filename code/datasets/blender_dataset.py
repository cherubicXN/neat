import os
import os.path as osp
import torch
import numpy as np
import cv2

import utils.general as utils
from utils import rend_util

class BlenderDataset(torch.utils.data.Dataset):
    def __init__(self,
        data_dir,
        img_res,
        reverse_coordinate = False
        ):

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
        # self.mask_images = []
        for path in image_paths:
            mask_path = path.replace('.png','-mask.png')
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            mask = cv2.imread(mask_path,0)
            # self.mask_images.append(mask)
            self.rgb_images.append(torch.from_numpy(rgb).float())
        if reverse_coordinate:
            self.normalization = torch.diag(torch.tensor([1,-1,-1,1])).float()
        else:
            self.normalization = torch.diag(torch.tensor([1,1,1,1])).float()

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):       
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        R = self.pose_all[idx][:3,:3]
        # normalization = torch.diag(torch.tensor([1,-1,-1,1])).float()
        # normalization = torch.diag(torch.tensor([1,1,1,1])).float()
        normalization = self.normalization
        # pose = torch.zeros_like(self.pose_all[idx])
        # pose = self.pose_all[idx].clone()
        # pose[:3,:3] = R.t()

        # t = c = -R't t = -Rc
        
        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]@normalization
            # "pose": pose
        }
        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            # sampling_idx = self.mask_images[idx].reshape(-1).nonzero()[0]
            # sampling_idx = np.random.choice(sampling_idx,len(self.sampling_idx))
            # ground_truth["rgb"] = self.rgb_images[idx][sampling_idx, :]
            # sample["uv"] = uv[sampling_idx, :]
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]

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
        return np.eye(4)
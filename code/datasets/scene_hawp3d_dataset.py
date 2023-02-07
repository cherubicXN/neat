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
                 scan_id=0,
                 n_images=-1,
                 ):

        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))

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
        self.masks = []
        self.wireframes = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            hat_path = Path(self.instance_dir)/'hats'/Path(path).with_suffix('.npz').name
            hawp_path = Path(self.instance_dir)/'hawp'/Path(path).with_suffix('.json').name
            hats = np.load(hat_path)
            mask = hats['mask']
            self.masks.append(torch.from_numpy(mask.reshape(-1)))

            wireframe = WireframeGraph.load_json(hawp_path)
            self.wireframes.append(wireframe)

        self.wireframe_file = '{}/wireframes.npz'.format(self.instance_dir)
        wireframes_3d = np.load(self.wireframe_file,allow_pickle=True)

        self.view_pairs = wireframes_3d['view_pairs']
        self.junctions2d = wireframes_3d['junctions_2D']
        self.junctions3d = wireframes_3d['junctions_3D']
        self.edges = wireframes_3d['edges']

        if n_images>0:
            self.n_images = n_images

    def __len__(self):
        # return self.n_images
        return len(self.view_pairs)

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        vi, vj = self.view_pairs[idx]

        intrinsics = torch.stack((self.intrinsics_all[vi],self.intrinsics_all[vj]),dim=0)
        poses = torch.stack((self.pose_all[vi],self.pose_all[vj]),dim=0)

        rgbs = torch.stack((self.rgb_images[vi],self.rgb_images[vj]),dim=0)

        junctions2d_0 = torch.tensor(self.junctions2d[idx][0],dtype=torch.float32)
        junctions2d_1 = torch.tensor(self.junctions2d[idx][1],dtype=torch.float32)
        junctions2d = torch.stack((junctions2d_0,junctions2d_1),dim=0)

        junctions3d = torch.tensor(self.junctions3d[idx],dtype=torch.float32)

        edges = torch.tensor(self.edges[idx],dtype=torch.long)

        sample = {
            "uv": torch.stack((uv,uv),dim=0),
            "intrinsics": intrinsics,
            "pose": poses,
            "views": (vi,vj),
            "juncs2d": junctions2d,
            "juncs3d": junctions3d,
            "edges": edges,
        }


        ground_truth = {
            "rgb": rgbs,
        }

        if self.sampling_idx is not None:
            # sampling_idx0 = self.masks[vi].reshape(-1).nonzero().flatten().numpy()
            # sampling_idx1 = self.masks[vj].reshape(-1).nonzero().flatten().numpy()
            # sampling_idx0 = np.random.choice(sampling_idx0,len(self.sampling_idx))
            # sampling_idx1 = np.random.choice(sampling_idx1,len(self.sampling_idx))
            sampling_idx0 = torch.randperm(self.total_pixels)[:len(self.sampling_idx)]
            sampling_idx1 = torch.randperm(self.total_pixels)[:len(self.sampling_idx)]
            rgb_sampled0 = ground_truth['rgb'][0,sampling_idx0]
            rgb_sampled1 = ground_truth['rgb'][1,sampling_idx1]

            ground_truth['rgb'] = torch.stack((rgb_sampled0,rgb_sampled1))
            sample['uv'] = torch.stack((uv[sampling_idx0],uv[sampling_idx1]),dim=0)

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        assert len(batch_list) == 1

        batch_list = batch_list[0]

        idx = torch.LongTensor([batch_list[0]])
        sample = batch_list[1]
        ground_truth = batch_list[2]
        return idx,sample,ground_truth

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

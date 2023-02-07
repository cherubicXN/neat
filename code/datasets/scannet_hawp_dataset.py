import os
import pandas as pd
import torch
import numpy as np

import utils.general as utils
from utils import rend_util
from pathlib import Path
from sslib import WireframeGraph
from hawp import _C
import matplotlib.pyplot as plt 
def _normalize(inp):
    mag = torch.sqrt(inp[0]*inp[0]+inp[1]*inp[1])
    return inp/(mag+1e-6)

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 n_images=-1,
                 line_detector = 'hawp',
                 distance_threshold = 5.0,
                 ):

        self.instance_dir = os.path.join('../data', data_dir, '{0}'.format(scan_id))
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        image_dir = '{0}/images'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        self.n_images = len(image_paths)
        pose_dir = '{0}/pose'.format(self.instance_dir)
        intrinsic_file = '{0}/intrinsic.txt'.format(self.instance_dir)
        intrinsic = np.loadtxt(intrinsic_file)
        self.intrinsics_all = []
        self.pose_all = []
        self.rgb_images = []
        self.depth_colmap = []
        self.masks = []
        self.wireframes = []
        self.lines = []
        self.labels = []
        self.distance = distance_threshold
        self.score_threshold = 0.05
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

            self.intrinsics_all.append(torch.from_numpy(intrinsic).float())

            img_id = path.split('/')[-1].split('.')[0]
            pose_path = os.path.join(pose_dir, '{0}.txt'.format(img_id))
            pose_c2w = np.loadtxt(pose_path)
            self.pose_all.append(torch.from_numpy(pose_c2w).float())

            depth_path = f'{self.instance_dir}/depth_colmap/{img_id}.npy'

            if os.path.exists(depth_path):
                depth_colmap = np.load(depth_path)
                depth_colmap[depth_colmap > 2.0] = 0
            else:
                depth_colmap = np.zeros(img_res, np.float32)
            depth_colmap = depth_colmap.reshape(-1, 1)
            self.depth_colmap.append(torch.from_numpy(depth_colmap).float())

            hawp_path = Path(self.instance_dir)/line_detector/Path(path).with_suffix('.json').name
            # hats = np.load(hat_path)
            # mask = hats['mask']
            # self.masks.append(torch.from_numpy(mask.reshape(-1)))

            wireframe = WireframeGraph.load_json(hawp_path)
            assert wireframe.frame_height == img_res[0] and wireframe.frame_width ==img_res[1]
            self.wireframes.append(wireframe)
            self.lines.append(wireframe.line_segments(self.score_threshold))
       
        if n_images>0:
            self.n_images = n_images

        from tqdm import tqdm
        print('precomputing the support regions of 2D attraction fields')
        for lines in tqdm(self.lines):
            mask, labels = self.compute_point_line_attraction(lines)
            self.masks.append(mask)
            self.labels.append(labels)
    def __len__(self):
        return self.n_images

    def compute_point_line_attraction(self, lines):
        # lines_ = lines[:512,:-1].cuda()
        lines_ = lines[:,:-1].cuda()
        lmap, labels_onehot, _ = _C.encodels(lines_,self.img_res[0],self.img_res[1],self.img_res[0],self.img_res[1],lines_.shape[0])

        mask, labels = labels_onehot.max(dim=0)

        dismap = torch.sqrt(lmap[0]**2+lmap[1]**2)
        md_map = _normalize(lmap[:2])
        st_map = _normalize(lmap[2:4])
        ed_map = _normalize(lmap[4:])
        st_map = lmap[2:4]
        ed_map = lmap[4:]

        md_ = md_map.reshape(2,-1).t()
        st_ = st_map.reshape(2,-1).t()
        ed_ = ed_map.reshape(2,-1).t()
        Rt = torch.cat(
                (torch.cat((md_[:,None,None,0],md_[:,None,None,1]),dim=2),
                 torch.cat((-md_[:,None,None,1], md_[:,None,None,0]),dim=2)),dim=1)
        R = torch.cat(
                (torch.cat((md_[:,None,None,0], -md_[:,None,None,1]),dim=2),
                 torch.cat((md_[:,None,None,1], md_[:,None,None,0]),dim=2)),dim=1)
        #Rtst_ = torch.matmul(Rt, st_[:,:,None]).squeeze(-1).t()
        #Rted_ = torch.matmul(Rt, ed_[:,:,None]).squeeze(-1).t()
        Rtst_ = torch.bmm(Rt, st_[:,:,None]).squeeze(-1).t()
        Rted_ = torch.bmm(Rt, ed_[:,:,None]).squeeze(-1).t()
        swap_mask = (Rtst_[1]<0)*(Rted_[1]>0)
        pos_ = Rtst_.clone()
        neg_ = Rted_.clone()
        temp = pos_[:,swap_mask]
        pos_[:,swap_mask] = neg_[:,swap_mask]
        neg_[:,swap_mask] = temp

        pos_[0] = pos_[0].clamp(min=1e-9)
        pos_[1] = pos_[1].clamp(min=1e-9)
        neg_[0] = neg_[0].clamp(min=1e-9)
        neg_[1] = neg_[1].clamp(max=-1e-9)
        
        mask = (dismap<=self.distance)*mask

        pos_map = pos_.reshape(-1,self.img_res[0],self.img_res[1])
        neg_map = neg_.reshape(-1,self.img_res[0],self.img_res[1])

        md_angle  = torch.atan2(md_map[1], md_map[0])
        pos_angle = torch.atan2(pos_map[1],pos_map[0])
        neg_angle = torch.atan2(neg_map[1],neg_map[0])

        mask *= (pos_angle>0)
        mask *= (neg_angle<0)
        return mask.cpu().reshape(-1), labels.cpu().reshape(-1)
    
    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        # lines = self.wireframes[idx].line_segments(self.score_threshold)
        lines = self.lines[idx]
        mask = self.masks[idx]
        labels = self.labels[idx]
        # mask, labels = self.compute_point_line_attraction(lines)
        sample = {
            "uv": uv,
            "juncs2d": self.wireframes[idx].vertices,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx],
            'wireframe' : self.wireframes[idx],
            "mask": mask,
            "labels": labels,
            "lines": lines[labels],
            "lines_uniq": lines,
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth_colmap": self.depth_colmap[idx],
        }
        if self.sampling_idx is not None:
            # sampling_idx = self.masks[idx].reshape(-1).nonzero().flatten().numpy()
            # sampling_idx = np.random.choice(sampling_idx,len(self.sampling_idx))
            sampling_idx = mask.nonzero().flatten()
            sampling_idx = np.random.choice(sampling_idx,len(self.sampling_idx))
            ground_truth['rgb'] = self.rgb_images[idx][sampling_idx, :]
            ground_truth['depth_colmap'] = self.depth_colmap[idx][sampling_idx, :]
            ground_truth['lines2d'] = lines[labels[sampling_idx]]
            # ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            # sample["uv"] = uv[self.sampling_idx, :]
            sample['labels'] = labels[sampling_idx]
            sample['uv'] = uv[sampling_idx,:]

            # import pdb; pdb.set_trace()


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
        import pdb; pdb.set_trace()
        return np.load(self.cam_file)['scale_mat_0']

import os
import os.path as osp
import torch
import numpy as np
import cv2

import utils.general as utils
from utils import rend_util
# from utils import hawp_util
from .utils.wireframe import WireframeGraph
from hawp.base import _C
from pathlib import Path

def _normalize(inp):
    mag = torch.sqrt(inp[0]*inp[0]+inp[1]*inp[1])
    return inp/(mag+1e-6)
class BlenderDataset(torch.utils.data.Dataset):
    def __init__(self,
        data_dir,
        img_res,
        reverse_coordinate = False,
        line_detector = 'hawp',
        distance_threshold = 10.0,
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
        self.masks = []
        self.wireframes = []
        self.lines = []
        self.labels = []
        self.att_points = []
        self.distance = distance_threshold
        self.score_threshold = 0.05

        valid_ids = []
        for i, path in enumerate(image_paths):
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            hawp_path = Path(self.instance_dir)/line_detector/Path(path).with_suffix('.json').name

            wireframe = WireframeGraph.load_json(hawp_path)
            if wireframe.vertices.shape[0] == 0 or wireframe.edges.shape[0] == 0:
                continue
            if wireframe.line_segments(self.score_threshold).shape[0] == 0:
                continue
            valid_ids.append(i)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            self.wireframes.append(wireframe)
            assert wireframe.frame_height == img_res[0] and wireframe.frame_width ==img_res[1]
            self.lines.append(wireframe.line_segments(self.score_threshold))
        self.intrinsics_all = self.intrinsics_all[valid_ids]
        self.pose_all = self.pose_all[valid_ids]
        self.n_images = len(valid_ids)
        
        if reverse_coordinate:
            self.normalization = torch.diag(torch.tensor([1,-1,-1,1])).float()
        else:
            self.normalization = torch.diag(torch.tensor([1,1,1,1])).float()
        
        from tqdm import tqdm
        print('precomputing the support regions of 2D attraction fields')
        for lines in tqdm(self.lines):
            mask, labels, att_points = self.compute_point_line_attraction(lines)
            self.masks.append(mask)
            self.labels.append(labels)
            self.att_points.append(att_points)
        
    def __len__(self):
        return self.n_images

    def compute_point_line_attraction(self, lines):
        # lines_ = lines[:512,:-1].cuda()
        lines_ = lines[:,:-1].cuda()
        lmap, labels_onehot, _ = _C.encodels(lines_,self.img_res[0],self.img_res[1],self.img_res[0],self.img_res[1],lines_.shape[0])
        mask, labels = labels_onehot.max(dim=0)
        assert labels.max()<lines.shape[0]
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

        offsets = lmap[:2].permute(1,2,0)
        proj_points = torch.zeros((*mask.shape,2),device=mask.device,dtype=torch.float32)
        proj_points[mask,:] = offsets[mask] + mask.nonzero()[:,[1,0]].float()
        return mask.cpu().reshape(-1), labels.cpu().reshape(-1), proj_points.reshape(-1,2)

    def __getitem__(self, idx):       
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        lines = self.lines[idx]
        mask = self.masks[idx]
        labels = self.labels[idx]
        # mask, labels = self.compute_point_line_attraction(uv, lines)

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
            "uv_proj": self.att_points[idx],
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
            "rgb": self.rgb_images[idx]
            # "hat": self.hats[idx][:,:1]
        }

        if self.sampling_idx is not None:
            # sampling_idx = self.masks[idx].reshape(-1).nonzero().flatten().numpy()
            # sampling_idx = np.random.choice(sampling_idx,len(self.sampling_idx))
            sampling_idx = mask.nonzero().flatten()
            sampling_idx = np.random.choice(sampling_idx,len(self.sampling_idx))
            ground_truth['rgb'] = self.rgb_images[idx][sampling_idx, :]
            ground_truth['lines2d'] = lines[labels[sampling_idx]]
            sample['lines'] = lines[labels[sampling_idx]]
            # ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            # sample["uv"] = uv[self.sampling_idx, :]
            sample['labels'] = labels[sampling_idx]
            sample['uv'] = uv[sampling_idx,:]
            sample['uv_proj'] = self.att_points[idx][sampling_idx]

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
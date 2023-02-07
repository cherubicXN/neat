import os
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
        self.lines = []
        self.labels = []
        self.att_points = []
        self.distance = distance_threshold
        if self.distance <0:
            self.distance = 1e10
        self.score_threshold = 0.05

        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            hawp_path = Path(self.instance_dir)/line_detector/Path(path).with_suffix('.json').name
            # hats = np.load(hat_path)
            # mask = hats['mask']
            # self.masks.append(torch.from_numpy(mask.reshape(-1)))

            wireframe = WireframeGraph.load_json(hawp_path)
            assert wireframe.frame_height == img_res[0] and wireframe.frame_width ==img_res[1]
            self.wireframes.append(wireframe)
            self.lines.append(wireframe.line_segments(self.score_threshold))
        cam_loc = torch.stack([x[:3,3] for x in self.pose_all])
        principle_axis = torch.stack([x[:3,2] for x in self.pose_all])
        view_affinity = torch.ones(self.n_images, self.n_images)*1e10
        for i in range(self.n_images):
            for j in range(i+1, self.n_images):
                acos = torch.acos(torch.sum(principle_axis[i]*principle_axis[j]))
                if acos>np.pi/3:
                    continue
                view_affinity[i,j] = torch.norm(cam_loc[i]-cam_loc[j])
                view_affinity[j,i] = torch.norm(cam_loc[i]-cam_loc[j])
        self.view_adj = view_affinity.min(dim=-1)[1]
        # for i in range(self.n_images):
        #     plt.subplot(1,2,1)
        #     plt.imshow(self.rgb_images[i].reshape(*self.img_res,-1))
        #     plt.subplot(1,2,2)
        #     plt.imshow(self.rgb_images[view_adj[i]].reshape(*self.img_res,-1))
        #     plt.show()
        # import pdb; pdb.set_trace()
        if n_images>0:
            self.n_images = n_images



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

        import pdb; pdb.set_trace()
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

        # lines = self.wireframes[idx].line_segments(self.score_threshold)
        lines = self.lines[idx]
        mask = self.masks[idx]
        labels = self.labels[idx]
        # mask, labels = self.compute_point_line_attraction(lines)
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
        # while True:
            # idx_j = np.random.choice(len(self.pose_all))
            # if idx_j != idx:
                # break
        # idx_j = (idx+1) % len(self.pose_all)
        idx_j = self.view_adj[idx]

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "pose_j": self.pose_all[idx_j],
            "intrinsics": self.intrinsics_all[idx_j],
            "wireframe": self.wireframes[idx_j],
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
        return np.load(self.cam_file)['scale_mat_0']

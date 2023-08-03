import torch
from torch import nn
import utils.general as utils
import torch.nn.functional as F 

class VolSDFLoss(nn.Module):
    def __init__(self, rgb_loss, eikonal_weight, point_sdf_weight, point_eikonal_weight):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')
        self.point_sdf_weight = point_sdf_weight
        self.point_eikonal_weight = point_eikonal_weight

    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_depth_loss(self, depth, depth_colmap, mask=None):
        depth_colmap = depth_colmap.reshape(-1)
        depth_colmap_mask = depth_colmap > 0
        if mask is not None:
            depth_colmap_mask *= mask
        # import pdb; pdb.set_trace()
        if depth_colmap_mask.sum() > 0:
            depth_loss = F.l1_loss(depth[depth_colmap_mask], depth_colmap[depth_colmap_mask], reduction='none')
            depth_loss = depth_loss.mean()
            return depth_loss
        else:
            return torch.tensor([0.]).cuda()

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb'].cuda()

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        
        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss
                # + \
                # self.point_sdf_weight * sdf_loss + \
                # self.point_eikonal_weight * pts_eikonal
        
        depth_pred = model_outputs['depth-pts']
        depth_gt = ground_truth['depth'].flatten().cuda()
        depth_weight = ground_truth['depth.w'].flatten().cuda()
        depth_loss = F.l1_loss(depth_pred,depth_gt,reduction='none')
        depth_loss = torch.sum(depth_loss*depth_weight)/torch.sum(depth_weight).clamp_min(1e-10)

        loss += depth_loss*0.1
        
        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss, 
            'depth_loss': depth_loss,
        }

        # if self.point_sdf_weight > 0:
        #     output['sdf_loss'] = sdf_loss
        # if self.point_eikonal_weight > 0:
        #     output['pts_eloss']: pts_eikonal

        return output

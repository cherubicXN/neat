import torch
from torch import nn
import utils.general as utils
import torch.nn.functional as F

import matplotlib.pyplot as plt
@torch.no_grad()
def plt_lines(lines,*args, **kwargs):
    if isinstance(lines, torch.Tensor):
        lines = lines.cpu().numpy()
    return plt.plot([lines[:,0],lines[:,2]],[lines[:,1],lines[:,3]],*args, **kwargs)

class VolSDFLoss(nn.Module):
    def __init__(self, rgb_loss, eikonal_weight, line_weight):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.line_weight = line_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')
        self.steps = 0

    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_line_loss(self, lines2d, lines2d_gt, lines_weight, threshold = 100):
        dist1 = torch.sum((lines2d-lines2d_gt)**2,dim=-1,keepdim=True).detach()
        dist2 = torch.sum((lines2d-lines2d_gt[:,[2,3,0,1]])**2,dim=-1,keepdim=True).detach()

        lines_tgt = torch.where(dist1<dist2,lines2d_gt,lines2d_gt[:,[2,3,0,1]])

        lines2d_loss = torch.abs(lines2d-lines_tgt).mean(dim=-1)

        labels = (lines2d_loss.detach()<threshold).long()

        total_loss = torch.sum(lines2d_loss*lines_weight.flatten()*labels)/labels.sum().clamp_min(1)
        return total_loss, lines2d_loss.detach()
    
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
        self.steps +=1
        rgb_gt = ground_truth['rgb'].cuda()

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()
        
        import pdb; pdb.set_trace()
        depth_colmap = ground_truth['depth-pts'].cuda()
        depth_loss = self.get_depth_loss(model_outputs['depth'], depth_colmap)

        loss = rgb_loss + \
               depth_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.line_weight*lines2d_loss #+ \
            #    loss_cls*0.0

        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'depth_loss': depth_loss
        }

        # if self.steps>500:
        if 'lines2d-aux' in model_outputs:
            for it, aux in enumerate(lines2d_loss_aux):
                output['aux-{}-loss'.format(it)] = aux
                loss += aux*self.line_weight*0.1
        return output

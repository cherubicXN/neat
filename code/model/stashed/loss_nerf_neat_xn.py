import torch
from torch import nn
import utils.general as utils

import matplotlib.pyplot as plt
@torch.no_grad()
def plt_lines(lines,*args, **kwargs):
    if isinstance(lines, torch.Tensor):
        lines = lines.cpu().numpy()
    return plt.plot([lines[:,0],lines[:,2]],[lines[:,1],lines[:,3]],*args, **kwargs)


class VolSDFLoss(nn.Module):
    def __init__(self, rgb_loss, eikonal_weight, line_weight):
        super().__init__()
        self.line_weight = line_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')
        self.steps = 0

    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_line_loss(self, lines2d, lines2d_gt, lines_weight, threshold = 100):
        dist1 = torch.sum((lines2d-lines2d_gt)**2,dim=-1,keepdim=True).detach()
        dist2 = torch.sum((lines2d-lines2d_gt[:,[2,3,0,1]])**2,dim=-1,keepdim=True).detach()

        lines_tgt = torch.where(dist1<dist2,lines2d_gt,lines2d_gt[:,[2,3,0,1]])

        lines2d_loss = torch.abs(lines2d-lines_tgt).mean(dim=-1)

        labels = (lines2d_loss.detach()<threshold).long()

        total_loss = torch.sum(lines2d_loss*lines_weight.flatten()*labels)/labels.sum().clamp_min(1)
        return total_loss, lines2d_loss.detach()

    def forward(self, model_outputs, ground_truth):
        self.steps +=1
        lines2d_gt, lines_weight =ground_truth['lines2d'][0].cuda().split(4,dim=-1)
        if 'labels' in ground_truth:
            lines_weight = lines_weight*ground_truth['labels'][0,:,None].cuda()

        import pdb; pdb.set_trace()
        lines2d = model_outputs['lines2d']
        lines2d_gt_rep = lines2d_gt[:,None].repeat(1,lines2d.shape[1],1)
        lines_weight_rep = lines_weight[:,None].repeat(1,lines2d.shape[1],1)
        lines2d_loss, stat = self.get_line_loss(lines2d.reshape(-1,4), lines2d_gt_rep.reshape(-1,4), lines_weight_rep.reshape(-1))#,threshold=10)

        density = stat.reshape(lines2d.shape[:-1])
        
        stat = (stat<100).sum()/stat.numel()


        lines2d_aux = model_outputs['lines2d-aux']
        aux_loss, _ = self.get_line_loss(lines2d_aux, lines2d_gt, lines_weight)
        
        
        rgb_gt = ground_truth['rgb'].cuda()

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        if 'rgb_values_coarse' in model_outputs:
            rgb_loss_coarse = self.get_rgb_loss(model_outputs['rgb_values_coarse'], rgb_gt)
        else:
            rgb_loss_coarse = 0.
            
        loss = rgb_loss + \
               rgb_loss_coarse + \
               self.line_weight*lines2d_loss + \
               self.line_weight*aux_loss
            #    loss_cls*0.0

        output = {
            'loss': loss,
            # 'cls_loss': loss_cls,
            'rgb_loss': rgb_loss,
            'line_loss': lines2d_loss,
            'aux_loss': aux_loss,
            'count': stat
        }


        return output

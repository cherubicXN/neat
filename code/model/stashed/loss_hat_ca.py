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
        self.eikonal_weight = eikonal_weight
        self.line_weight = line_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')

    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def forward(self, model_outputs, ground_truth):
        lines2d_gt, lines_weight =ground_truth['lines2d'][0].cuda().split(4,dim=-1)
        lines2d = model_outputs['lines2d'].reshape(-1,4)


        dist1 = torch.sum((lines2d-lines2d_gt)**2,dim=-1,keepdim=True).detach()
        dist2 = torch.sum((lines2d-lines2d_gt[:,[2,3,0,1]])**2,dim=-1,keepdim=True).detach()

        lines_tgt = torch.where(dist1<dist2,lines2d_gt,lines2d_gt[:,[2,3,0,1]])

        lines2d_loss = torch.abs(lines2d-lines_tgt).mean(dim=-1)

        labels = (lines2d_loss.detach()<100).long()
        # labels = torch.ones_like(labels)

        # loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(model_outputs['logits'].flatten(), labels.float())
        lines2d_loss = torch.sum(lines2d_loss*lines_weight.flatten()*labels)/labels.sum().clamp_min(1)

        lines2d_qpd = model_outputs['lines2d-query'].reshape(-1,4)
        lines2d_qgt = model_outputs['wireframe-gt'][0].line_segments(0.05).cuda()[:,:-1]

        d1 = torch.sum((lines2d_qpd[:,None]-lines2d_qgt[None])**2,dim=-1).detach()
        d2 = torch.sum((lines2d_qpd[:,None]-lines2d_qgt[None,:,[2,3,0,1]])**2,dim=-1).detach()
        dmap = torch.min(d1,d2)
        assignment = dmap.min(dim=0)[1]

        lines2d_qpd_sel = lines2d_qpd[assignment]
        lines2d_qgt_sel = torch.where(d1[assignment].diag()[:,None]<d2[assignment].diag()[:,None], lines2d_qgt,lines2d_qgt[:,[2,3,0,1]])

        query_loss = torch.nn.functional.l1_loss(lines2d_qpd_sel, lines2d_qgt_sel)


        rgb_gt = ground_truth['rgb'].cuda()

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.line_weight*lines2d_loss + \
                query_loss*0.01
            #    loss_cls*0.0

        output = {
            'loss': loss,
            # 'cls_loss': loss_cls,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'line_loss': lines2d_loss,
            'query_loss': query_loss
        }

        return output

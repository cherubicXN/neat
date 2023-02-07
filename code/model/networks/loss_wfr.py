import torch
from torch import nn
import utils.general as utils

import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import trimesh

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

    def forward(self, model_outputs, ground_truth):
        self.steps +=1
        lines2d_gt, lines_weight =ground_truth['lines2d'][0].cuda().split(4,dim=-1)
        if 'labels' in ground_truth:
            lines_weight = lines_weight*ground_truth['labels'][0,:,None].cuda()
        lines2d = model_outputs['lines2d'].reshape(-1,4)

        l2d_loss_uncalib, threshold = self.get_line_loss(lines2d, lines2d_gt, lines_weight)
        # import pdb; pdb.set_trace()
        count = (threshold<100).sum()

        lines2d_gt_calib = lines2d_gt.reshape(-1,2)
        lines2d_gt_calib_h = torch.cat([lines2d_gt_calib, torch.ones_like(lines2d_gt_calib[:,:1])],dim=-1)
        lines2d_gt_calib_h =  (model_outputs['K'].inverse()@lines2d_gt_calib_h.t()).t()
        lines2d_gt_calib = lines2d_gt_calib_h[:,:2]/lines2d_gt_calib_h[:,2,None]
        lines2d_gt_calib = lines2d_gt_calib.reshape(-1,4)
        lines2d_loss, _ = self.get_line_loss(model_outputs['lines2d_calib'].reshape(-1,4), lines2d_gt_calib, lines_weight)
        

        rgb_gt = ground_truth['rgb'].cuda()

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        j3d_local = model_outputs['j3d_local']
        j3d_global = model_outputs['j3d_global']
        j2d_local = model_outputs['j2d_local'].detach()
        j2d_global = model_outputs['j2d_global'].detach()
        j2d_local_calib = model_outputs['j2d_local_calib']
        j2d_global_calib = model_outputs['j2d_global_calib']

        with torch.no_grad():
            j3d_cost = torch.cdist(j3d_local,j3d_global, p=1)
            j2d_cost = torch.cdist(j2d_local_calib,j2d_global_calib, p=1)
            jcost_all = j3d_cost + j2d_cost*0.1
        # assign = linear_sum_assignment(j3d_cost.detach().cpu().numpy())
            jcost_all[torch.isnan(jcost_all)] = 100000
        
        assign = linear_sum_assignment(jcost_all.detach().cpu().numpy())
        assign_cost = jcost_all[assign[0],assign[1]]
        
        loss_j3d = torch.sum((j3d_local[assign[0]]-j3d_global[assign[1]]).abs(),dim=-1)
        loss_j3d = torch.mean(loss_j3d*(assign_cost<100000))
        loss_j2d = torch.sum((j2d_local_calib[assign[0]]-j2d_global_calib[assign[1]]).abs(),dim=-1)
        loss_j2d = torch.mean(loss_j2d*(assign_cost<100000))
        # loss_j2d = torch.sum((j2d_local_calib[assign[0]]-j2d_global_calib[assign[1]])**2,dim=-1).mean()
        
        with torch.no_grad():
            # loss_j2d_u = torch.sum((j2d_local[assign[0]]-j2d_global[assign[1]])**2,dim=-1).mean()
            loss_j2d_u = torch.sum((j2d_local[assign[0]]-j2d_global[assign[1]]).abs(),dim=-1).mean()

        # if self.steps%(49*20)==0:
            # print('loss',rgb_loss.item(),eikonal_loss.item(),lines2d_loss.item(),loss_j3d.item())
            # import pdb; pdb.set_trace()
        
        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.line_weight*lines2d_loss + \
                loss_j3d*0.1 + \
                loss_j2d*0.01
                # loss_j2d*0.0001
                # loss_j2d*0.01
            #    loss_cls*0.0

        
        output = {
            'loss': loss,
            # 'cls_loss': loss_cls,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'line_loss': lines2d_loss,
            'l2d_loss': l2d_loss_uncalib,
            'j3d_loss': loss_j3d,
            'j2d_loss': loss_j2d,
            'j2d_stat': loss_j2d_u,
            'count': count
        }
        if 'median' in model_outputs:
            output['median'] = model_outputs['median']
        # if self.steps>500:
      
        return output

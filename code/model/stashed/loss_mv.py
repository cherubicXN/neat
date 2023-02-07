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

    def project2D(self, K,R,T, points3d):
        shape = points3d.shape 
        assert shape[-1] == 3
        X = points3d.reshape(-1,3)
        
        x = K@(R@X.t()+T)
        x = x.t()
        x = x/x[:,-1:]
        x = x.reshape(*shape)[...,:2]
        return x
    def forward(self, model_outputs, ground_truth):
        self.steps +=1
        lines2d_gt, lines_weight =ground_truth['lines2d'][0].cuda().split(4,dim=-1)
        if 'labels' in ground_truth:
            lines_weight = lines_weight*ground_truth['labels'][0,:,None].cuda()
        lines2d = model_outputs['lines2d'].reshape(-1,4)
        lines2d_loss, threshold = self.get_line_loss(lines2d, lines2d_gt, lines_weight)
        # import pdb; pdb.set_trace()
        count = (threshold<100).sum()
        
        if 'lines2d-aux' in model_outputs:
            lines2d_loss_aux = []
            for l2d in model_outputs['lines2d-aux']:
                loss_it, threshold = self.get_line_loss(l2d.reshape(-1,4), lines2d_gt, lines_weight, threshold.clamp_max(100))
                lines2d_loss_aux.append(loss_it)

        rgb_gt = ground_truth['rgb'].cuda()

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        proj_mat = ground_truth['pose_j'][0].cuda().inverse()[:3]
        K = ground_truth['intrinsics'][0].cuda()[:3,:3]
        R = proj_mat[:,:3]
        T = proj_mat[:,3:]

        lines2d_gtj = ground_truth['wireframe'][0].line_segments().cuda()[:,:-1]
        lines2d_gtj = torch.cat((lines2d_gtj,lines2d_gtj[:,[2,3,0,1]]))
        lines2d_prj = self.project2D(K, R, T, model_outputs['lines3d']).reshape(-1,4)

        matching_cost = torch.sum((lines2d_prj[None]-lines2d_gtj[:,None])**2,dim=-1).detach()
        mincost, minidx = matching_cost.min(dim=1)

        is_valid = (mincost<100).float()
        n_valid = (mincost<100).sum()
        mv_loss = torch.abs(lines2d_prj[minidx] - lines2d_gtj).mean(dim=-1)

        mv_loss = torch.sum(mv_loss*is_valid)/n_valid.clamp_min(1)
        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.line_weight*lines2d_loss + \
               self.line_weight*mv_loss
            #    loss_cls*0.0

        
        output = {
            'loss': loss,
            # 'cls_loss': loss_cls,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'line_loss': lines2d_loss,
            'mv_loss': mv_loss,
            'count2': n_valid,
            'count': count
        }
        # if self.steps>500:
        if 'lines2d-aux' in model_outputs:
            for it, aux in enumerate(lines2d_loss_aux):
                output['aux-{}-loss'.format(it)] = aux
                loss += aux*self.line_weight*0.1
        return output

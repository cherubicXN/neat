import torch
from torch import nn
import utils.general as utils

import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

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

    @torch.no_grad()
    def group(self, lines3d_pred, lines2d_pred, wireframe):
        lines2d_gt = wireframe.line_segments(0.05).cuda()
        
        lines2d_pred = lines2d_pred.reshape(-1,4)
        dis1 = torch.sum((lines2d_pred[:,None] - lines2d_gt[None,:,:-1])**2,dim=-1)
        dis2 = torch.sum((lines2d_pred[:,None] - lines2d_gt[None,:,[2,3,0,1]])**2,dim=-1)
        dis = torch.min(dis1,dis2)

        mindis, minidx = dis.min(dim=1)

        i_range = torch.arange(len(minidx),device='cuda')
        mindis1 = dis1[i_range,minidx]
        mindis2 = dis2[i_range,minidx]
        N = sum(mindis<10)

        if N == 0:
            return None

        is_swap = mindis1>mindis2
        lines2d_pred_swap = lines2d_pred.clone()
        lines3d_pred_swap = lines3d_pred.clone()
        lines2d_pred_swap[is_swap] = lines2d_pred[is_swap][:,[2,3,0,1]]
        lines3d_pred_swap[is_swap] = lines3d_pred[is_swap][:,[1,0]]
        
        labels = minidx[mindis<10].unique()
        pts3d_grouped = []
        for label in labels:
            cur_idx = torch.nonzero((minidx == label)*(mindis<10)).flatten()
            if cur_idx.shape[0] == 0:
                continue
            # import pdb; pdb.set_trace()
            pts3d_grouped.append(lines3d_pred_swap[cur_idx].reshape(-1,3).mean(dim=0))
            # lines3d_cur = lines3d_pred_swap[cur_idx].mean(dim=0)
            # lines3d_grouped.append(lines3d_cur)
            # lines2d_grouped.append(lines2d_gt[label])

        # lines3d_grouped = torch.stack(lines3d_grouped)
        # lines2d_grouped = torch.stack(lines2d_grouped)
        # import pdb; pdb.set_trace()
        
        # return lines3d_grouped, lines2d_grouped
        pts3d_grouped = torch.stack(pts3d_grouped)
        return pts3d_grouped
    def forward(self, model_outputs, ground_truth):
        self.steps +=1
        lines2d_gt, lines_weight =ground_truth['lines2d'][0].cuda().split(4,dim=-1)
        if 'labels' in ground_truth:
            lines_weight = lines_weight*ground_truth['labels'][0,:,None].cuda()
        lines2d = model_outputs['lines2d'].reshape(-1,4)
        lines2d_loss, threshold = self.get_line_loss(lines2d, lines2d_gt, lines_weight)


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

        pts3d_grouped = self.group(model_outputs['lines3d'],model_outputs['lines2d'],model_outputs['wireframe-gt'][0])
        

        if pts3d_grouped is not None:
            lines3d_query = model_outputs['lines3d-query']
            # lines2d_query = model_outputs['lines2d-query'].reshape(-1,4)
            # import pdb; pdb.set_trace()
            # cost = torch.sum((lines3d_query[:,None].detach()-lines3d_grouped[None])**2,dim=-1).sum(dim=-1)
            cost = torch.sum((lines3d_query[:,None].detach()-pts3d_grouped[None]).abs(),dim=-1)
            idx_query, idx_tgt = linear_sum_assignment(cost.cpu())
            # pts2d_grouped_tgt = lines2d_grouped[:,:-1].clone()
            # is_swap = cost1[idx_query,idx_tgt]>cost2[idx_query,idx_tgt]
            # lines2d_grouped_tgt[is_swap] = lines2d_grouped_tgt[is_swap][:,[2,3,0,1]]
            

            loss_query = torch.abs(lines3d_query[idx_query]-pts3d_grouped[idx_tgt]).sum(dim=-1).mean()#.mean(dim=-1).mean(dim=-1).mean()
            # import pdb; pdb.set_trace()
        else:
            loss_query = torch.zeros([],device='cuda')

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.line_weight*lines2d_loss + \
                loss_query*0.1
            #    self.line_weight*loss_query
            #    loss_cls*0.0

        
        output = {
            'loss': loss,
            # 'cls_loss': loss_cls,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'line_loss': lines2d_loss,
            'query_loss': loss_query,
            'count': count
        }
        # if self.steps>500:
        if 'lines2d-aux' in model_outputs:
            for it, aux in enumerate(lines2d_loss_aux):
                output['aux-{}-loss'.format(it)] = aux
                loss += aux*self.line_weight*0.1
        return output

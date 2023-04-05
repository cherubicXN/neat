import torch
from torch import nn
import utils.general as utils

import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import trimesh
import torch.nn.functional as F
@torch.no_grad()
def plt_lines(lines,*args, **kwargs):
    if isinstance(lines, torch.Tensor):
        lines = lines.cpu().numpy()
    return plt.plot([lines[:,0],lines[:,2]],[lines[:,1],lines[:,3]],*args, **kwargs)

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)
def compute_scale_and_shift(prediction, target, mask = None):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    if mask == None:
        mask = torch.ones_like(target)
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1
def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)

def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)
class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)
class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total

class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy

class VolSDFLoss(nn.Module):
    def __init__(self, rgb_loss, eikonal_weight, line_weight, junction_3d_weight = 0.1, junction_2d_weight = 0.01):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.line_weight = line_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')
        self.steps = 0
        self.junction_3d_weight = junction_3d_weight
        self.junction_2d_weight = junction_2d_weight
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)


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
    
    def get_depth_loss(self, depth, depth_colmap):
        depth_colmap = depth_colmap.reshape(-1)
        depth_colmap_mask = depth_colmap > 0
        # import pdb; pdb.set_trace()
        if depth_colmap_mask.sum() > 0:
            depth_loss = F.l1_loss(depth[depth_colmap_mask], depth_colmap[depth_colmap_mask], reduction='none')
            depth_loss = depth_loss.mean()
            return depth_loss
        else:
            return torch.tensor([0.]).cuda()

    def forward(self, model_outputs, ground_truth):
        self.steps +=1
        # if self.steps >= 7047:
        #     import pdb; pdb.set_trace()
        lines2d_gt, lines_weight =ground_truth['lines2d'][0].cuda().split(4,dim=-1)
        if 'labels' in ground_truth:
            lines_weight = lines_weight*ground_truth['labels'][0,:,None].cuda()
        # lines_weight = torch.ones_like(lines_weight).cuda()
        lines2d = model_outputs['lines2d'].reshape(-1,4)

        l2d_loss_uncalib, threshold = self.get_line_loss(lines2d, lines2d_gt, lines_weight) #TODO: check if the lines_weight is necessary
        count = (threshold<100).sum()
        lines2d_gt_calib = lines2d_gt.reshape(-1,2)
        lines2d_gt_calib_h = torch.cat([lines2d_gt_calib, torch.ones_like(lines2d_gt_calib[:,:1])],dim=-1)
        lines2d_gt_calib_h =  (model_outputs['K'].inverse()@lines2d_gt_calib_h.t()).t()
        lines2d_gt_calib = lines2d_gt_calib_h[:,:2]/lines2d_gt_calib_h[:,2,None]
        lines2d_gt_calib = lines2d_gt_calib.reshape(-1,4)

        lines2d_loss, _ = self.get_line_loss(model_outputs['lines2d_calib'].reshape(-1,4), lines2d_gt_calib, lines_weight*(threshold<100).reshape(-1,1))
        if torch.isnan(lines2d_loss):
            import pdb; pdb.set_trace()        

        rgb_gt = ground_truth['rgb'].cuda()

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        with torch.no_grad():
            dscale, dshift = compute_scale_and_shift(model_outputs['depth'], ground_truth['depth_colmap'].cuda())

        pred = model_outputs['depth'].reshape(1,32,32)
        target = ground_truth['depth_colmap'].cuda().reshape(1,32,32)
        depth_loss = self.depth_loss(pred, target, torch.ones_like(target))
        # regularizer = GradientLoss(scales=1,reduction='batch-based')
        # depth_loss = self.get_depth_loss(model_outputs['depth'], ground_truth['depth_colmap'].cuda())

        loss = rgb_loss + \
            depth_loss*0.1 + \
            self.eikonal_weight * eikonal_loss + \
            self.line_weight*lines2d_loss 
            # self.junction_3d_weight*loss_j3d + \
            # self.junction_2d_weight*loss_j2d
        output = {
            'loss': loss,
            # 'cls_loss': loss_cls,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'line_loss': lines2d_loss,
            'l2d_loss': l2d_loss_uncalib,
            'depth_loss': depth_loss,
            'count': count,
            'j3d_loss': torch.tensor(0.0).cuda().float(),
            'j2d_loss': torch.tensor(0.0).cuda().float(),
            'j2d_stat': torch.tensor(0.0).cuda().float(),
            'jcount': torch.tensor(0.0).cuda().float(),
        }
        if model_outputs['j3d_local'].shape[0] > 0:
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
            
            assign = linear_sum_assignment(jcost_all.detach().cpu().numpy())
            assign_cost = jcost_all[assign[0],assign[1]]

            loss_j3d = torch.sum((j3d_local[assign[0]]-j3d_global[assign[1]]).abs(),dim=-1)
            loss_j3d = torch.mean(loss_j3d)
            loss_j2d = torch.sum((j2d_local_calib[assign[0]]-j2d_global_calib[assign[1]]).abs(),dim=-1)
            loss_j2d = torch.mean(loss_j2d)

            with torch.no_grad():
                loss_j2d_u = torch.sum((j2d_local[assign[0]]-j2d_global[assign[1]]).abs(),dim=-1).mean()


            loss += self.junction_3d_weight*loss_j3d + \
                self.junction_2d_weight*loss_j2d 
            jcount = (assign_cost<10).sum()

            output['j3d_loss'] = loss_j3d
            output['j2d_loss'] = loss_j2d
            output['j2d_stat'] = loss_j2d_u
            output['jcount'] = jcount
   
        
        
        if 'median' in model_outputs:
            output['median'] = model_outputs['median']
        # if self.steps>500:
      
        return output

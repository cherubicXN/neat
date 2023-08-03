import pdb
from turtle import pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from traitlets import default

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import UniformSampler

# class AttractionFieldNetwork(nn.Module):
class AttractionFieldNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        # x = self.sigmoid(x)
        y = points[:,None] + x.reshape(-1,2,3)

        return y

class CenterNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, feature_vectors):
        rendering_input = torch.cat([points, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        # x = self.sigmoid(x)
        y = points[:,None] + x.reshape(-1,2,3)

        return y

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.sigmoid(x)
        return x

class NeRF(nn.Module):
    def __init__(
            self, 
            D=8, 
            W=256, 
            output_ch=4, 
            skips=[4], 
            use_viewdirs=True,
            multires=0,
            multires_view=0
    ):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.embed_fn = None
        self.input_ch = 3
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            self.input_ch = input_ch
        
        self.embedview_fn = None
        self.input_ch_views = 3
        if multires_view > 0 and self.use_viewdirs:
            embedview_fn, input_ch_view = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            self.input_ch_views = input_ch_view
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        # input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)

        if self.embedview_fn is not None:
            input_views = self.embedview_fn(input_views)
        
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            feature = h
            outputs = self.output_linear(h)

        return outputs, feature 

class NeRFNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()
        self.ray_sampler = UniformSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        N_important = self.ray_sampler.N_important
        self.use_fine_sample = N_important > 0
        # import pdb; pdb.set_trace()

        self.attraction_network = AttractionFieldNetwork(self.feature_vector_size, **conf.get_config('attraction_network'))
        self.rendering_network = NeRF(**conf.get_config('rendering_network'))
        if self.use_fine_sample:
            self.rendering_network_fine = NeRF(**conf.get_config('rendering_network_fine'))
        self.use_center_network = conf.get_bool('use_center_network')
        if self.use_center_network:
            self.center_network = CenterNetwork(self.feature_vector_size, **conf.get_config('center_network'))
        self.raw_noise_std = conf.get_float('raw_noise_std')
        # import pdb; pdb.set_trace()

    def project2D(self, K,R,T, points3d):
        shape = points3d.shape 
        assert shape[-1] == 3
        X = points3d.reshape(-1,3)
        
        x = K@(R@X.t()+T)
        x = x.t()
        x = x/x[:,-1:]
        x = x.reshape(*shape)[...,:2]
        return x
    
    def forward(self, input):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        rays_d = z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        depth_ratio = rays_d.norm(dim=-1)
        points = cam_loc.unsqueeze(1) + rays_d
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)
        redner_outputs, feature_vectors = self.rendering_network(points_flat, dirs_flat)
        rgb_flat = torch.sigmoid(redner_outputs[...,:3])
        rgb = rgb_flat.reshape(-1, N_samples, 3)  # [N_rays, N_samples, 3]
        sigma = redner_outputs[...,3].reshape(-1, N_samples)

        weights = self.volume_rendering(z_vals, sigma, depth_ratio)  # [N_rays, N_samples]
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        if self.use_fine_sample:
            rgb_values_coarse = rgb_values
            z_vals_coarse = z_vals
            with torch.no_grad():
                z_vals = self.ray_sampler.get_z_vals_fine(z_vals, weights, self).detach()
            N_samples = z_vals.shape[1]
            rays_d = z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
            depth_ratio = rays_d.norm(dim=-1)
            points = cam_loc.unsqueeze(1) + rays_d
            points_flat = points.reshape(-1, 3)

            dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
            dirs_flat = dirs.reshape(-1, 3)
            redner_outputs, feature_vectors = self.rendering_network_fine(points_flat, dirs_flat)
            rgb_flat = torch.sigmoid(redner_outputs[...,:3])
            rgb = rgb_flat.reshape(-1, N_samples, 3)  # [N_rays, N_samples, 3]
            sigma = redner_outputs[...,3].reshape(-1, N_samples)

            weights = self.volume_rendering(z_vals, sigma, depth_ratio)  # [N_rays, N_samples]
            rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

            # import pdb; pdb.set_trace()

        lines3d = self.attraction_network(points_flat, dirs_flat, feature_vectors)
        lines3d = lines3d.reshape(-1, N_samples, 2,3)
        lines3d = torch.sum(weights[:,:,None,None]*lines3d,dim=1)
        # import pdb; pdb.set_trace()
        proj_mat = pose[0].inverse()[:3]
        R = proj_mat[:,:3]
        T = proj_mat[:,3:]

        lines2d = self.project2D(intrinsics[0,:3,:3], R, T, lines3d)

        if self.white_bkgd:
            import pdb; pdb.set_trace()
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)
            N_samples_fine = z_vals_fine.shape[1]

        depth = torch.sum(weights*depth_ratio,dim=-1)
        output = {
            'points': points,
            'rgb_values': rgb_values,
            'sdf': weights.max(dim=-1)[0],
            'depth': depth,
            'xyz': torch.sum(points*weights[...,None],dim=1),
        }
        if self.use_fine_sample:
            output['rgb_values_coarse'] = rgb_values_coarse
            # import pdb; pdb.set_trace()

        """Learning Attraction Fields: BEGIN"""
        output['lines3d'] = lines3d
        output['lines2d'] = lines2d
        # output['sdf'] = points3d_sdf.flatten()
        output['wireframe-gt'] = input['wireframe']

        centers = torch.sum(lines3d,dim=1)*0.5
        centers2d = self.project2D(intrinsics[0,:3,:3], R, T, centers)

        center_dirs, center_loc = rend_util.get_camera_params(centers2d[None].detach(), pose, intrinsics)
        center_dirs = center_dirs.reshape(-1,3)
        
        _, cfeat = self.rendering_network(centers.detach(), center_dirs.detach())
        if self.use_center_network:
            lines3d_aux = self.center_network(centers.detach(), cfeat.detach())
        else:
            lines3d_aux = self.attraction_network(centers.detach(), center_dirs.detach(), cfeat.detach())
        # import pdb; pdb.set_trace()
        # with torch.no_grad():
            # _, cfeat = self.rendering_network(centers.detach(), center_dirs.detach())
        # lines3d_aux = self.attraction_network(centers, center_dirs, cfeat)
        lines2d_aux = self.project2D(intrinsics[0,:3,:3], R, T, lines3d_aux)
        output['lines3d-aux'] = [lines3d_aux]
        output['lines2d-aux'] = [lines2d_aux]

        if not self.training:
            output['lines3d'] = lines3d_aux
            output['lines2d'] = lines2d_aux
            output['lines3d-aux'] = [lines3d]
            output['lines2d-aux'] = [lines2d] 

            output['normal_map'] = F.normalize(torch.ones_like(rgb_values), dim=-1)

        """Learning Attraction Fields: END"""

        return output

    def volume_rendering(self, z_vals, sigma, depth_ratio):
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).cuda()], -1)  # [N_rays, N_samples]
        dists = dists * depth_ratio  # [N_rays, N_samples]

        noise = 0.
        if self.raw_noise_std > 0.:
            noise = torch.randn(sigma.shape) * raw_noise_std
            noise = noise.cuda()
        
        alpha = 1.-torch.exp(-F.relu(sigma+noise)*dists) # [N_rays, N_samples]
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        
        return weights

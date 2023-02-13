import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler

import open3d as o3d
import trimesh
from scipy.optimize import linear_sum_assignment
class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_out=False,

    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.inside_out = inside_out

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
        if self.inside_out:
            x[:, :1] = -x[:, :1]
        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf

class AttractionFieldNetwork(nn.Module):
    def __init__(self,
        feature_vector_size,
        d_in,
        d_out,
        dims,
        geometric_init = True,
        bias=1.0,
        weight_norm=True,
    ):
        super().__init__()
        dims = [d_in + feature_vector_size] + dims + [d_out]
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l+1]
            lin = nn.Linear(dims[l], out_dim)
            # if weight_norm:
                # lin = nn.utils.weight_norm(lin)
            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin"+str(l), lin)

        self.relu = nn.ReLU()
        
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward_from_emb(self, x):
        points = x[:,:3]
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        offsets = x[:,:6].reshape(-1,2,3)
        lines = points[:,None] + offsets
        # import pdb; pdb.set_trace()
        # logits = x[:,6:]
        return lines
    def forward(self, points, normals, feature_vectors):
        
        x = torch.cat((points,normals,feature_vectors),dim=-1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        offsets = x[:,:6].reshape(-1,2,3)
        lines = points[:,None] + offsets
        # import pdb; pdb.set_trace()
        # logits = x[:,6:]
        return lines

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

class VolSDFNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()

        self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.attraction_network = AttractionFieldNetwork(self.feature_vector_size, **conf.get_config('attraction_network'))

        self.density = LaplaceDensity(**conf.get_config('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))

        conf_junctions = conf.get_config('global_junctions')
        self.latents = nn.Parameter(torch.randn(conf_junctions.get_int('num_junctions',default=1024), conf_junctions.get_int('dim_hidden',default=256))*10)

        ffn = []
        for i in range(conf_junctions.get_int('num_layers',default=2)):
            ffn.append(nn.Linear(conf_junctions.get_int('dim_hidden',default=256), conf_junctions.get_int('dim_hidden',default=256)))
            ffn.append(nn.ReLU())
        ffn.append(nn.Linear(conf_junctions.get_int('dim_hidden',default=256), 3))

        self.ffn = nn.Sequential(*ffn)
        # self.ffn = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 3))
        # import pdb; pdb.set_trace()
        self.dbscan_enabled = conf.get_bool('dbscan_enabled', default=True)
        self.use_median = conf.get_bool('use_median', default=False)
        self.junction_eikonal = conf.get_bool('junction_eikonal', default=False)

    def project2D(self, K,R,T, points3d):
        shape = points3d.shape 
        assert shape[-1] == 3
        X = points3d.reshape(-1,3)
        
        x = K@(R@X.t()+T)
        x = x.t()
        x = x/x[:,-1:]
        x = x.reshape(*shape)[...,:2]
        return x
    
    def cluster_dbscan(self, points, eps=0.01, min_samples=2):
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        clustered_points = []
        for i in range(labels.max()+1):
            clustered_points.append(points[labels==i].mean(axis=0))
        clustered_points = torch.tensor(np.array(clustered_points)).float().cuda()
        
        return clustered_points
    def forward(self, input):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        rays_d = z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        depth_ratio = rays_d.norm(dim=-1)
        points = cam_loc.unsqueeze(1) + rays_d
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_flat)
        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights = self.volume_rendering(z_vals, sdf)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)
        sdf_ = sdf.reshape(weights.shape).detach()
        depth = torch.sum(weights*depth_ratio,dim=-1)
        output = {
            'points': points,
            'rgb_values': rgb_values,
            'sdf': sdf_,
            'depth': depth,
            'xyz': torch.sum(points*weights[...,None],dim=1),
        }

        """Learning Attraction Fields: BEGIN"""
        
        points3d = torch.sum(weights.unsqueeze(-1)*points, 1)
        points2d = uv.clone()[0]
        lines2d_gt = input['wireframe'][0].line_segments().cuda()
        points3d_sdf, points_features, points_gradients = self.implicit_network.get_outputs(points3d)

        # points3d_att, logits = self.attraction_network.forward(points3d, points_gradients, points_features)

        proj_mat = pose[0].inverse()[:3]
        R = proj_mat[:,:3]
        T = proj_mat[:,3:]

        lines3d  = self.attraction_network.forward(points3d.detach(), points_gradients.detach(), points_features.detach())
        # lines3d = torch.stack((e1,e2),dim=1)
        lines2d = self.project2D(intrinsics[0,:3,:3], R, T, lines3d.detach())
        lines2d_calib = self.project2D(torch.eye(3).cuda(), R, T, lines3d)
        
        if self.training:
            
            if self.dbscan_enabled:
                junctions3d = self.cluster_dbscan(lines3d.detach().cpu().numpy().reshape(-1,3),eps=0.01,min_samples=2)
            else:
                junctions3d = lines3d.detach().reshape(-1,3)
            junctions2d = self.project2D(intrinsics[0,:3,:3], R, T, junctions3d)
            junctions2d_calib = self.project2D(torch.eye(3).cuda(), R, T, junctions3d)
            junctions2d_gt = input['wireframe'][0].vertices.cuda()
            jcost = torch.sum((junctions2d[None]-junctions2d_gt[:,None])**2,dim=-1).sqrt()
            jassign = linear_sum_assignment(jcost.cpu())
            
            if self.use_median:
                median = jcost[jassign[0],jassign[1]].detach().median()
                is_correct = jcost[jassign[0],jassign[1]]<median
                output['median'] = median
            else:
                is_correct = jcost[jassign[0],jassign[1]]<10
            junctions3d = junctions3d[jassign[1]][is_correct]
            junctions2d = junctions2d[jassign[1]][is_correct]
            junctions2d_calib = junctions2d_calib[jassign[1]][is_correct]
            # junctions2d = junctions2d_gt[jassign[0]][is_correct]

            junctions3d_global = self.ffn(self.latents)
            junctions2d_global = self.project2D(intrinsics[0,:3,:3], R, T, junctions3d_global)
            junctions2d_global_calib = self.project2D(torch.eye(3).cuda(), R, T, junctions3d_global)
            output['j2d_local'] = junctions2d
            output['j3d_local'] = junctions3d
            output['j3d_global'] = junctions3d_global
            output['j2d_global'] = junctions2d_global
            output['j2d_local_calib'] = junctions2d_calib
            output['j2d_global_calib'] = junctions2d_global_calib

        if not self.training:
            line_ray_d, line_ray_o = rend_util.get_camera_params(input['uv_proj'], pose, intrinsics)
            line_ray_d = line_ray_d.reshape(-1,3)
            line_ray_o = line_ray_o.repeat(line_ray_d.shape[0],1)
            denominator = torch.sum(line_ray_d*points_gradients,dim=-1)
            denom_eps = torch.where(denominator>=0, 
                torch.ones_like(denominator)*1e-6,
                -torch.ones_like(denominator)*1e-6
                )
            t = torch.sum((points3d-line_ray_o)*points_gradients,dim=-1)/(denominator+denom_eps)
            t = t.detach()
            l3d = line_ray_o + line_ray_d*t.unsqueeze(-1)
            points3d_sdf, points_features, points_gradients = self.implicit_network.get_outputs(l3d)
            lines3d  = self.attraction_network.forward(l3d.detach(), points_gradients.detach(), points_features.detach())
            lines2d = self.project2D(intrinsics[0,:3,:3], R, T, lines3d)
            output['l3d'] = l3d

        output['points3d'] = points3d
        # output['points3d_att'] = points3d_att
        output['lines3d'] = lines3d
        # output['lines3d-aux'] = lines3d_aux
        output['lines2d_calib'] = lines2d_calib
        output['lines2d'] = lines2d
        # output['lines2d-aux'] = lines2d_aux
        output['sdf'] = points3d_sdf.flatten()
        output['wireframe-gt'] = input['wireframe']
        output['K'] = intrinsics[0,:3,:3]

       
        """Learning Attraction Fields: END"""

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)

            if self.junction_eikonal:
                eikonal_points = torch.cat([eikonal_points, junctions3d_global], 0)
            grad_theta = self.implicit_network.gradient(eikonal_points)
            output['grad_theta'] = grad_theta


        if not self.training:
            gradients = gradients.detach()
            normals = gradients / gradients.norm(2, -1, keepdim=True)
            normals = normals.reshape(-1, N_samples, 3)
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)

            output['normal_map'] = normal_map

        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here

        return weights

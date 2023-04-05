import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler

import open3d as o3d
import trimesh
from scipy.optimize import linear_sum_assignment

from pyhocon import ConfigTree
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

        conf_junctions = conf.get_config('global_junctions', default=ConfigTree())

        ffn = []
        num_layers = conf_junctions.get_int('num_layers',default=2)
        # self.latents = nn.Parameter(torch.randn(conf_junctions.get_int('num_junctions',default=1024), conf_junctions.get_int('dim_hidden',default=256)))
        self.latents = nn.Parameter(torch.empty(conf_junctions.get_int('num_junctions',default=1024), conf_junctions.get_int('dim_hidden',default=256)))
        # torch.nn.init.normal_(self.latents, mean=0.0, std=2**num_layers)
        torch.nn.init.normal_(self.latents, mean=0.0, std=1)
        for i in range(num_layers+1):
            if i != num_layers:
                dim_in = conf_junctions.get_int('dim_hidden',default=256)
                dim_out = conf_junctions.get_int('dim_hidden',default=256)
            else:
                dim_in = conf_junctions.get_int('dim_hidden',default=256)
                dim_out = 3
            lin = nn.Linear(dim_in,dim_out)
            # if conf_junctions.get_bool('geometric_init',default=False):
            #     if i != num_layers:
            #         torch.nn.init.constant_(lin.bias, 0.0)
            #         torch.nn.init.normal_(lin.weight, mean=0.0, std=np.sqrt(2)/np.sqrt(dim_out))
            #     else:
            #         torch.nn.init.constant_(lin.bias, 0.0)
            #         torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi)/np.sqrt(dim_out), std=0.0001)
            # if conf_junctions.get_bool('weight_norm',default=False):
            #     lin = nn.utils.weight_norm(lin)
            ffn.append(lin)

            if i!= num_layers:
                ffn.append(nn.ReLU())
        # ffn.append(nn.Linear(conf_junctions.get_int('dim_hidden',default=256), 3))

        self.ffn = nn.Sequential(*ffn)
        # self.ffn = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 3))
        # temp = self.ffn(self.latents)
        # import pdb; pdb.set_trace()
        self.dbscan_enabled = conf.get_bool('dbscan_enabled', default=True)
        self.use_median = conf.get_bool('use_median', default=False)
        self.junction_eikonal = conf.get_bool('junction_eikonal', default=False)
        self.use_l3d = conf.get_bool('use_l3d', default=False)

        self.obj = trimesh.load('../data/abc/00075213/00075213_9d4173fb39d54d16a718fe39_trimesh_007.obj')
        vertices = np.array(self.obj.vertices)
        center = 0.5*(vertices.max(axis=0) - vertices.min(axis=0))
        scale = max(vertices.max(axis=0) - vertices.min(axis=0))
        transform = np.diag([1/scale, 1/scale, 1/scale, 1])
        transform[:3,3] = -center/scale
        self.obj = self.obj.apply_transform(transform)
        
    def project2D(self, K,R,T, points3d):
        shape = points3d.shape 
        assert shape[-1] == 3
        X = points3d.reshape(-1,3)
        
        x = K@(R@X.t()+T)
        x = x.t()
        # sign = x[:,-1:]>=0
        # x = x/x[:,-1:]
        denominator = x[:,-1:]
        sign = torch.where(denominator>=0, torch.ones_like(denominator), -torch.ones_like(denominator))
        eps = torch.where(denominator.abs()<1e-8, torch.ones_like(denominator)*1e-8, torch.zeros_like(denominator))
        x = x/(denominator+eps*sign)
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

    def render_rgb(self, input):
        assert not self.training
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
        return rgb_values
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
        
        scene = trimesh.Scene()

# Add the mesh to the scene with transparency
        # mesh_visual = trimesh.visual.ColorVisuals(
        #     # vertices=self.obj.vertices,
        #     # faces=self.obj.faces,
        #     face_colors=[255, 255, 255, int(255 * 0.3)]
        # )
        # scene.add_geometry(self.obj, visual=mesh_visual)
        
        lines3d = self.attraction_network.forward(points_flat, gradients, dirs_flat, feature_vectors)
        lines3d = lines3d.reshape(-1, N_samples, 2,3)
        lines3d = torch.sum(weights[:,:,None,None].detach()*lines3d,dim=1)
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)
        sdf_ = sdf.reshape(weights.shape).detach()
        depth = torch.sum(weights*depth_ratio,dim=-1)


        # rays_trimesh = trimesh.load_path(points[:,[0,-1]].cpu().detach().numpy()).show()
        trimesh_sampled_points = []
        for points_along_ray, weights_along_ray, rgb_along_ray in zip(points, weights, rgb_values):
            # pcd = trimesh.points.PointCloud(points_along_ray.cpu().numpy())
            pcd = trimesh.load_path(points_along_ray[[0,-1]].cpu().numpy())
            pcd.colors = [[255, 140, 0]]
            trimesh_sampled_points.append(pcd)
        
        scene = trimesh.Scene([])
        scene.add_geometry(self.obj,'mesh')
        from scipy.spatial.transform import Rotation 
        rot = Rotation.from_matrix(pose[0][:3,:3].cpu())
        rote = rot.as_euler('xyz', degrees=True)
        scene.set_camera(angles=rote, center=pose[0][:3,3].cpu().numpy(), resolution=(1200, 1600), fov=(60, 60))
        import pdb; pdb.set_trace()
        # viewer = trimesh.viewer.windowed.SceneViewer(scene=scene)
        # scene = trimesh.Scene([trimesh_sampled_points,self.obj]).show()

        proj_mat = pose[0].inverse()[:3]
        R = proj_mat[:,:3]
        T = proj_mat[:,3:]
        K = intrinsics[0,:3,:3]

        rays3d = points[:,[0,-1]]
        rays2d = self.project2D(K,R,T,rays3d).cpu().numpy()
        lines2d = self.project2D(K,R,T,lines3d).detach().cpu().numpy()

        gjc3d = self.ffn(self.latents).detach()
        gjc2d = self.project2D(K,R,T,gjc3d).cpu().numpy()
        import matplotlib.pyplot as plt
        plt.plot([rays2d[:,0,0],rays2d[:,1,0]],[rays2d[:,0,1],rays2d[:,1,1]],'r-')
        plt.plot([lines2d[:,0,0],lines2d[:,1,0]],[lines2d[:,0,1],lines2d[:,1,1]],'b-')
        plt.plot(gjc2d[:,0],gjc2d[:,1],'g.')

        import pdb; pdb.set_trace()

        output = {
            'points': points,
            'rgb_values': rgb_values,
            'sdf': sdf_,
            'depth': depth,
            'xyz': torch.sum(points*weights[...,None],dim=1),
        }

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

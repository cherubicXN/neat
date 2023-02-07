import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler

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

        self.density = LaplaceDensity(**conf.get_config('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))

    def render(self, uv, pose, intrinsics):
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        # import pdb; pdb.set_trace()
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

        points3d = torch.sum(points*weights.unsqueeze(-1),dim=-2)

        points3d = points3d.reshape(*uv.shape[:-1],-1)

        return points3d
    def forward_minstance(self, input):
        intrinsics = input["intrinsics"]
        juncs2d = input["juncs2d"]
        edges = input['edges']
        weights = input['weights']
        pose = input["pose"]
        lines2d = juncs2d[0,edges[0]][None]
        # lines2d_tangent = lines2d[:,:,1:] - lines2d[:,:,:1]
        # lines2d_len = lines2d_tangent.norm(dim=-1,keepdim=True)
        # lines2d_tangent = lines2d_tangent / torch.sum(lines2d_tangent**2,dim=-1,keepdim=True).sqrt()
        # lines2d_tangent *= 0.05
        # lines2d_normal = torch.cat((-lines2d_tangent[...,1:],lines2d_tangent[...,:1]),dim=-1)

        # n_points = 1 if self.training else 16
        n_points = 3
        num_lines = lines2d.shape[1]
        # if self.training:
            # n_points = 16

        # lambdas = torch.linspace(0,1,n_points,device=intrinsics.device).reshape(1,1,-1,1)

        lambdas = torch.rand(num_lines,device=intrinsics.device).reshape(1,-1,1,1)
        lambdas = torch.cat((lambdas*0,lambdas,lambdas*0+1),dim=-2)

        lines2d_points = lines2d[:,:,:1] + lambdas*(lines2d[:,:,1:]-lines2d[:,:,:1])
        # lines2d_points_delta_a = lines2d_points + lines2d_normal*5
        # lines2d_points_delta_b = lines2d_points - lines2d_normal*5
        uv = lines2d_points.reshape(-1,num_lines*n_points,2)
        points3d = self.render(lines2d_points.reshape(-1,num_lines*n_points,2), pose, intrinsics)

        lines3d = points3d.reshape(1,num_lines,n_points,-1)

        x1 = lines3d[:,:,:1]
        x2 = lines3d[:,:,-1:]
        x0 = lines3d[:,:,1:-1]
        
        norm2 = torch.sum((x2-x1)**2,dim=-1,keepdim=True)
        with torch.no_grad():
            t =  -(x1-x0)*(x2-x1)/norm2
            t = t.clamp(min=0,max=1.0)
            xp = x1 + (x2-x1)*t

        loss = torch.nn.functional.l1_loss(x0,xp,reduction='none')

        total_loss = torch.mean(loss.sum(dim=-1).sum(dim=-1)*weights)


        #

        return total_loss
    def forward_two_view(self, input):
        intrinsics = input["intrinsics"]
        juncs2d = input["juncs2d"]
        edges = input['edges']
        weights = input['weights']
        pose = input["pose"]
        lines2d = juncs2d[0,edges[0]][None]
        lines2d_tangent = lines2d[:,:,1:] - lines2d[:,:,:1]
        # lines2d_len = lines2d_tangent.norm(dim=-1,keepdim=True)
        # lines2d_tangent = lines2d_tangent / torch.sum(lines2d_tangent**2,dim=-1,keepdim=True).sqrt()
        lines2d_tangent *= 0.05
        lines2d_normal = torch.cat((-lines2d_tangent[...,1:],lines2d_tangent[...,:1]),dim=-1)

        # n_points = 1 if self.training else 16
        n_points = 16
        num_lines = lines2d.shape[1]
        # if self.training:
            # n_points = 16

        lambdas = torch.linspace(0,1,n_points,device=intrinsics.device).reshape(1,1,-1,1)
        # if self.training:
            # lambdas = torch.rand((num_lines,n_points),device='cuda')
            # lambdas = torch.sort(lambdas,dim=-1)[0].reshape(1,num_lines,n_points,1)

        lines2d_points = lines2d[:,:,:1] + lambdas*(lines2d[:,:,1:]-lines2d[:,:,:1])
        # lines2d_points_delta_a = lines2d_points + lines2d_normal*5
        # lines2d_points_delta_b = lines2d_points - lines2d_normal*5
        uv = lines2d_points.reshape(-1,num_lines*n_points,2)

        points2d = lines2d_points.reshape(-1,num_lines*n_points,2)

        chunksize = 2048
        points3d = [self.render(pp,pose,intrinsics).detach() for pp in points2d.split(chunksize,dim=1)]
        points3d = torch.cat(points3d,dim=1)
        # points3d = self.render(lines2d_points.reshape(-1,num_lines*n_points,2), pose, intrinsics)

        lines3d = points3d.reshape(1,num_lines,n_points,-1)

        lines3d_dir = lines3d[:,:,-1:] - lines3d[:,:,:1]
        lines3d_dir = lines3d_dir/(torch.sum(lines3d_dir**2,dim=-1,keepdim=True)+1e-10).sqrt()
        lines3d_subdir = lines3d[:,:,1:] - lines3d[:,:,:-1]
        lines3d_subdir = lines3d_subdir/(torch.sum(lines3d_subdir**2,dim=-1,keepdim=True)+1e-10).sqrt()

        loss = torch.sum(lines3d_subdir*lines3d_dir,dim=-1)
        loss = torch.nn.functional.l1_loss(loss, torch.ones_like(loss),reduction='none')

        total_loss = torch.mean(loss.mean(dim=-1)*weights)

        # w2c = pose.inverse()[:,:3,:4]
        # proj_mat = intrinsics[:,:3,:3]@w2c

        # lines2d_reproj = (proj_mat[:,:,:3]@points3d.transpose(1,2)+proj_mat[:,:,3:]).transpose(1,2)
        # lines2d_reproj = lines2d_reproj/lines2d_reproj[...,-1:]
        # lines2d_reproj = lines2d_reproj.reshape(1,num_lines,n_points,-1)
        # points3d_side_a = self.render(lines2d_points_delta_a.reshape(-1,num_lines*n_points,2), pose, intrinsics)
        # points3d_side_b = self.render(lines2d_points_delta_b.reshape(-1,num_lines*n_points,2), pose, intrinsics)
        
        # points3d = points3d.detach()
        # points3d_side_a = points3d_side_a.detach()
        # points3d_side_b = points3d_side_b.detach()
        # points3d_tang_a = points3d_side_a-points3d
        # points3d_tang_a = points3d_tang_a/(torch.norm(points3d_tang_a,dim=-1,keepdim=True)+1e-10)
        # points3d_tang_b = points3d_side_b-points3d
        # points3d_tang_b = points3d_tang_b/(torch.norm(points3d_tang_b,dim=-1,keepdim=True)+1e-10)
        # points3d_norm_a = self.implicit_network.gradient(points3d_side_a.reshape(-1,3)).reshape(1,-1,3)#.detach()
        # points3d_norm_a = points3d_norm_a/(points3d_norm_a.norm(dim=-1,keepdim=True)+1e-10)
        # points3d_norm_b = self.implicit_network.gradient(points3d_side_b.reshape(-1,3)).reshape(1,-1,3)#.detach()
        # points3d_norm_b = points3d_norm_b/(points3d_norm_b.norm(dim=-1,keepdim=True)+1e-10)
            
        if not self.training:
            import trimesh
            pc = trimesh.points.PointCloud(points3d[0].cpu().detach())
            # tan1 = trimesh.load_path(torch.stack((points3d[1],points3d_side_a[1]),dim=1).cpu().detach())
            # tan2 = trimesh.load_path(torch.stack((points3d[1],points3d_side_b[1]),dim=1).cpu().detach())
            # nor1 = trimesh.load_path(torch.stack((points3d_side_a[1],points3d_side_a[1] + points3d_norm_a[0]*0.001),dim=1).cpu().detach())
            # nor2 = trimesh.load_path(torch.stack((points3d_side_b[1],points3d_side_b[1]+points3d_norm_b[0]*0.001),dim=1).cpu().detach())

            # scene = trimesh.Scene([tan1,tan2,pc,nor1,nor2])

            # plt.plot([lines2d[0,:,0,0].cpu().numpy(),lines2d[0,:,1,0].cpu().numpy()],
            # [lines2d[0,:,0,1].cpu().numpy(),lines2d[0,:,1,1].cpu().numpy()],
            # 'r-'
            # )
            # plt.show()
            return pc,lines3d.cpu(),lines2d_points.cpu()
        # loss_a = (torch.sum(points3d_tang_a*points3d_norm_a,dim=-1)**2)
        # loss_a = loss_a.reshape(1,num_lines,n_points)
        # loss_b = (torch.sum(points3d_tang_b*points3d_norm_b,dim=-1)**2)
        # loss_b = loss_b.reshape(1,num_lines,n_points)

        # total_loss = (loss_a*weights.unsqueeze(-1)).mean() + (loss_b*weights.unsqueeze(-1)).mean() 
        """ 
        print(loss_a,loss_b)
        
        """

        return total_loss

        # rgb = rgb_flat.reshape(-1, N_samples, 3)
    def forward(self, input):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        # import pdb; pdb.set_trace()
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
        weights_max, weights_argmax = weights.detach().max(dim=-1)
        idx_arange = torch.arange(weights.shape[0],device=weights.device)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        # white background assumption
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

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)

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

import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from collections import defaultdict
class AverageMeter(object):
    def __init__(self):
        self.loss_dict = defaultdict(list)
        
    def push(self, loss_dict):
        with torch.no_grad():
            for key, val in loss_dict.items():
                self.loss_dict[key].append(val)
    
    def __call__(self):
        out_dict = {}
        for key, val in self.loss_dict.items():
            out_dict[key] = sum(val)/len(val)
        
        return out_dict
class VolSDFTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        if not kwargs.get('test', False):
            utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        if not kwargs.get('test', False):
            utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        if not kwargs.get('test', False):
            utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        if not kwargs.get('test', False):
            utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        if not kwargs.get('test', False):
            utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        if not kwargs.get('test', False):
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)

        self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))
        if scan_id < 24 and scan_id > 0: # BlendedMVS, running for 200k iterations
            self.nepochs = int(200000 / self.ds_len)
            print('RUNNING FOR {0}'.format(self.nepochs))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=False,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))

        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def valid(self):
        self.model.eval()
        
        import trimesh
        points_all = []
        lines_all = []
        from tqdm import tqdm 
        for indices, model_input, ground_truth in tqdm(self.plot_dataloader):
            for key in model_input:
                if isinstance(model_input[key],torch.Tensor):
                    model_input[key] = model_input[key].cuda()
            pc, lines3d, lines2d = self.model.forward_two_view(model_input)

            x1 = lines3d[:,:,:1]
            x2 = lines3d[:,:,-1:]
            x0 = lines3d[:,:,1:-1]

            x0m1 = x0 - x1
            x0m2 = x0 - x2

            temp = torch.sum(torch.cross(x0m1,x0m2)**2,dim=-1)
            length = torch.sum((x2-x1)**2,dim=-1)

            t = -(x1-x0)*(x2-x1)/length[:,:,None]
            t = t.clamp(min=0,max=1.0)
            xp = x1 + (x2-x1)*t
            xp = xp.detach()

            cost = torch.norm(xp-x0,dim=-1)
            # cost = torch.sqrt(temp/length).max(dim=-1)[0]
            cost = cost.max(dim=-1)[0]
            # cost = (temp/length).mean(dim=-1)
            num_lines = lines3d.shape[1]

            idx = cost<5e-2
            # idx = cost<0.01*length.sqrt()[...,0]
            if idx.sum()>0:
                paths = torch.cat((x1,x2),dim=2)[0].detach()
                paths = paths[idx[0]]
                lines_all.append(trimesh.load_path(paths.cpu()))

            # for i in range(num_lines):
            #     tpc = trimesh.points.PointCloud(lines3d[0,i].detach().cpu())
            #     tlc = trimesh.load_path(lines3d[0,i,[0,-1]].detach().cpu())
            #     scene = trimesh.Scene([tpc,tlc])
            #     print(cost[0,i])
            #     scene.show()
            points_all.append(pc)
        trimesh.Scene()
        scene = trimesh.Scene(points_all)
        scene.show()

        trimesh.Scene(lines_all).show()
            # points_ = self.model.forward_two_view(model_input)

            # points_all.append(points_.cpu().numpy())

    def run(self,use_pl_loss=False):
        print("training...")

        if self.do_vis:
            from utils.plots import get_3D_quiver_trace
            import plotly.graph_objs as go
            import plotly.offline as offline
            data = []
            for indices, model_input, ground_truth in self.plot_dataloader:
                cam_loc, cam_dir = rend_util.get_camera_for_plot(model_input['pose'])
                data.append(
                    get_3D_quiver_trace(cam_loc, cam_dir,
                    name='camera_({0})'.format(indices.item()))
                )
            fig = go.Figure(data=data)
            scene_dict = dict(xaxis=dict(range=[-6, 6], autorange=False),
                      yaxis=dict(range=[-6, 6], autorange=False),
                      zaxis=dict(range=[-6, 6], autorange=False),
                      aspectratio=dict(x=1, y=1, z=1))
            fig.update_layout(scene=scene_dict, width=1200, height=1200, showlegend=True)
            filename = '{0}/cameras.html'.format(self.plots_dir)
            offline.plot(fig, filename=filename, auto_open=False)

        for epoch in range(self.start_epoch, self.nepochs + 1):

            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoints(epoch)

            if self.do_vis and epoch % self.plot_freq == 0:
                self.model.eval()

                self.train_dataset.change_sampling_idx(-1)
                indices, model_input, ground_truth = next(iter(self.plot_dataloader))

                # model_input["intrinsics"] = model_input["intrinsics"].cuda()
                # model_input["uv"] = model_input["uv"].cuda()
                # model_input['pose'] = model_input['pose'].cuda()
                for key in model_input:
                    if isinstance(model_input[key],torch.Tensor):
                        model_input[key] = model_input[key].cuda()
                split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)

                res = []
                for s in tqdm(split):
                    out = self.model(s)
                    d = {'rgb_values': out['rgb_values'].detach(),
                         'normal_map': out['normal_map'].detach()}
                    res.append(d)

                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                plot_data = self.get_plot_data(model_outputs, model_input['pose'], ground_truth['rgb'])
                plt.plot(self.model.implicit_network,
                         indices,
                         plot_data,
                         self.plots_dir,
                         epoch,
                         self.img_res,
                         **self.plot_conf
                         )

                self.model.train()

            self.train_dataset.change_sampling_idx(self.num_pixels)

            loss_meters = AverageMeter()
            
            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                # model_input["intrinsics"] = model_input["intrinsics"].cuda()
                # model_input["uv"] = model_input["uv"].cuda()
                # model_input['pose'] = model_input['pose'].cuda()
                for key in model_input:
                    if isinstance(model_input[key],torch.Tensor):
                        model_input[key] = model_input[key].cuda()

                model_outputs = self.model(model_input)
                loss_output = self.loss(model_outputs, ground_truth)

                pl_loss = self.model.forward_minstance(model_input)
                loss_output['pl_loss'] = pl_loss
                # loss = loss_output['loss'] + odel_outputs['pl_loss']*0.1
                loss = loss_output['loss'] + pl_loss*use_pl_loss
                loss_meters.push(loss_output)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1,3))
                loss_meters.push({'psnr': psnr})
                loss_msg = []
                for key in loss_output.keys():
                    loss_msg.append('{} = {:.3f} ({:.3f})'.format(key, loss_output[key].item(),loss_meters()[key].item()))
                loss_msg = ' '.join(loss_msg)

                print(
                    '{0}_{1} [{2}] ({3}/{4}): {5}, psnr = {6} ({7:.3f})'
                        .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, 
                        loss_msg,
                        psnr.item(),
                                loss_meters()['psnr'].item(),
                                ))

                self.train_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()

        self.save_checkpoints(epoch)

    def get_plot_data(self, model_outputs, pose, rgb_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.

        plot_data = {
            'rgb_gt': rgb_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
        }

        return plot_data

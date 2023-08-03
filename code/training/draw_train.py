import os
import git
from datetime import datetime
from pyhocon import ConfigFactory
from pyhocon.converter import HOCONConverter
import sys
import torch
from tqdm import tqdm

# import pdb; pdb.set_trace()
import utils.general as utils
import utils.plots as plt
from utils import rend_util
from collections import defaultdict
import logging
import json
try:
    import wandb
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False
    pass

# backward hook with module name
def get_backward_hook(module_name: str):
    
    class BackwardHook:
        name: str
            
        def __init__(self, name):
            self.name = name
            
        def __call__(self, module, grad_input, grad_output):
            for i, g_in in enumerate(grad_input):
                # if self.name == 'attraction_network.relu':
                if not isinstance(g_in, torch.Tensor):
                    continue
                    
                # print(module_name, torch.any(torch.isnan(g_in)))
                if torch.any(torch.isnan(g_in)):
                    print(module_name, torch.any(torch.isnan(g_in)))
                    print(f"{module_name}'s {i}th input gradient is nan")
                    import pdb; pdb.set_trace()
            for i, g_out in enumerate(grad_output):
                if torch.any(torch.isnan(g_out)):
                    print(module_name, torch.any(torch.isnan(g_in)))
                    print(f"{module_name}'s {i}th output gradient is nan")
                    import pdb; pdb.set_trace()
                
    return BackwardHook(module_name)
class AverageMeter(object):
    def __init__(self):
        self.loss_dict = defaultdict(list)
        
    @torch.no_grad()
    def push(self, loss_dict):
        with torch.no_grad():
            for key, val in loss_dict.items():
                self.loss_dict[key].append(val)
    @torch.no_grad()
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
        self.conf.put('train.expname', self.expname)
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '/{0}'.format(scan_id)
            self.conf.put('dataset.scan_id', scan_id)

        self.repo =  git.Repo(search_parent_directories=True)
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

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        # utils.mkdir_ifnotexists(self.expdir)
        os.makedirs(self.expdir,exist_ok=True)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        # create junction dirs
        self.junctions_path = os.path.join(self.expdir, self.timestamp, 'junctions')
        utils.mkdir_ifnotexists(self.junctions_path)

        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        
        with open(os.path.join(self.expdir, self.timestamp, 'runconf.conf'), 'w') as f:
            f.write(HOCONConverter.convert(self.conf))
        
        self.repo.index.add(os.path.abspath(os.path.join(self.expdir, self.timestamp, 'runconf.conf')))
        self.repo.index.commit('new experiment {0}'.format(self.expdir),committer= git.Actor(name='expbot',email='expbot'))
        # os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

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
        # if kwargs['fixed_data_len']:
            # self.ds_len = kwargs['fixed_data_len']
        # train_sampler =  torch.utils.data.RandomSampler(self.train_dataset,num_samples=self.ds_len)
        # batch_sampler = torch.utils.data.BatchSampler(train_sampler, 1, False)

        # if scan_id < 24 and scan_id > 0: # BlendedMVS, running for 200k iterations
            # self.nepochs = int(200000 / self.ds_len)
            # print('RUNNING FOR {0}'.format(self.nepochs))
        if dataset_conf['data_dir'] == 'BlendedMVS':
            self.nepochs = int(200000 / self.ds_len)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=False,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
                                                            
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )
        
        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)
        if torch.cuda.is_available():
            self.model.cuda()
        # for name, module in self.model.named_modules():
            # import pdb; pdb.set_trace()
            # module.register_full_backward_hook(get_backward_hook(name))

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
            load_res = self.model.load_state_dict(saved_model_state["model_state_dict"],strict=False)
            # if load_res.unexpected_keys:
            #     for key in load_res.unexpected_keys:
            #         saved_model_state['model_state_dict'].pop(key)

            #     print('Unexpected keys in model state dict: {0} are removed'.format(load_res.unexpected_keys))
            #     self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            # self.optimizer.load_state_dict(data["optimizer_state_dict"])#,strict=False)

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            # self.scheduler.load_state_dict(data["scheduler_state_dict"])#,strict=False)

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')

        logger = logging.getLogger('train')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.FileHandler(os.path.join(self.expdir,self.timestamp, 'train.log'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        self.logger = logger
        self.log_freq = 1 if kwargs['verbose'] else len(self.train_dataloader)
        if kwargs['wandb'] and WANDB_AVAILABLE:
            wandb.init(project="NEAT", 
                name='{}/{}/{}'.format(
                    self.conf.train.expname,
                    scan_id,
                    self.timestamp),
                config=self.conf.as_plain_ordered_dict(), 
                dir=self.expdir)
            self.wandb = True
        else:
            self.wandb = False

    def commit_log(self, msg='update log'):
        log_path = os.path.join(self.expdir,self.timestamp, 'train.log')
        self.repo.index.add(os.path.abspath(log_path))
        self.repo.index.commit(msg,committer= git.Actor(name='expbot',email='expbot'))

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

    def run(self):
        print("training...")

        # for epoch in range(self.start_epoch, self.nepochs + 1):

        #     if epoch % self.checkpoint_freq == 0:
        #         self.save_checkpoints(epoch)
        #         self.commit_log('checkpoint at epoch {}'.format(epoch))

        #     self.model.eval()

        #     if hasattr(self.model, 'latents'):
        #         with torch.no_grad():
        #             global_junctions = self.model.ffn(self.model.latents).cpu()
        #         torch.save(global_junctions, os.path.join(self.junctions_path, str(epoch) + '.pth'))
                

        self.train_dataset.change_sampling_idx(self.num_pixels)

        self.model.train()
        for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
            if data_index < 21:
                continue
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input['pose'] = model_input['pose'].cuda()

            model_outputs = self.model(model_input)
            

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

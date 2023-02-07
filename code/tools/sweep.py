import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm

from collections import defaultdict
import logging
import argparse
import shutil
import os.path as osp
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default=None)
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument("--exps_folder", type=str, default="exps")
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--epoch', type=int, default=0)

    opt = parser.parse_args()

    if opt.conf is not None:
        conf = ConfigFactory.parse_file(opt.conf)
        expname = conf.get_string('train.expname') + opt.expname
        scan_id = opt.scan_id if opt.scan_id != -1 else conf.get_int('dataset.scan_id',default=-1)
        if scan_id != -1:
            expname = expname + '/{}'.format(scan_id)

        exps_folder = os.path.join('..',opt.exps_folder,expname)
    else:
        exps_folder = opt.expname
    if os.path.exists(exps_folder):    
        timestamps = os.listdir(exps_folder)
        timestamps = [t for t in timestamps if t[0]!='.']


    num_removed = 0
    for t in timestamps:
        ckpt_folder = osp.join(exps_folder,t,'checkpoints','ModelParameters')
        try:
            ckpts = os.listdir(ckpt_folder)
        except:
            ckpts = []
        if len(ckpts) == 0:
            shutil.rmtree(osp.join(exps_folder,t))
            print('{} is removed as it is empty'.format(t))
            num_removed += 1
            continue

        latest_id = ckpts.index('latest.pth')
        ckpts.pop(latest_id)

        ckpts = sorted(ckpts)
        epochs = sorted([int(c[:-4]) for c in ckpts])
        max_epoch = epochs[-1]
        print('{} checkpoints are in {}'.format(len(ckpts),t))
        print('{} is the last one'.format(max_epoch))

        if max_epoch <= opt.epoch:
            shutil.rmtree(osp.join(exps_folder,t))
            num_removed += 1
            print('{} is removed as its max_epoch is {}.pth'.format(t,max_epoch))
    
    if len(timestamps) == num_removed:
        print('Removing {} because it is empty'.format(exps_folder))
        shutil.rmtree(exps_folder)
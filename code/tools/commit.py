from __future__ import print_function, unicode_literals

import argparse
import os
import os.path as osp

from PyInquirer import prompt, print_json

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str,required=True)
    
    opt = parser.parse_args()

    exp = opt.exp

    ckpt_list = os.listdir(osp.join(exp,'checkpoints','ModelParameters'))
    ckpt_list = sorted([int(c[:-4]) for c in ckpt_list if c != 'latest.pth'])


    question = [{
            'type': 'list',
            'name': 'checkpoint',
            'message': 'Which checkpoint would you like to commit?',
            'choices': ['{}.pth'.format(c) for c in ckpt_list]
        }]
    answer  = prompt(question)

    answer['checkpoint'] = osp.join(exp,'checkpoints','ModelParameters',answer['checkpoint'])


    _ = prompt([
        {
            'type': 'list',
            'name': 'wireframe',
            'message': 'Which wireframe model would you like to commit?',
            'choices': [ f for f in os.listdir(osp.join(exp,'wireframes')) if f.endswith('npz')]
        }
    ])
    answer.update(_)
    answer['wireframe'] = osp.join(exp,'wireframes',answer['wireframe'])

    answer['trainlog'] = osp.join(exp,'train.log')

    for value in answer.values():
        git_cmd = "git add -f {}".format(value)
        # print(git_cmd)
        os.system(git_cmd)
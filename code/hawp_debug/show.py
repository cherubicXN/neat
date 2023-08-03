import sys
sys.path.append('../code')
import os

from datetime import datetime
from pyhocon import ConfigFactory
import torch
from tqdm import tqdm

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils import hawp_util
from collections import defaultdict
import cv2
import matplotlib.pyplot as plot 
import json


ROOT = '../data/nerf/lego'

IM_ROOT = os.path.join(ROOT,'images')
WF_ROOT = os.path.join(ROOT,'hawp')

fnames = sorted(utils.glob_imgs(IM_ROOT))

for f in fnames:
    basename = os.path.basename(f)

    image = cv2.imread(f)

    hawp_path = os.path.join(WF_ROOT,basename.replace('.png','.json'))

    wireframe = hawp_util.WireframeGraph.load_json(hawp_path)

    line_segments = wireframe.line_segments(threshold=0.05)

    line_segments = line_segments[:,:4].cpu().numpy()

    plot.imshow(image[...,::-1])
    plot.plot([line_segments[:,0],line_segments[:,2]],[line_segments[:,1],line_segments[:,3]],'r-')
    plot.show()

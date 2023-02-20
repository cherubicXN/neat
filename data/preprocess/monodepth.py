import torch
import numpy as np
import cv2
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import sys
from torchvision import transforms
import PIL
from PIL import Image

def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
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

def align_x(depth1, depth2, s1, e1, s2, e2):
    assert depth1.shape[0] == depth2.shape[0]
    assert depth1.shape[1] == depth2.shape[1]

    assert (e1 - s1) == (e2 - s2)
    # aligh depth2 to depth1
    scale, shift = compute_scale_and_shift(depth2[:, :, s2:e2], depth1[:, :, s1:e1], torch.ones_like(depth1[:, :, s1:e1]))

    depth2_aligned = scale * depth2 + shift   
    result = torch.ones((1, depth1.shape[1], depth1.shape[2] + depth2.shape[2] - (e1 - s1)))

    result[:, :, :s1] = depth1[:, :, :s1]
    result[:, :, depth1.shape[2]:] = depth2_aligned[:, :, e2:]

    weight = np.linspace(1, 0, (e1-s1))[None, None, :]
    result[:, :, s1:depth1.shape[2]] = depth1[:, :, s1:] * weight + depth2_aligned[:, :, :e2] * (1 - weight)

    return result

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

    parser.add_argument('--omnidata_path', dest='omnidata_path', help="path to omnidata model")
    parser.set_defaults(omnidata_path='/home/xn/repo/omnidata/omnidata_tools/torch/')

    parser.add_argument('--pretrained_models', dest='pretrained_models', help="path to pretrained models")
    parser.set_defaults(pretrained_models='/home/xn/repo/omnidata/omnidata_tools/torch/pretrained_models/')

    parser.add_argument('--task', dest='task', help="normal or depth")
    parser.set_defaults(task='NONE')

    parser.add_argument('--image_root', dest='root', help="image_directory")
    
    parser.add_argument('--ext', choices=['png', 'jpg'], default='png', help="image format")

    return parser.parse_args()

def main():
    args = parse_args()
    root_dir = args.pretrained_models
    omnidata_path = args.omnidata_path
    sys.path.append(args.omnidata_path)
    map_location = 'cpu'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(sys.path)
    from modules.unet import UNet
    from modules.midas.dpt_depth import DPTDepthModel
    from data.transforms import get_transform

    pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt' 
    model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint 

    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()

    filenames = [os.path.join(args.root,f) for f in os.listdir(args.root) if f.endswith(args.ext)]
    
    depth_list = []
    from tqdm import tqdm
    for f in tqdm(filenames):
        depth = get_depth(model, f)
        depth_list.append(depth)

    #TODO: save depth_list to a file
    import pdb; pdb.set_trace()

def get_depth(model, img_path):
    image_resize = 384
    resize_fn = transforms.Resize(image_resize,interpolation=PIL.Image.BILINEAR)
    crop_fn = transforms.CenterCrop(image_resize)
    to_fn = transforms.ToTensor()
    normalize_fn = transforms.Normalize(0.5, 0.5)

    img = Image.open(img_path)
    img_rescaled = np.array(resize_fn(img))
    
    stride = 128

    width = img_rescaled.shape[1]
    height = img_rescaled.shape[0]

    x = width//stride
    y = height//stride


    depth_rows = []
    for j in range(y-2):
        depths = []
        for i in range(x-2):
            image_cur = img_rescaled[j*stride:j*stride+image_resize, i*stride:i*stride+image_resize, :]

            depth = model(normalize_fn(to_fn(image_cur))[None].cuda()).detach().cpu()
            depth = standardize_depth_map(depth)
            depths.append(depth)
        depth_left = depths[0]
        s1 = 128
        s2 = 0
        e2 = 128 *2
        for depth_right in depths[1:]:
            depth_right = depths[1]
            depth_left = align_x(depth_left, depth_right, s1, depth_left.shape[2], s2, e2)
        depth_rows.append(depth_left)

    depth_top = depth_rows[0]
        # align depth maps from top to down
    s1 = 128
    s2 = 0
    e2 = 128 *2
    for depth_bottom in depth_rows[1:]:
        depth_top = align_y(depth_top, depth_bottom, s1, depth_top.shape[1], s2, e2)
        s1 += 128

    image_center = img_rescaled[height//2-image_resize//2:height//2+image_resize//2,width//2-image_resize//2:width//2+image_resize//2]
    depth_center = model(normalize_fn(to_fn(image_center))[None].cuda()).detach().cpu()

    scale, shift = compute_scale_and_shift(depth_top[:, height//2-image_resize//2:height//2+image_resize//2,width//2-image_resize//2:width//2+image_resize//2], depth_center, torch.ones_like(depth_center))
    depth_top = scale * depth_top + shift
    depth_top = (depth_top - depth_top.min()) / (depth_top.max() - depth_top.min())


    out = np.array(depth_top[0])
    out = cv2.resize(out,img.size,interpolation=cv2.INTER_LINEAR)
    # cv2 bilinear interpolate

    return out


if __name__ == "__main__":
    main()
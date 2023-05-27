import argparse
import numpy as np 
import os
import os.path as osp 
import cv2
import struct
import matplotlib.pyplot as plt 
# from pyntcloud import PyntCloud
def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

# def load_point_vis(path, masks):
#     with open(path, 'rb') as f:
#         # n = struct.unpack('<Q', f.read(8))[0]
#         n = read_next_bytes(f, 8, "Q")[0]
#         for i in range(n):
#             # m = struct.unpack('<I', f.read(4))[0]
#             _ = f.read(4)
#             if len(_) != 4:
#                 continue
#             # m = read_next_bytes(f, 4, "I")[0]
#             m = struct.unpack('<I', _)[0]
#             # idxuv = read_next_bytes(f, 4*m*3, format_char_sequence="I"*m)
#             for j in range(m):
#                 string = f.read(4*3)
#                 if len(string) != 4*3:
#                     continue
#                 idx, u, v = struct.unpack('<III', string)
#                 masks[idx][v, u] = 1
                
#                 print('idx: {}, u: {}, v: {}'.format(idx, u, v))
def load_point_vis(path, masks):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        print('point number: {}'.format(n))
        for i in range(n):
            m = struct.unpack('<I', f.read(4))[0]
            for j in range(m):
                idx, u, v = struct.unpack('<III', f.read(4 * 3))
                masks[idx][v, u] = 1
                print('idx: {}, u: {}, v: {}'.format(idx, u, v))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_dir', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--dest', type=str, required=True)
    
    args = parser.parse_args()

    colmap_dir = args.colmap_dir
    image_dir = args.image_dir
    
    names = sorted(os.listdir(osp.join(colmap_dir,'dense','stereo','depth_maps')))
    names = [name for name in names if 'geometric' in name]
    image = cv2.imread(osp.join(image_dir, names[0][:-14]))
    shape = image.shape[:2]
    masks = [np.zeros(shape, dtype=np.uint8) for name in names]
    
    # load_point_vis(osp.join(colmap_dir,'dense','fuse.ply.vis'), masks)
    os.makedirs(args.dest, exist_ok=True)
    for name in names:
        depth = read_array(osp.join(colmap_dir,'dense','stereo','depth_maps',name))
        depth = depth.astype(np.float32)
        out = osp.join(args.dest, name.split('.')[0]+'.npy')
        np.save(out, depth)
        print(out)
    import pdb; pdb.set_trace()
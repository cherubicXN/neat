from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import cv2
import os
import os.path as osp
import json

def load_wireframe_json(fname):
    with open(fname,'r') as f:
        data = json.load(f)
    return data

def load_line_segments_from_json(data):
    junctions = np.array(data['vertices'])
    edges = np.array(data['edges'])
    lines = junctions[edges].reshape(-1,4)
    weights = np.array(data['edges-weights'])
    return lines[weights>0.05]


if __name__ == "__main__":
    path = 'data/DTU/scan24/L3dpp-HAWP/Line3D++__W_FULL__N_10__sigmaP_2.5__sigmaA_10__epiOverlap_0.25__kNN_10__OPTIMIZED__vis_3.txt'
    image_info = 'data/DTU/scan24/colmap-txt/images.txt'
    image_dir = 'data/DTU/scan24/image'
    root = 'data/DTU/scan24/analysis-L3dpp-HAWP'
    os.makedirs(root, exist_ok=True)

    with open(image_info,'r') as f:
        image_info = f.readlines()
        image_info = [l for l in image_info if l[0] != '#']
    
    image_files = [l.split() for l in image_info[0::2]]
    ids = [int(l[0]) for l in image_files]
    fnames = [l[-1] for l in image_files]
    fnames = [fnames[a] for a in np.argsort(ids)]

    hawps = [os.path.join(image_dir, '../hawp',i[:-4]+'.json') for i in fnames]
    hawps = [load_wireframe_json(h) for h in hawps]
    hawp_lines = [load_line_segments_from_json(h) for h in hawps]
    # idimage_files = [(int(l[0]),l[-1]) for l in image_files]

    # images = sorted(os.listdir(image_dir))
    images = [os.path.join(image_dir, i) for i in fnames]
    images = [cv2.imread(image) for image in images]
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
    

    # matched_lines = np.zeros((len(images), len(images), 2,4))
    matched_lines = defaultdict(list)
    count_matches = np.zeros((len(images), len(images)))
    for line in lines:
        num_3d_lines = int(line[0])
        line_segments = []
        for i in range(num_3d_lines):
            xyz = [float(x) for x in line[1+6*i:1+6*(i+1)]]
            line_segments.append(xyz)
        line_segments = np.array(line_segments).reshape(-1,2,3)

        cursor = 1+6*num_3d_lines
        num_2d_lines, substr = int(line[cursor]), line[cursor+1:]
        view_list = []
        line_id_list = []
        line2d_list = []
        for i in range(num_2d_lines):
            view_id = int(substr[0])
            line_id = int(substr[1])
            line2d = [float(x) for x in substr[2:6]]
            substr = substr[6:]
            view_list.append(view_id)
            line_id_list.append(line_id)
            line2d_list.append(line2d)
        line2d_list = np.array(line2d_list)
        for i, view_i in enumerate(view_list):
            for j, view_j in enumerate(view_list):
                if view_i == view_j:
                    continue
                # import pdb; pdb.set_trace()
                matched_lines[(view_i-1, view_j-1)].append(np.stack([line2d_list[i], line2d_list[j]],axis=0))
                # count_matches[(view_i, view_j)] += 1
        # fig, axes = plt.subplots(1,2)
        # axes[0].imshow(images[view_list[0]])
        # axes[0].plot(line2d_list[0][[0,2]], line2d_list[0][[1,3]], 'r-')
        # axes[1].imshow(images[view_list[1]])
        # axes[1].plot(line2d_list[1][[0,2]], line2d_list[1][[1,3]], 'r-')
        # plt.show()
    height, width = images[0].shape[:2]

    sorted_keys = list(matched_lines.keys())
    sorted_index = np.argsort([len(matched_lines[k]) for k in sorted_keys])[::-1]
    sorted_keys = [sorted_keys[i] for i in sorted_index]
    # for p0 in range(len(images)):
        # for p1 in range(len(images)):
    for (p0,p1) in sorted_keys[:10]:
            # if p0 == p1:
            #     continue
            # if len(matched_lines[(p0,p1)]) <2:
            #     continue
            # matches = np.array(matched_lines[(p0,p1)])
        matches = np.stack(matched_lines[(p0,p1)],axis=0)
        matches_0 = matches[:,0]
        matches_1 = matches[:,1]
        detection_0 = hawp_lines[p0]
        detection_1 = hawp_lines[p1]

        # fig, axes = plt.subplots(1,2)
        fig = plt.figure()
        fig.set_size_inches(width/height,1,forward=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(images[p0][...,::-1])
        plt.plot([detection_0[:,0],detection_0[:,2]],[detection_0[:,1],detection_0[:,3]], 'g-',alpha=0.5,linewidth=0.5)
        plt.plot([matches_0[:,0],matches_0[:,2]],[matches_0[:,1],matches_0[:,3]], 'r-',linewidth=0.5)
        plt.scatter(detection_0[:,0],detection_0[:,1], color='b',s=1.2,edgecolors='none',zorder=5)
        plt.scatter(detection_0[:,2],detection_0[:,3], color='b',s=1.2,edgecolors='none',zorder=5)
        plt.savefig(
            os.path.join(root,'{}_{}-0.pdf'.format(p0,p1)),dpi=600)
        
        fig = plt.figure()
        fig.set_size_inches(width/height,1,forward=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(images[p1][...,::-1])
        plt.plot([detection_1[:,0],detection_1[:,2]],[detection_1[:,1],detection_1[:,3]], 'g-',alpha=0.5,linewidth=0.5)
        plt.plot([matches_1[:,0],matches_1[:,2]],[matches_1[:,1],matches_1[:,3]], 'r-',linewidth=0.5)
        plt.scatter(detection_1[:,0],detection_1[:,1], color='b',s=1.2,edgecolors='none',zorder=5)
        plt.scatter(detection_1[:,2],detection_1[:,3], color='b',s=1.2,edgecolors='none',zorder=5)
        plt.savefig(
            os.path.join(root,'{}_{}-1.pdf'.format(p0,p1)),dpi=600)
        plt.close('all')
            # axes[0].imshow(images[p0][...,::-1])
            # axes[0].plot([detection_0[:,0],detection_0[:,2]],[detection_0[:,1],detection_0[:,3]], 'g-',alpha=0.5)
            # axes[0].plot([matches_0[:,0],matches_0[:,2]],[matches_0[:,1],matches_0[:,3]], 'r-')
            # axes[1].imshow(images[p1][...,::-1])
            # axes[1].plot([detection_1[:,0],detection_1[:,2]],[detection_1[:,1],detection_1[:,3]], 'g-',alpha=0.5)
            # axes[1].plot([matches_1[:,0],matches_1[:,2]],[matches_1[:,1],matches_1[:,3]], 'r-')

    # axes[1].plot(line2d_list[1][[0,2]], line2d_list[1][[1,3]], 'r-')
            # plt.show()

    lines3d = [[float(x) for x in l[1:7]] for l in lines]
    lines3d = np.array(lines3d).reshape(-1,2,3)

    # trimesh.load_path(lines3d).show()
    import pdb; pdb.set_trace()
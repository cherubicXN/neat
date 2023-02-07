import sys
sys.path.append('../code')
import trimesh
import open3d as o3d
import torch
from pyhocon import ConfigFactory
import utils.general as utils
import matplotlib.pyplot as plt


data = torch.load('neat-wfr.pth')

scores = data['scores']

queried_points3d = data['queried_points3d']

queried_points3d = queried_points3d[scores<0.01]


# random_ind = torch.randperm(queried_points3d.shape[0])[0]
random_ind = 763 

junctions3d = data['junctions3d']

pairs = data['pair']

query_pcd = trimesh.points.PointCloud(queried_points3d)

line1 = [queried_points3d[random_ind], junctions3d[pairs[random_ind,0]]]
line2 = [queried_points3d[random_ind], junctions3d[pairs[random_ind,1]]]

lcd1 = trimesh.load_path(torch.stack(line1,dim=0)[None])
lcd2 = trimesh.load_path(torch.stack(line2,dim=0)[None])
lcd1.colors = [[0,128,64,255]]
lcd2.colors = [[0,128,64,255]]

query_pcd.colors = [255,0,0,32]
# sphere = trimesh.creation.uv_sphere(radius=0.01)
sphere = trimesh.primitives.Sphere(radius=0.015,center=queried_points3d[random_ind])
pcd_hl = trimesh.points.PointCloud(queried_points3d[random_ind][None])

conf = 'confs/neat-simple/dtu-wfr.conf'
scan_id = 24
conf = ConfigFactory.parse_file(conf)
dataset_conf = conf.get_config('dataset')
dataset_conf['scan_id'] = 24
# eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)

# for data in eval_dataset:
#     rgb = data[2]['rgb'].reshape(1200,1600,3)
#     K = data[1]['intrinsics'][:3,:3]
#     pose = data[1]['pose']
#     proj_mat = pose.inverse()[:3]
#     R = proj_mat[:,:3]
#     T = proj_mat[:,3:]

#     proj_3d = (R@queried_points3d.t() + T)
#     proj_3d = proj_3d/proj_3d[2:]
#     import pdb; pdb.set_trace()
#     plt.imshow(rgb)
#     plt.show()
trimesh.Scene([sphere,query_pcd, lcd1, lcd2, ]).show()
trimesh.Scene([sphere,query_pcd]).show()
import pdb; pdb.set_trace()

# def pick_points(pcd):
#     print("")
#     print(
#         "1) Please pick at least three correspondences using [shift + left click]"
#     )
#     print("   Press [shift + right click] to undo point picking")
#     print("2) After picking points, press 'Q' to close the window")
#     vis = o3d.visualization.VisualizerWithEditing()
#     vis.create_window()
#     vis.add_geometry(pcd)
#     vis.run()  # user picks points
#     vis.destroy_window()
#     print("")
#     return vis.get_picked_points()


# pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(queried_points3d))

# point = pick_points(pcd)
print(random_ind)
# import pdb; pdb.set_trace()
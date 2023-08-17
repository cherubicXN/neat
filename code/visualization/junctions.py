import torch
import trimesh
import os
import os.path as osp
import argparse
import open3d as o3d
import numpy as np
import cv2 

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=np.float32)

rot_psi = lambda psi : np.array([
    [np.cos(psi),-np.sin(psi),0,0],
    [np.sin(psi), np.cos(psi),0,0],
    [0,0,1,0],
    [0,0,0,1],
], dtype=np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
], dtype=np.float32)


def pose_spherical(psi, theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = rot_psi(psi/180.*np.pi) @c2w
    c2w = np.array(
        [
            [0,0,-1,0],
            [1,0,0,0],
            [0,-1,0,0],
         [0,0,0,1]]) @ c2w
    return c2w

def WireframeVisualizer(points3d_list, render_dir = None, rx=0, ry=0,rz=0,t=0,point_size=0.5):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points3d_list[-1]))
    import matplotlib.pyplot as plt
    from collections import deque

    if render_dir is not None:
        os.makedirs(render_dir,exist_ok=True)
    
    WireframeVisualizer.view_cnt = 0
    WireframeVisualizer.render_dir = render_dir
    WireframeVisualizer.camera_path = []
    WireframeVisualizer.image_path = []
    WireframeVisualizer.vis = o3d.visualization.VisualizerWithKeyCallback()
    WireframeVisualizer.rot_psi = rx
    WireframeVisualizer.rot_theta = ry
    WireframeVisualizer.rot_phi = rz
    WireframeVisualizer.t = t
    WireframeVisualizer.done = False


    min_w = 10000
    min_h = 10000
    max_w = 0
    max_h = 0

    def adjust_viewpoint(vis, rx,ry,rz,t):
        ctr = vis.get_view_control()
        glb = WireframeVisualizer

        cam = ctr.convert_to_pinhole_camera_parameters()
        glb.rot_psi += rx
        glb.rot_theta += ry
        glb.rot_phi += rz
        glb.t += t

        print('(rot_x,rot_y,rot_z,t) = ({:.3f},{:.3f},{:.3f},{:.3f})'.format(
            glb.rot_psi,
            glb.rot_theta,
            glb.rot_phi,
            glb.t,
        ))

        extrinsic = pose_spherical(glb.rot_psi, glb.rot_theta, glb.rot_phi, glb.t)
        extrinsic = np.linalg.inv(extrinsic)

        cam.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(cam)
        ctr.set_lookat(np.zeros(3))
        return False
        

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0,0,0])
    vis = WireframeVisualizer.vis
    vis.create_window(height=1024,width=1024, left=0, top=0, visible=True, window_name='Wireframe Visualizer')
    # vis.create_window()
    render_option = vis.get_render_option()
    vis.add_geometry(
        pcd
        )

    # adjust_psi(vis, sign=0)
    adjust_viewpoint(vis,0,0,0,0)
    # vis.add_geometry(mesh_frame)
    vis.register_key_callback(ord('W'), lambda x: adjust_viewpoint(x,5,0,0,0))
    vis.register_key_callback(ord('E'), lambda x: adjust_viewpoint(x,-5,0,0,0))
    vis.register_key_callback(ord('S'), lambda x: adjust_viewpoint(x,0,5,0,0))
    vis.register_key_callback(ord('D'), lambda x: adjust_viewpoint(x,0,-5,0,0))
    vis.register_key_callback(ord('X'), lambda x: adjust_viewpoint(x,0,0,5,0))
    vis.register_key_callback(ord('C'), lambda x: adjust_viewpoint(x,0,0,-5,0))
    vis.register_key_callback(ord('A'), lambda x: adjust_viewpoint(x,0,0,0,0.1))
    vis.register_key_callback(ord('Z'), lambda x: adjust_viewpoint(x,0,0,0,-0.1))
    # vis.register_key_callback(ord('H'), print_pose)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    ctr = vis.get_view_control()
    cam = ctr.convert_to_pinhole_camera_parameters()
    K = cam.intrinsic.intrinsic_matrix
    RT = cam.extrinsic

    vis.destroy_window()



    width = 512
    height = 512

    # fig = plt.figure()
    import os.path as osp
    from tqdm import tqdm 
    os.makedirs(render_dir,exist_ok=True)
    
    colors = np.random.rand(points3d_list.shape[1],3)
    for i, points in enumerate(tqdm(points3d_list)):
        
        points2d =  K@(RT[:3,:3]@points.T + RT[:3,3:])
        points2d = points2d/points2d[2:,:]

        # lines2d[:,0,:] -= [xmin,ymin]
        # lines2d[:,1,:] -= [xmin,ymin]

        fig = plt.figure()
        fig.set_size_inches(width/height,1,forward=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.xlim([-0.5, width-0.5])
        plt.ylim([height-0.5, -0.5])
        # plt.plot([lines2d[:,0,0],lines2d[:,1,0]],[lines2d[:,0,1],lines2d[:,1,1]],'-',color='black',linewidth=0.03)

        plt.scatter(points2d[0],points2d[1],color=colors,s=point_size,edgecolors='none',zorder=5)
            # plt.plot(lines2d[:,0,0],lines2d[:,0,1],'o',color='black',markersize=0.1)
            # plt.plot(lines2d[:,1,0],lines2d[:,1,1],'o',color='black',markersize=0.1)
        # path = osp.join(render_dir,'{:04d}.pdf'.format(i))
        # plt.savefig(path,dpi=width)
        path = osp.join(render_dir,'{:04d}.png'.format(i))
        plt.savefig(path,dpi=width)
        plt.close(fig)
        # plt.show()
    rendered_images = []
    for i in range(len(points3d_list)):
        path = osp.join(render_dir,'{:04d}.png'.format(i))
        image = cv2.imread(path)
        rendered_images.append(image)

    output_size = rendered_images[0].shape[:2]
    output_size = (output_size[1],output_size[0])
    out = cv2.VideoWriter(render_dir+'.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 125 , output_size)
    for im in rendered_images:
        out.write(im)
    out.release()

    print('Done! Saved to {}'.format(render_dir+'.mp4'))
    import imageio
    imageio.mimsave(render_dir+'.gif', [cv2.cvtColor(img,cv2.COLOR_BGR2RGB) for img in rendered_images], duration=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # python junctions.py --root ../exps/abc_000075213_neat/2023_02_15_21_42_00
    parser.add_argument('--root', type=str, required=True, help='the path of the reconstructed wireframe model')
    parser.add_argument('--pose', default=None, type=str, choices=['dtu','scan'])
    parser.add_argument('--name', default='junctions-show', type=str, help='the name of the output video')
    parser.add_argument('--start', default=0, type=int,)
    parser.add_argument('--point-size', default=0.05, type=float, help='the size of the points')

    opt = parser.parse_args()
    root = opt.root

    if opt.pose == 'dtu':
        rx = -155
        ry = 0
        rz = -25
        t  = 3
    elif opt.pose == 'scan':
        rx = 0
        ry = 170
        rz = -45
        t = 3
    else:
        rx = ry = rz = 0
        t = 3
    junctions_dir = osp.join(root, 'junctions')
    junctions_files = os.listdir(junctions_dir)
    junctions_files = [f for f in junctions_files if f.endswith('.pth')]
    epochs = sorted([int(f[:-4]) for f in junctions_files])

    global_junctions = []
    for epoch in epochs:
        print('epoch: ', epoch)
        junctions_file = osp.join(junctions_dir, '{}.pth'.format(epoch))
        x = torch.load(junctions_file, map_location='cpu')
        global_junctions.append(x)
    global_junctions = torch.stack(global_junctions, dim=0).cpu().numpy()[opt.start:]

    rend_path = os.path.join(root,opt.name)
    WireframeVisualizer(global_junctions,render_dir=rend_path, rx=rx,ry=ry,rz=rz,t=t,point_size=opt.point_size)
    # import pdb; pdb.set_trace()
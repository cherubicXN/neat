import torch
import trimesh
import os
import os.path as osp
import argparse
import taichi as ti

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # python junctions.py --root ../exps/abc_000075213_neat/2023_02_15_21_42_00
    parser.add_argument('--root', type=str, required=True, help='the path of the reconstructed wireframe model')

    opt = parser.parse_args()
    root = opt.root

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
    global_junctions = torch.stack(global_junctions, dim=0)

    ti.init(arch=ti.cuda)

    scene = ti.ui.Scene()

    window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(5, 2, 2)
    camera.lookat(0, 0, 0)
    # camera.up(0, 1, 0)
    camera.projection_mode(ti.ui.ProjectionMode(0))


    points = ti.Vector.field(3, dtype=ti.f32, shape=(global_junctions.shape[1],))
    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in points:
            points[i] = [i for j in ti.static(range(3))]
            

    init_points_pos(points)
    import time
    i = 0
    # video_manager = ti.tools.VideoManager(os.path.join(expdir, timestamp))

    junctions_interpolated = global_junctions
        # trimesh.points.PointCloud(x).show()
        
    i = 0

    rendered_images = []
    while window.running:
        camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.LMB)
        # camera.lookat(0, 0, 0)
        scene.set_camera(camera)
        # import pdb; pdb.set_trace()
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        cur_points = junctions_interpolated[i].numpy()
        points.from_numpy(cur_points)
        scene.particles(points, color = (0.68, 0.26, 0.19), radius = 0.01)
        time.sleep(0.02)
            
        # Draw 3d-lines in the scene

        canvas.scene(scene)
        img = window.get_image_buffer_as_numpy()
        # video_manager.write_frame(img)
        rendered_images.append(img)
        window.show()

        if i < junctions_interpolated.shape[0] - 1:
            i += 1


    trimesh.points.PointCloud(cur_points).show()
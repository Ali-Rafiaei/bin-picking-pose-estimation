import copy
import json
import numpy as np
from PIL import Image
import open3d as o3d
import os
import h5py as h5
from numba import jit, prange
from tqdm import tqdm


# IO function from PVNet
def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    actual_xyz = xyz

    xyz = np.dot(xyz, K.T)

    xy = xyz[:, :2] / xyz[:, 2:]
    return xy, actual_xyz


def rgbd_to_point_cloud(K, depth, rgb=None):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    if rgb is not None:
        rgb = rgb[vs, us]
        rgb_flatten = rgb.reshape(-1, 3)
    else:
        rgb_flatten = rgb

    return pts, vs, us


@jit(nopython=True, parallel=True)
def fast_for_map(yList, xList, xyz, distance_list, Radius3DMap):
    for i in prange(len(xList)):
        Radius3DMap[yList[i], xList[i]] = distance_list[i]
    return Radius3DMap


@jit(nopython=True, parallel=True)
def fast_for(pixel_coor, xy, actual_xyz, distance_list, Radius3DMap):
    z_mean = np.mean(actual_xyz[:, 2])
    for coor in pixel_coor:
        iter_count = 0
        z_loc = 0
        z_min = 99999999999999999
        for xy_single in xy:
            if (coor[0] == xy_single[1] and coor[1] == xy_single[0]):
                if (actual_xyz[iter_count, 2] < z_min):
                    z_loc = iter_count
                    z_min = actual_xyz[iter_count, 2]
            iter_count += 1

        if (z_min <= z_mean):
            Radius3DMap[xy[z_loc][1], xy[z_loc][0]] = distance_list[z_loc]
            pre_z_loc = z_loc
        else:
            Radius3DMap[xy[z_loc][1], xy[z_loc][0]] = distance_list[pre_z_loc]

    return Radius3DMap

def quantize_radii_maps(radii_maps, num_bits=16):

    scale = (2**num_bits - 1)

    quantized_maps = (radii_maps * scale).astype(np.uint16)

    return quantized_maps, scale


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default=None)

    parser.add_argument('--visualize',
                        type=bool,
                        default=False)

    parser.add_argument('--use_KGNet',
                        type=bool,
                        default=True)

    args = parser.parse_args()

    no_kpts_per_obj = 4

    key_g_net_point = np.load(os.path.join(args.dataset_dir, 'KeyGNet_kpts.npy'))[:no_kpts_per_obj]
    key_g_net_point = key_g_net_point[:4]

    cycles_path = os.path.join(args.dataset_dir, 'train_pbr')
    cycles_list = sorted(os.listdir(cycles_path), key=lambda x: int(x))
    for cycle in tqdm(cycles_list):


        os.makedirs(os.path.join(args.output_dir, cycle, "gt_uint16"), exist_ok=True)

        radii_maps_h5 = h5.File(os.path.join(args.output_dir, cycle, "gt_uint16/radii_maps_syn.h5"), 'a')


        mask_path = os.path.join(cycles_path, cycle, "mask_visib")
        depth_path = os.path.join(cycles_path, cycle, "depth")
        rgb_path = os.path.join(cycles_path, cycle, "rgb")

        with open(os.path.join(cycles_path, cycle, "scene_camera.json")) as f:
            cam_params = json.load(f)

        sorted_data = sorted(os.listdir(rgb_path), key=lambda x: int(x.split('.')[0]))

        for scene in tqdm(sorted_data):
            scene_id = int(scene.split('.')[0])

            rgb = np.asarray(Image.open(os.path.join(rgb_path, scene)))
            img_h = rgb.shape[0]
            img_w = rgb.shape[1]

            scene_cam_params = cam_params[str(scene_id)]
            depth_scale = scene_cam_params["depth_scale"]
            cam_k = np.asarray(scene_cam_params["cam_K"]).reshape((3, 3))

            depth = np.asarray(Image.open(os.path.join(depth_path, str(scene_id).zfill(6) + ".png")))
            depth = depth * depth_scale

            scene_xyz, _, _ = rgbd_to_point_cloud(cam_k, depth)

            with open(os.path.join(cycles_path, cycle, "scene_gt.json")) as f:
                pose_annots = json.load(f)
            with open(os.path.join(cycles_path, cycle, "scene_gt_info.json")) as f:
                metadata = json.load(f)


            scene_annots = pose_annots[str(scene_id)][0]
            scene_metadata = metadata[str(scene_id)][0]

            if scene_metadata['visib_fract'] < 0.1:
                continue
            cam_R = np.asarray(scene_annots["cam_R_m2c"]).reshape((3, 3))
            cam_t = np.asarray(scene_annots["cam_t_m2c"]).reshape((3, 1))
            gt_pose = np.hstack([cam_R, cam_t])
            obj_id = scene_annots["obj_id"]


            mesh = o3d.io.read_triangle_mesh(os.path.join(args.dataset_dir, "BW_interested_cad.ply"))
            mesh_pointcloud = mesh.sample_points_uniformly(number_of_points=5000)
            mesh_points = np.asarray(mesh_pointcloud.points)
            transferred_mesh_points = np.dot(mesh_points, cam_R.T) + cam_t.T

            transferred_kpts = np.dot(key_g_net_point, cam_R.T) + cam_t.T

            cam_k_kpts = np.dot(transferred_kpts, cam_k.T)
            projected_kpts = cam_k_kpts[:, :2] / cam_k_kpts[:, 2:]
            projected_kpts[:, 0] = projected_kpts[:, 0] / img_w
            projected_kpts[:, 1] = projected_kpts[:, 1] / img_h

            mask_visib = np.asarray(Image.open(os.path.join(mask_path, f"{str(scene_id).zfill(6)}_000000.png")))
            mask_visib = np.where(mask_visib > 0, 1, 0)
            vs, us = mask_visib.nonzero()
            masked_depth = np.multiply(depth, mask_visib)

            radii_maps = np.zeros((transferred_mesh_points.shape[0], no_kpts_per_obj))

            for j in range(no_kpts_per_obj):
                distance_list = (((transferred_mesh_points[:, 0] - transferred_kpts[j, 0]) ** 2 +
                                  (transferred_mesh_points[:, 1] - transferred_kpts[j, 1]) ** 2 +
                                  (transferred_mesh_points[:, 2] - transferred_kpts[j, 2]) ** 2) ** 0.5)
                radii_maps[:, j] = distance_list


            if args.visualize:
                scene_pc = o3d.geometry.PointCloud()
                scene_pc.points = o3d.utility.Vector3dVector(scene_xyz)

                transferred_mesh = o3d.geometry.PointCloud()
                transferred_mesh.points = o3d.utility.Vector3dVector(transferred_mesh_points)
                transferred_mesh.paint_uniform_color(np.array([160, 0, 160]) / 255.0)

                all_kepoints_pc = o3d.geometry.PointCloud()
                all_kepoints_pc.points = o3d.utility.Vector3dVector(np.array(transferred_kpts))
                all_kepoints_pc.paint_uniform_color(np.array([0, 0, 1]))

                o3d.visualization.draw_geometries([scene_pc, all_kepoints_pc, transferred_mesh])
                continue

            max_value = np.max(radii_maps)
            normalized_radii_maps = radii_maps / max_value
            quantized_radial_maps, scale = quantize_radii_maps(normalized_radii_maps)

            radii_maps_h5.create_dataset(f"{scene_id}/radial_maps", data=quantized_radial_maps, compression="gzip", compression_opts=9)
            radii_maps_h5.create_dataset(f"{scene_id}/max_value", data=np.array([max_value]), compression="gzip", compression_opts=9)

        radii_maps_h5.close()





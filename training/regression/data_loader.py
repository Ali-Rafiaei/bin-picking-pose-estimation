import copy
import json
import open3d as o3d
import numpy as np
import os
import torch
from PIL import Image

from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils import data



def rgbd_to_point_cloud(K, depth, rgb=None):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts, vs, us

def obj_id_to_index(obj_id):
    """
    This function converts the obj_id to the index of the KeyGNet keypoints
    """
    id_dict = {
        '0': 0,
        '1': 1,
        '4': 2,
        '8': 3,
        '10': 4,
        '11': 5,
        '14': 6,
        '18': 7,
        '19': 8,
        '20': 9,
    }
    return id_dict[obj_id]

class RadialMapLoader(Dataset):
    def __init__(self, args, set="train"):

        self.train_pbr_path = args.train_pbr_path
        self.split_path = args.split_path
        self.include_depth = args.input_include_depth
        self.objs_to_use = args.objs_to_use
        self.kpoint_type = args.kpoint_type
        with open(os.path.join(self.split_path, "list_of_usable_masks.txt"), "r") as f:
            ids_list = f.readlines()

        ids_list = [x.strip() for x in ids_list]
        ids_list = ids_list[1:]  # first row is a header

        if isinstance(self.objs_to_use, list):
            ids_list = [x for x in ids_list if x.split("/")[-1] in self.objs_to_use]

        if isinstance(self.objs_to_use, int):
            ids_list = [x for x in ids_list if x.split("/")[-1] == str(self.objs_to_use)]

        train_list = ids_list[:int(len(ids_list)*0.9)]
        val_list = ids_list[int(len(ids_list)*0.9):]

        if set == "train":
            self.list_to_load = train_list
        else:
            self.list_to_load = val_list

        if self.kpoint_type == "bbox_corners":
            self.keypoints = np.load(os.path.join(self.split_path, "bounding_box_kpoints.npy"), allow_pickle=True)
            self.kpoint_indices_to_use = np.array([1, 2, 3, 4])
            self.objs_to_process = np.array([0, 8, 18, 19, 20])
            self.keypoints = self.keypoints[:, self.kpoint_indices_to_use]

        elif self.kpoint_type == "keygnet":
            self.keypoints = np.load("KeyGNet_kpts.npy")

    def __len__(self):
        return len(self.list_to_load)

    def __getitem__(self, idx):
        cycle_id, scene_id, cam_id, object_index, obj_id = self.list_to_load[idx].split('/')
        cycle_path = os.path.join(self.train_pbr_path, cycle_id)

        rgb_path = os.path.join(cycle_path, f"rgb_cam{cam_id}")
        mask_path = os.path.join(cycle_path, f"mask_visib_cam{cam_id}")
        depth_path = os.path.join(cycle_path, f"depth_cam{cam_id}")

        rgb = np.asarray(Image.open(os.path.join(rgb_path, scene_id + ".jpg")))
        img_height, img_width = rgb.shape[0], rgb.shape[1]

        obj_mask = np.asarray(Image.open(os.path.join(mask_path, f"{scene_id}_{object_index.zfill(6)}.png")))
        binary_mask = np.where(obj_mask > 0, 1, 0).astype(np.uint8)
        depth_image = np.asarray(Image.open(os.path.join(depth_path, f"{scene_id}.png")))

        with open(os.path.join(cycle_path, f"scene_camera_cam{cam_id}.json"), "r") as f:
            all_scene_cam = json.load(f)
            scene_cam = all_scene_cam[str(int(scene_id))]

        with open(os.path.join(cycle_path, f"scene_gt_cam{cam_id}.json")) as f:
            pose_annots = json.load(f)
        with open(os.path.join(cycle_path, f"scene_gt_info_cam{cam_id}.json")) as f:
            metadata = json.load(f)

        cam_k = np.array(scene_cam["cam_K"]).reshape((3, 3))
        depth_scale = scene_cam["depth_scale"]

        obj_pose = pose_annots[str(int(scene_id))][int(object_index)]
        obj_metadata = metadata[str(int(scene_id))][int(object_index)]
        cam_R = np.asarray(obj_pose["cam_R_m2c"]).reshape((3, 3))
        cam_t = np.asarray(obj_pose["cam_t_m2c"]).reshape((3, 1))
        gt_pose = np.hstack([cam_R, cam_t])

        if self.kpoint_type == "bbox_corners":
            model_kpoints = self.keypoints[obj_id_to_index(obj_id)]
        elif self.kpoint_type == "keygnet":
            model_kpoints = self.keypoints
        transferred_kpts = np.dot(model_kpoints, cam_R.T) + cam_t.T

        segmented_rgb = rgb.copy()
        segmented_rgb[binary_mask == 0] = 0

        depth_image = depth_image * depth_scale # Convert to mm
        segmented_depth = depth_image.copy()
        segmented_depth[binary_mask == 0] = 0

        object_xyz, vs, us = rgbd_to_point_cloud(cam_k, segmented_depth)
        radii_maps = np.zeros((4, img_height, img_width))

        for j in range(4):
            distance_list = (((object_xyz[:, 0] - transferred_kpts[j, 0]) ** 2 +
                              (object_xyz[:, 1] - transferred_kpts[j, 1]) ** 2 +
                              (object_xyz[:, 2] - transferred_kpts[j, 2]) ** 2) ** 0.5)

            radii_maps[j, vs, us] = distance_list

        segmented_rgb = segmented_rgb / 255.0
        segmented_object_tensor = torch.tensor(segmented_rgb.transpose(2, 0, 1))
        segmented_object_tensor = segmented_object_tensor.to(dtype=torch.float16)

        if self.include_depth:
            segmented_depth = segmented_depth / 20900  # max depth in dataset (mm)
            segmented_object_tensor = torch.cat((segmented_object_tensor, torch.tensor(segmented_depth).unsqueeze(0)), dim=0)


        return segmented_object_tensor, radii_maps


    def vis_image_maps(self, image, maps):
        segmented_object_tensor = image
        radii_maps = maps

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(segmented_object_tensor.permute(1, 2, 0).cpu().numpy())
        plt.subplot(2, 3, 2)
        plt.imshow(radii_maps[0])
        plt.subplot(2, 3, 3)
        plt.imshow(radii_maps[1])
        plt.subplot(2, 3, 4)
        plt.imshow(radii_maps[2])
        plt.subplot(2, 3, 5)
        plt.imshow(radii_maps[3])
        plt.show()

    def vis_pc_w_keypoints(self, object_xyz, transferred_kpts):
        object_pc = o3d.geometry.PointCloud()
        object_pc.points = o3d.utility.Vector3dVector(object_xyz)
        object_pc.paint_uniform_color(np.array([0, 0, 1]))

        keypoints_pc = o3d.geometry.PointCloud()
        keypoints_pc.points = o3d.utility.Vector3dVector(transferred_kpts)
        keypoints_pc.paint_uniform_color(np.array([1, 0, 0]))

        o3d.visualization.draw_geometries([object_pc, keypoints_pc])

def custom_collate(batch):
    return list(zip(*batch))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pbr_path", type=str, default=None)

    parser.add_argument("--split_path", type=str,
                        default="dataset_splits")

    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--input_include_depth", type=bool, default=False)
    parser.add_argument("--objs_to_use", default=19)
    parser.add_argument("--kpoint_type", type=str, default="keygnet")

    args = parser.parse_args()

    test_dataset = RadialMapLoader(args, set="train")
    test_loader = data.DataLoader(test_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    for i, batch in enumerate(test_loader):
        print(np.array(batch[1]).shape)
        break

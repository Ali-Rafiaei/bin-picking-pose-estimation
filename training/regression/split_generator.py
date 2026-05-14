"""
The purpose of this script is to go through all the scenes in the dataset and find which masks have a decent visibility to be
used in the regressor training.
"""
import os
import numpy as np
import json

from tqdm import tqdm

train_pbr_path = "/media/ali/SecondSSD/MyResearch/Datasets/BOP/bpc_phase2/bpc_phase2/train_pbr"
split_dir = "/media/ali/SecondSSD/MyResearch/Datasets/BOP/bpc_phase2/dataset_splits"

if not os.path.exists(split_dir):
    os.makedirs(split_dir)

cycle_list = sorted(os.listdir(os.path.join(train_pbr_path)), key=lambda x: int(x))
num_cycles = len(cycle_list)
num_scenes_from_each_camera = 100
num_cameras = 3

# # For the first half of the dataset (cycles 0-24), uncomment the following line:
# cycle_list = np.arange(num_cycles)
# # For the second half of the dataset (cycles 25-49), uncomment the following line:
cycle_list = np.arange(0, 50)
camera_list = np.arange(num_cameras) + 1
scene_list = np.arange(num_scenes_from_each_camera)
cycle_camera_scene_combinations = np.array(np.meshgrid(cycle_list, scene_list, camera_list)).T.reshape(-1, 3)

for cycle_scene_camera in tqdm(cycle_camera_scene_combinations):
    cycle_id = cycle_scene_camera[0]
    scene_id = cycle_scene_camera[1]
    camera_id = cycle_scene_camera[2]

    cycle_id = str(cycle_id).zfill(6)
    scene_id = str(scene_id).zfill(6)

    cycle_path = os.path.join(train_pbr_path, cycle_id)
    
    with open(os.path.join(cycle_path, f"scene_gt_info_cam{camera_id}.json"), "r") as f:
        gt_info = json.load(f)
        scene_gt_info = gt_info[str(int(scene_id))]

    with open(os.path.join(cycle_path, f"scene_gt_cam{camera_id}.json"), "r") as f:
        gt = json.load(f)
        scene_gt = gt[str(int(scene_id))]


    for object_index, object_info in enumerate(tqdm(scene_gt_info)):
        if object_info["visib_fract"] < 0.4:
            continue
        else:
            obj_id = scene_gt[object_index]["obj_id"]
            with open(os.path.join(split_dir, "list_of_usable_masks_second_half.txt"), "a") as f:
                f.write(f"{cycle_id}/{scene_id}/{camera_id}/{object_index}/{obj_id}\n")

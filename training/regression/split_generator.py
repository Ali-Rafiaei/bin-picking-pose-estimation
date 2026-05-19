"""
Go through all scenes in the dataset and find which masks have sufficient visibility
for regressor training.
"""
import argparse
import os
import numpy as np
import json

from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pbr_path", type=str, default=None)
    parser.add_argument("--split_dir", type=str, default=None)
    args = parser.parse_args()

    train_pbr_path = args.train_pbr_path
    split_dir = args.split_dir

    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    num_scenes_from_each_camera = 100
    num_cameras = 3

    cycle_list = np.arange(0, 50)
    camera_list = np.arange(num_cameras) + 1
    scene_list = np.arange(num_scenes_from_each_camera)
    cycle_camera_scene_combinations = np.array(np.meshgrid(cycle_list, scene_list, camera_list)).T.reshape(-1, 3)

    for cycle_scene_camera in tqdm(cycle_camera_scene_combinations):
        cycle_id = str(cycle_scene_camera[0]).zfill(6)
        scene_id = str(cycle_scene_camera[1]).zfill(6)
        camera_id = cycle_scene_camera[2]

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
                with open(os.path.join(split_dir, "list_of_usable_masks.txt"), "a") as f:
                    f.write(f"{cycle_id}/{scene_id}/{camera_id}/{object_index}/{obj_id}\n")

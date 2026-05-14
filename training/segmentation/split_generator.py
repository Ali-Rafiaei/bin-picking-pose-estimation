import os
from os.path import split

import numpy as np

dataset_dir = "/media/ali/SecondSSD/MyResearch/data_generation/new_data_generation_pipeline/output/bin_picking_final/train_pbr"
split_dir = "/media/ali/SecondSSD/MyResearch/data_generation/new_data_generation_pipeline/output/bin_picking_final/Split"
if not os.path.exists(split_dir):
    os.makedirs(split_dir)

cycle_list = sorted(os.listdir(dataset_dir), key=lambda x: int(x))
sampled_cylces = np.random.choice(cycle_list, len(cycle_list), replace=False)

for cycle in sampled_cylces:
    scene_list = sorted(os.listdir(os.path.join(dataset_dir, cycle, "rgb")), key=lambda x: int(x.split(".")[0]))
    sampled_training_scenes = np.random.choice(scene_list, int(0.9*len(scene_list)), replace=False)
    sampled_validation_scenes = [scene for scene in scene_list if scene not in sampled_training_scenes]

    for scene in sampled_training_scenes:
        with open(os.path.join(split_dir, "train.txt"), "a") as f:
            f.write(f"{cycle}/{scene.split('.')[0]}\n")

    for scene in sampled_validation_scenes:
        with open(os.path.join(split_dir, "val.txt"), "a") as f:
            f.write(f"{cycle}/{scene.split('.')[0]}\n")
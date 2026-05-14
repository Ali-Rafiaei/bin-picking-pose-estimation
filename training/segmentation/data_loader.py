import copy
import json
from time import time

import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class IpdDataset(Dataset):
    def __init__(self, args, augment=None, mode="train"):

        self.augment = augment
        self.train_pbr_path = args.train_pbr_path
        self.num_scenes_from_each_camera = 100
        self.num_cameras = 3

        # # If training on the first half of the dataset, uncomment the following line:
        # cycle_list = np.arange(self.num_cycles)
        # # If training on the second half of the dataset, uncomment the following line:
        cycle_list = np.arange(25, 50)
        camera_list = np.arange(self.num_cameras) + 1
        scene_list = np.arange(self.num_scenes_from_each_camera)
        self.cycle_camera_scene_combinations = np.array(np.meshgrid(cycle_list, scene_list, camera_list)).T.reshape(-1,
                                                                                                                    3)
        # self.train_list = np.random.choice(self.cycle_camera_scene_combinations, len(self.cycle_camera_scene_combinations)*0.9, replace=False)
        # self.val_list = np.array([x for x in self.cycle_camera_scene_combinations if x not in self.train_list])
        # choosing the 90% of the data randomly for training and 10% for validation
        train_indices = np.random.choice(np.arange(len(self.cycle_camera_scene_combinations)),
                                         size=int(len(self.cycle_camera_scene_combinations) * 0.9), replace=False)
        self.train_list = self.cycle_camera_scene_combinations[train_indices]
        val_indices = np.array(
            [x for x in np.arange(len(self.cycle_camera_scene_combinations)) if x not in train_indices])
        self.val_list = self.cycle_camera_scene_combinations[val_indices]

        if mode == "train":
            self.list_to_process = self.train_list
        else:
            self.list_to_process = self.val_list

    def __getitem__(self, idx):
        """

        During training, the model expects both the input tensors and targets (list of dictionary),
        containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.

        - labels (Int64Tensor[N]): the class label for each ground-truth box

        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

        """
        cycle_id = self.list_to_process[idx][0]
        scene_id = self.list_to_process[idx][1]
        camera_id = self.list_to_process[idx][2]

        cycle_id = str(cycle_id).zfill(6)
        scene_id = str(scene_id).zfill(6)

        cycle_path = os.path.join(self.train_pbr_path, cycle_id)
        with open(os.path.join(cycle_path, f"scene_gt_info_cam{camera_id}.json"), "r") as f:
            scene_gt_info = json.load(f)

        with open(os.path.join(cycle_path, f"scene_gt_cam{camera_id}.json"), "r") as f:
            scene_gt = json.load(f)

        image = np.array(Image.open(os.path.join(cycle_path, f"rgb_cam{camera_id}/{scene_id}.jpg")))
        # Normalize the image:
        image = image / 255.0
        # Create the image tensor
        image = torch.tensor(image.transpose(2, 0, 1))

        masks = []
        boxes = []
        labels = []
        masks_path = os.path.join(cycle_path, f"mask_visib_cam{camera_id}")
        for mask in os.listdir(masks_path):
            if mask.startswith(scene_id):
                scene, mask_index = (mask.split(".")[0]).split("_")
                obj_info = scene_gt_info[str(int(scene_id))][int(mask_index)]
                if obj_info["visib_fract"] < 0.3:
                    continue
                obj_id = scene_gt[str(int(scene_id))][int(mask_index)]["obj_id"]
                mask_image = np.array(Image.open(os.path.join(masks_path, mask)))
                binary_mask = np.where(mask_image > 0, 1, 0).astype(np.uint8)

                x1, y1, height, width = obj_info["bbox_visib"]
                x2 = x1 + height
                y2 = y1 + width
                boxes.append([x1, y1, x2, y2])
                masks.append(binary_mask)
                labels.append(self.obj_id_to_label(str(obj_id)))

        masks = np.array(masks)
        # labels = np.ones((len(boxes),), dtype=np.int64)

        target = {
            "boxes": torch.from_numpy(np.array(boxes)).to(dtype=torch.float32),
            "labels": torch.from_numpy(np.array(labels)).to(dtype=torch.int64),
            "masks": torch.from_numpy(masks).to(dtype=torch.uint8),
        }

        if self.augment and np.random.rand() > 0.4:
            # if True:
            image = self.augment_image(image, masks)

        image = image.to(dtype=torch.float32)
        return image, target

    def __len__(self):
        return len(self.list_to_process)

    def augment_image(self, image, binary_masks):
        """
        steps:
            1. Create a copy of the original image
            2. Apply all the augmentations on the image copy
            3. using the mask transfer the object pixels from the augmented copy image to the original image
        """

        augmented_object_image = copy.deepcopy(image)
        tr_object = v2.Compose([v2.RandomPhotometricDistort(p=0.3)])
        augmented_object_image = tr_object(augmented_object_image)

        # Transfer the object pixels from the augmented image to the original image:
        for mask in binary_masks:
            image[:, mask == 1] = augmented_object_image[:, mask == 1]

        tr_image = v2.Compose([v2.RandomGrayscale(0.3),
                               v2.RandomChannelPermutation(), ])

        image = tr_image(image)

        return image

    def obj_id_to_label(self, obj_id):
        if self.training_first_half:
            id_to_label = {
                "0": 1,
                "8": 2,
                "18": 3,
                "19": 4,
                "20": 5
            }
        else:
            id_to_label = {
                "1": 1,
                "4": 2,
                "10": 3,
                "11": 4,
                "14": 5
            }

        return id_to_label[obj_id]


def collate_fn(batch):
    return tuple(zip(*batch))


def generate_loaders(args):
    train_dataset = IpdDataset(args, augment=True, mode="train")
    val_dataset = IpdDataset(args, augment=True, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=0)

    return train_loader, val_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # Parameters to set
    parser.add_argument("--train_pbr_path",
                        type=str,
                        default='/media/ali/SecondSSD/MyResearch/Datasets/BOP/IPD/ipd/train_pbr')
    parser.add_argument("--batch_size",
                        type=int,
                        default=16)

    args = parser.parse_args()

    test_dataset = CustomDataset(args, augment=True, mode="train")
    test_dataset[0]
    # test_dataset[0]
    # exit()
    # test_loader, _ = generate_loaders(args)

    # loop_start_time = time()
    # batch_start_time = time()
    # for i, batch in enumerate(test_loader):
    #     print("Batch Time: ", time() - batch_start_time)
    #     batch_start_time = time()
    #     print(f"Total Time after {i} iterations: {time() - loop_start_time}")
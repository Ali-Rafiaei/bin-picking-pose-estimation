import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import sys
from typing import List, Optional, Union


import os
import open3d as o3d
import torch
import subprocess
from utils.horn import HornPoseFitting
from utils.ransac_experimental import RANSAC_w_refinement_adaptive
from utils.epipolar_matching import compute_cost_matrix, match_objects, multi_view_match
from utils.model_loaders import RegLoader, SegLoader
from utils import ddd_utils
from utils.ddd_utils import refinement_by_rotation, transform_point_cloud, depth_to_point_cloud, perform_icp


def rot_to_quat(rot):
    r = Rotation.from_matrix(rot)
    q = r.as_quat()
    return q

def object_id_to_segmentation_label(obj_id):
    id_to_label = {
        "0": 1,
        "1": 2,
        "4": 3,
        "8": 4,
        "10": 5,
        "11": 6,
        "14": 7,
        "18": 8,
        "19": 9,
        "20": 10
    }
    return id_to_label[str(obj_id)]

class PoseEstimator():

    def __init__(self):
        super().__init__("bpc_pose_estimator")
        self.get_logger().info("Starting bpc_pose_estimator...")

        self.model_dir = (
            self.declare_parameter("model_dir", "").get_parameter_value().string_value
        )
        if self.model_dir == "":
            raise Exception("ROS parameter model_dir not set.")
        self.get_logger().info(f"Model directory set to {self.model_dir}.")
        srv_name = "/get_pose_estimates"
        self.get_logger().info(f"Pose estimates can be queried over srv {srv_name}.")
        self.srv = self.create_service(GetPoseEstimates, srv_name, self.srv_cb)


        # Declare parameters
        self.part_regression = True
        self.regression_part_size = 6
        self.part_segmentation = False
        self.cache_models = True
        self.perform_refinement = True
        self.call_counter = 0
        self.estimator_construction_time = time.time()

        self.first_half_obj_ids = [0, 8, 18, 19, 20]
        self.second_half_obj_ids = [1, 4, 10, 11, 14]
        self.keypoints = np.array([[-18.9783262, 16.60012318, -26.71720371],
                                   [63.48793434, 5.03256196, -20.61116947],
                                   [-60.04081006, 21.36026804, 9.61595647],
                                   [-24.7195646, -13.01965225, -22.26686055]])

        self.segmentation_models_cache = {}
        self.regression_models_cache = {}
        self.objects_model_points = {}
        for id in self.first_half_obj_ids + self.second_half_obj_ids:
            obj_mesh = o3d.io.read_triangle_mesh(f"/opt/ros/underlay/install/3d_models/obj_{str(id).zfill(6)}.ply")
            obj_mesh_as_pc = obj_mesh.sample_points_uniformly(500)
            obj_points = np.asarray(obj_mesh_as_pc.points)
            self.objects_model_points.update({str(id): obj_points})

        if self.cache_models:
            for id in self.first_half_obj_ids + self.second_half_obj_ids:
                regression_checkpoint_path = os.path.join(self.model_dir, "checkpoints", "regression",
                                                          f"obj_{id}.ckpt")
                self.regression_models_cache.update({str(id): RegLoader.load_from_checkpoint(regression_checkpoint_path)})


            segmentation_checkpoint_path = os.path.join(self.model_dir, "checkpoints", "segmentation", "all_objs.ckpt")
            self.segmentation_models_cache.update({"all_objs": SegLoader.load_from_checkpoint(segmentation_checkpoint_path)})


    def get_pose_estimates(
        self,
        object_ids: List[int],
        cam_1: Camera,
        cam_2: Camera,
        cam_3: Camera,
        photoneo: Camera,
    ) -> List[PoseEstimateMsg]:

        pose_estimates = []
        self.get_logger().info(f"Received request to estimates poses for object_ids: {object_ids}")

        rgb_cam1 = np.tile(cam_1.rgb[:, :, None], (1, 1, 3))
        # pallet = np.zeros_like(rgb_cam1)
        # pallet[:, 750:3200, :] = 1
        # rgb_cam1 = rgb_cam1 * pallet

        rgb_cam2 = np.tile(cam_2.rgb[:, :, None], (1, 1, 3))
        # pallet = np.zeros_like(rgb_cam2)
        # pallet[:1750, 1000:3380, :] = 1
        # rgb_cam2 = rgb_cam2 * pallet

        rgb_cam3 = np.tile(cam_3.rgb[:, :, None], (1, 1, 3))
        # pallet = np.zeros_like(rgb_cam3)
        # pallet[:, 620:3380, :] = 1
        # rgb_cam3 = rgb_cam3 * pallet

        depth_cam1 = cam_1.depth * 0.1
        depth_cam2 = cam_2.depth * 0.1
        depth_cam3 = cam_3.depth * 0.1

        Rt_cam1 = cam_1.pose
        Rt_cam2 = cam_2.pose
        Rt_cam3 = cam_3.pose

        K_cam1 = cam_1.intrinsics
        K_cam2 = cam_2.intrinsics
        K_cam3 = cam_3.intrinsics
        K_ref = photoneo.intrinsics

        RT1_to_ref = np.linalg.inv(Rt_cam1)
        RT2_to_ref = np.linalg.inv(Rt_cam2)
        RT3_to_ref = np.linalg.inv(Rt_cam3)

        # TODO: Moving the segementation to before the object id iteration loop so that it is run only once for each scene

        for object_id in object_ids:
            global_time = time.time()
            if self.cache_models:
                regression_model = self.regression_models_cache[str(object_id)]
                segmentation_model = self.segmentation_models_cache["all_objs"]

            else:
                segmentation_model, regression_model = self.load_models(object_id)

            # TODO: Figure out if it is possible to black out the irrelevant parts of the image. It is possible to do if we are allowed to change the pose estimation code for phase2
            #       Here's the template code for blacking out the rgb images:

            # TODO: Cache the models in order to reduce loading time



            st_time = time.time()
            matched_masks = self.detect_and_match([rgb_cam1, rgb_cam2, rgb_cam3], [K_cam1, K_cam2, K_cam3],
                                                  [Rt_cam1, Rt_cam2, Rt_cam3], segmentation_model, object_id)

            self.get_logger().info("Time to run the segmentation model: {:.2f} seconds".format(time.time() - st_time))

            if (not matched_masks or any(len(matched_masks.get(cam, [])) == 0 for cam in ("cam1", "cam2", "cam3"))):
                self.get_logger().warn(f"No valid masks for object {object_id}: ")
                continue

            st_time = time.time()
            all_object_xyzs, all_object_radii_maps = self.estimate_radii_maps([rgb_cam1, rgb_cam2, rgb_cam3],
                                             [depth_cam1, depth_cam2, depth_cam3], [K_cam1, K_cam2, K_cam3],
                                                [RT1_to_ref, RT2_to_ref, RT3_to_ref], matched_masks, regression_model)

            self.get_logger().info("Time to run the regression model: {:.2f} seconds".format(time.time() - st_time))

            all_estimated_kpts = np.zeros((len(all_object_xyzs), 4, 3))
            ransac_time = 0
            refinement_time = 0
            for i in range(len(all_object_xyzs)):
                object_estimated_kpts = np.zeros((4, 3))
                object_xyz = all_object_xyzs[i]
                object_radii = all_object_radii_maps[i]
                # Only keeping the non-zero radii maps
                u, v = np.where(object_radii[:, :, 0] != 0)
                object_radii = object_radii[u, v, :]

                st_time = time.time()
                for keypoint_index in range(4):
                    center_mm_s = RANSAC_w_refinement_adaptive(object_xyz, object_radii[:, keypoint_index], iterations=1000, initial_epsilon=0.5, MAX_REFINEMENTS=1)
                    all_estimated_kpts[i, keypoint_index] = center_mm_s[0]
                    object_estimated_kpts[keypoint_index] = center_mm_s[0]
                ransac_time += time.time() - st_time

                estimated_pose = np.eye(4)
                horn_solver = HornPoseFitting()

                horn_solver.lmshorn(self.keypoints, object_estimated_kpts, 4, estimated_pose)

                mesh_points = (self.objects_model_points[str(object_id)]).copy()

                st_time = time.time()
                if self.perform_refinement:
                    transfered_mesh_with_icp, refined_transformation = refinement_by_rotation(
                        all_object_xyzs[i], estimated_pose, mesh_points)
                # transfered_mesh_with_icp, refined_transformation = refinement_by_rotation(
                #     all_object_xyzs[i], refined_transformation, mesh_points)
                else:
                    object_pc = o3d.geometry.PointCloud()
                    object_pc.points = o3d.utility.Vector3dVector(all_object_xyzs[i])
                    transfered_mesh_with_icp, refined_transformation = perform_icp(object_pc, mesh_points, estimated_pose)

                refinement_time += time.time() - st_time

                final_transformation = refined_transformation.copy()
                pose_estimate = PoseEstimateMsg()
                pose_estimate.obj_id = object_id
                # TODO: calculate the score based on the two point clouds euclidean distance
                pose_estimate.score = 1.0
                pose_estimate.pose.position.x = final_transformation[0, 3]
                pose_estimate.pose.position.y = final_transformation[1, 3]
                pose_estimate.pose.position.z = final_transformation[2, 3]
                rot = rot_to_quat(final_transformation[0:3, 0:3])
                pose_estimate.pose.orientation.x = rot[0]
                pose_estimate.pose.orientation.y = rot[1]
                pose_estimate.pose.orientation.z = rot[2]
                pose_estimate.pose.orientation.w = rot[3]
                pose_estimates.append(pose_estimate)

            # torch.cuda.empty_cache()
            # torch.cuda.ipc_collect()
            # gc.collect()

            # self.get_logger().info(f"Detected {len(all_object_xyzs)} object with id {object_id} and estimated their pose")
            # out = subprocess.check_output(['free', '-b']).decode().splitlines()[1].split()
            # total, used, free = map(int, (out[1], out[2], out[3]))
            # self.get_logger().info("Ram Stats:")
            # self.get_logger().info(
            #     f"Total: {total / (1024 ** 3):.2f} GiB, Used: {used / (1024 ** 3):.2f} GiB, Free: {free / (1024 ** 3):.2f} GiB")
            self.get_logger().info(f"RANSAC time for a scene with {len(all_object_xyzs)} objects detected: {ransac_time:.2f} seconds")
            self.get_logger().info(f"Here are the ICP thresholds: {ddd_utils.get_thresholds()}")
            self.get_logger().info(f"Refinement time for a scene with {len(all_object_xyzs)} objects detected: {refinement_time:.2f} seconds")
            self.get_logger().info(f"Total time for a scene with {len(all_object_xyzs)} objects detected: {time.time() - global_time:.2f} seconds")
        # print("\nHere are the estimated poses: \n", pose_estimates)
        self.call_counter += 1
        self.get_logger().info(f"Pose estimates called {self.call_counter} times in the totatl time of {time.time() - self.estimator_construction_time:.2f} seconds")
        return pose_estimates

    def load_models(self, object_id):
        # if self.cache_models:
        #     if object_id in self.first_half_obj_ids:
        #         if "first_half" not in self.segmentation_models_cache:
        #             segmentation_checkpoint_path = os.path.join(self.model_dir, "checkpoints", "segmentation",
        #                                                         "first_half.ckpt")
        #             self.segmentation_models_cache["first_half"] = SegLoader.load_from_checkpoint(
        #                 segmentation_checkpoint_path)
        #     else:
        #         if "second_half" not in self.segmentation_models_cache:
        #             segmentation_checkpoint_path = os.path.join(self.model_dir, "checkpoints", "segmentation",
        #                                                         "second_half.ckpt")
        #             self.segmentation_models_cache["second_half"] = SegLoader.load_from_checkpoint(
        #                 segmentation_checkpoint_path)
        #     if object_id not in self.regression_models_cache:
        #         regression_checkpoint_path = os.path.join(self.model_dir, "checkpoints", "regression",
        #                                                   f"obj_{object_id}.ckpt")
        #         self.regression_models_cache[object_id] = RegLoader.load_from_checkpoint(regression_checkpoint_path)
        #
        #     regression_model = self.regression_models_cache[object_id]
        #     segmentation_model = self.segmentation_models_cache[
        #         "first_half"] if object_id in self.first_half_obj_ids else self.segmentation_models_cache["second_half"]
        #
        # else:
        # if object_id in self.first_half_obj_ids:
        #     segmentation_checkpoint_path = os.path.join(self.model_dir, "checkpoints", "segmentation",
        #                                                 "first_half.ckpt")
        # else:
        #     segmentation_checkpoint_path = os.path.join(self.model_dir, "checkpoints", "segmentation",
        #                                                 "second_half.ckpt")
        segmentation_checkpoint_path = os.path.join(self.model_dir, "checkpoints", "segmentation",
                                                    "all_objs.ckpt")

        regression_checkpoint_path = os.path.join(self.model_dir, "checkpoints", "regression",
                                                  f"obj_{object_id}.ckpt")

        segmentation_model = SegLoader.load_from_checkpoint(segmentation_checkpoint_path)
        regression_model = RegLoader.load_from_checkpoint(regression_checkpoint_path)

        return segmentation_model, regression_model

    def detect_and_match(self, rgbs, Ks, Rts, segmentation_model, object_id):
        rgb_cam1, rgb_cam2, rgb_cam3 = rgbs
        K_cam1, K_cam2, K_cam3 = Ks
        Rt_cam1, Rt_cam2, Rt_cam3 = Rts

        if self.part_segmentation:
            seg_preds = []
            for rgb in [rgb_cam1, rgb_cam2, rgb_cam3]:
                with torch.no_grad():
                    rgb = rgb.transpose(2, 0, 1) / 255.0
                    rgb_tensor = torch.tensor(rgb).to(device="cuda").float()
                    rgb_tensor = rgb_tensor.unsqueeze(0)
                    seg_pred = segmentation_model(rgb_tensor)
                    seg_pred = seg_pred
                    seg_preds.append(seg_pred)

                    # delete_var(rgb_tensor)
                    # delete_var(seg_pred)
                    # torch.cuda.empty_cache()

            seg_preds = np.concatenate(seg_preds, axis=0)

        else:
            rgbs = np.stack((rgb_cam1, rgb_cam2, rgb_cam3), axis=0).transpose((0, 3, 1, 2)) / 255.0

            with torch.no_grad():
                rgbs = torch.from_numpy(rgbs).to(device="cuda").float()
                seg_preds = segmentation_model(rgbs)

                rgbs = rgbs.detach().cpu()
                # delete_var(rgbs)

        segmentation_label_of_obj = object_id_to_segmentation_label(object_id)
        threshold_for_confidence = 0.93
        segmentation_masks = {}
        segmentation_boxes = {}
        matched_masks = {}

        for i, cam_preds in enumerate(seg_preds):
            cam_masks = cam_preds["masks"].cpu().numpy()
            cam_scores = cam_preds["scores"].cpu().numpy()
            cam_labels = cam_preds["labels"].cpu().numpy()
            cam_boxes = cam_preds["boxes"].cpu().numpy()

            indices_of_interested_obj = np.where(cam_labels == segmentation_label_of_obj)[0]
            indices_of_accepted_masks = np.where(cam_scores[indices_of_interested_obj] > threshold_for_confidence)[0]

            if len(indices_of_accepted_masks) == 0:
                print("At least one of the cameras did not detect the object")
                return matched_masks

            masks_of_obj = cam_masks[indices_of_interested_obj][indices_of_accepted_masks]
            boxes_of_obj = cam_boxes[indices_of_interested_obj][indices_of_accepted_masks]

            masks_of_obj = masks_of_obj.squeeze(axis=1)
            masks_of_obj = np.where(masks_of_obj > 0.6, 1, 0).astype(np.uint8)

            segmentation_masks.update({f"cam{i + 1}": masks_of_obj})
            segmentation_boxes.update({f"cam{i + 1}": boxes_of_obj})

        # delete_var(seg_preds)
        # torch.cuda.empty_cache()

            matched_masks["cam1"], matched_masks["cam2"], matched_masks["cam3"] = multi_view_match(segmentation_boxes,
                                                                                                   segmentation_masks,
                                                                                                   [K_cam1, K_cam2,
                                                                                                    K_cam3],
                                                                                                   [Rt_cam1, Rt_cam2,
                                                                                                    Rt_cam3])

            return matched_masks

    def estimate_radii_maps(self, rgbs, depths, Ks, Rts_to_ref, matched_masks, regression_model):
        rgb_cam1, rgb_cam2, rgb_cam3 = rgbs
        depth_cam1, depth_cam2, depth_cam3 = depths
        K_cam1, K_cam2, K_cam3 = Ks
        RT1_to_ref, RT2_to_ref, RT3_to_ref = Rts_to_ref

        segmented_rgbs = {}
        segmented_depths = {}
        C1, C2, C3 = len(matched_masks["cam1"]), len(matched_masks["cam2"]), len(matched_masks["cam3"])

        segmented_rgbs["cam1"] = np.tile(rgb_cam1, [C1, 1, 1, 1])
        segmented_depths["cam1"] = np.tile(depth_cam1, [C1, 1, 1])
        segmented_rgbs["cam2"] = np.tile(rgb_cam2, [C2, 1, 1, 1])
        segmented_depths["cam2"] = np.tile(depth_cam2, [C2, 1, 1])
        segmented_rgbs["cam3"] = np.tile(rgb_cam3, [C3, 1, 1, 1])
        segmented_depths["cam3"] = np.tile(depth_cam3, [C3, 1, 1])

        segmented_rgbs["cam1"][np.logical_not(matched_masks["cam1"]), :] = 0
        segmented_depths["cam1"][np.logical_not(matched_masks["cam1"])] = 0
        segmented_rgbs["cam2"][np.logical_not(matched_masks["cam2"]), :] = 0
        segmented_depths["cam2"][np.logical_not(matched_masks["cam2"])] = 0
        segmented_rgbs["cam3"][np.logical_not(matched_masks["cam3"]), :] = 0
        segmented_depths["cam3"][np.logical_not(matched_masks["cam3"])] = 0

        # Create one batch with all the segmented rgbs
        segmented_rgbs = np.concatenate((segmented_rgbs["cam1"], segmented_rgbs["cam2"], segmented_rgbs["cam3"]),
                                        axis=0)

        # Perform regression
        segmented_rgbs = segmented_rgbs / 255.0
        segmented_rgbs = segmented_rgbs.transpose(0, 3, 1, 2)
        # regression_model = regression_model.to("cuda")
        # print(f"{len(segmented_rgbs) = }")
        if self.part_regression:
            part_size = self.regression_part_size
            reg_preds = []
            for i in range(len(segmented_rgbs) // part_size if len(segmented_rgbs) % part_size == 0 else len(
                    segmented_rgbs) // part_size + 1):
                parted_segmented_rgbs = segmented_rgbs[i * part_size:(i + 1) * part_size] if i * part_size < len(
                    segmented_rgbs) else segmented_rgbs[i * part_size:]

                with torch.no_grad():
                    parted_segmented_rgbs = torch.from_numpy(parted_segmented_rgbs).to(device="cuda").float()
                    parted_reg_preds = regression_model(parted_segmented_rgbs)
                    self.get_logger().info(f"{torch.cuda.memory_summary()}")

                parted_reg_preds = parted_reg_preds.cpu().numpy()

                reg_preds.append(parted_reg_preds)

                # delete_var(parted_segmented_rgbs)
                # delete_var(parted_reg_preds)
                # torch.cuda.empty_cache()

            reg_preds = np.concatenate(reg_preds, axis=0)

        else:
            with torch.cuda.amp.autocast(), torch.no_grad():
                # segmented_rgbs = torch.from_numpy(segmented_rgbs).to(device="cuda").float()
                segmented_rgbs = torch.from_numpy(segmented_rgbs).to(device="cuda").half()
                reg_preds = regression_model(segmented_rgbs).float().cpu().numpy()


        reg_preds = reg_preds.transpose(0, 2, 3, 1)

        radii_maps = {}
        radii_maps["cam1"] = reg_preds[0:C1, :, :, :]
        radii_maps["cam2"] = reg_preds[C1:C1 + C2, :, :, :]
        radii_maps["cam3"] = reg_preds[C1 + C2:, :, :, :]

        # Segmenting the estimated radial maps using the zero values in the depth rather than the mask
        n, u, v = np.where(segmented_depths["cam1"] == 0)
        radii_maps["cam1"][n, u, v, :] = 0
        n, u, v = np.where(segmented_depths["cam2"] == 0)
        radii_maps["cam2"][n, u, v, :] = 0
        n, u, v = np.where(segmented_depths["cam3"] == 0)
        radii_maps["cam3"][n, u, v, :] = 0

        object_xyzs_cam1, non_zero_indices_cam1 = depth_to_point_cloud(K_cam1, segmented_depths["cam1"])
        object_xyzs_cam2, non_zero_indices_cam2 = depth_to_point_cloud(K_cam2, segmented_depths["cam2"])
        object_xyzs_cam3, non_zero_indices_cam3 = depth_to_point_cloud(K_cam3, segmented_depths["cam3"])

        object_xyzs_cam1_2_ref = [transform_point_cloud(x, RT1_to_ref) for x in object_xyzs_cam1]
        object_xyzs_cam2_2_ref = [transform_point_cloud(x, RT2_to_ref) for x in object_xyzs_cam2]
        object_xyzs_cam3_2_ref = [transform_point_cloud(x, RT3_to_ref) for x in object_xyzs_cam3]

        all_object_xyzs = [
            np.concatenate([object_xyzs_cam1_2_ref[i], object_xyzs_cam2_2_ref[i], object_xyzs_cam3_2_ref[i]],
                           axis=0) for i in range(len(object_xyzs_cam1_2_ref))]
        all_object_radii_maps = [
            np.concatenate([radii_maps["cam1"][i], radii_maps["cam2"][i], radii_maps["cam3"][i]], axis=0) for i in
            range(len(radii_maps["cam1"]))]

        return all_object_xyzs, all_object_radii_maps


if __name__ == "__main__":
    main()

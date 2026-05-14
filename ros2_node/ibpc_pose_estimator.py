import gc
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation
import sys
from typing import List, Optional, Union

from geometry_msgs.msg import Pose as PoseMsg
from ibpc_interfaces.msg import Camera as CameraMsg
from ibpc_interfaces.msg import Photoneo as PhotoneoMsg
from ibpc_interfaces.msg import PoseEstimate as PoseEstimateMsg
from ibpc_interfaces.srv import GetPoseEstimates

import rclpy
from rclpy.node import Node

import os
import open3d as o3d
import torch
import subprocess
from inference.utils.horn import HornPoseFitting
from inference.utils.ransac_experimental import RANSAC_w_refinement_adaptive
from inference.utils.epipolar_matching import compute_cost_matrix, match_objects, multi_view_match, multi_view_match_two_cams
from inference.utils.model_loaders import RegLoader, SegLoader
from inference.utils import ddd_utils
from inference.utils.ddd_utils import refinement_by_rotation, transform_point_cloud, depth_to_point_cloud, perform_icp


# Helper functions
def ros_pose_to_mat(pose: PoseMsg):
    r = Rotation.from_quat(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )
    matrix = r.as_matrix()
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = matrix
    pose_matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return pose_matrix

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

class Camera:
    """
    Represents a camera with its pose, intrinsics, and image data,
    initialized from either a CameraMsg or a PhotoneoMsg ROS message.

    Attributes:
        name (str): The name of the camera (taken from the message's frame_id).
        pose (np.ndarray): The 4x4 camera pose matrix (world-to-camera transformation),
                          converted from the ROS message's pose.
        intrinsics (np.ndarray): The 3x3 camera intrinsics matrix, reshaped from the
                                  message's K matrix.
        rgb (np.ndarray): The RGB image data, converted from the ROS message.
        depth (np.ndarray): The depth image data, converted from the ROS message.
        aolp (np.ndarray, optional): The Angle of Linear Polarization data,
                                     converted from the CameraMsg (None for PhotoneoMsg).
        dolp (np.ndarray, optional): The Degree of Linear Polarization data,
                                     converted from the CameraMsg (None for PhotoneoMsg).
    """

    def __init__(self, msg: Union[CameraMsg, PhotoneoMsg]):
        """
        Initializes a new Camera object from a ROS message.

        Args:
            msg: Either a CameraMsg or a PhotoneoMsg ROS message containing
                 camera information.

        Raises:
           TypeError: If the input `msg` is not of the expected type.
        """
        br = CvBridge()

        if not isinstance(msg, (CameraMsg, PhotoneoMsg)):
            raise TypeError("Input message must be of type CameraMsg or PhotoneoMsg")

        self.name: str = (msg.info.header.frame_id,)
        self.pose: np.ndarray = ros_pose_to_mat(msg.pose)
        self.intrinsics: np.ndarray = np.array(msg.info.k).reshape(3, 3)
        self.rgb = br.imgmsg_to_cv2(msg.rgb)
        self.depth = br.imgmsg_to_cv2(msg.depth)
        if isinstance(msg, CameraMsg):
            self.aolp: Optional[np.ndarray] = br.imgmsg_to_cv2(msg.aolp)
            self.dolp: Optional[np.ndarray] = br.imgmsg_to_cv2(msg.dolp)
        else:  # PhotoneoMsg
            self.aolp: Optional[np.ndarray] = None
            self.dolp: Optional[np.ndarray] = None


class PoseEstimator(Node):

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
        self.regression_part_size = 2
        self.part_segmentation = True
        self.cache_models = False
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
        object_ids = os.listdir("/opt/ros/underlay/install/3d_models/")
        object_ids = [int(id.split("_")[1].split(".")[0]) for id in object_ids if id.endswith(".ply")]
        for id in object_ids:
            obj_mesh = o3d.io.read_triangle_mesh(f"/opt/ros/underlay/install/3d_models/obj_{str(id).zfill(6)}.ply")
            obj_mesh_as_pc = obj_mesh.sample_points_uniformly(500)
            obj_points = np.asarray(obj_mesh_as_pc.points)
            self.objects_model_points.update({str(id): obj_points})

        if self.cache_models:
            for id in object_ids:
                regression_checkpoint_path = os.path.join(self.model_dir, "checkpoints", "regression",
                                                          f"obj_{id}.ckpt")
                self.regression_models_cache.update(
                    {str(id): RegLoader.load_from_checkpoint(regression_checkpoint_path)})

        segmentation_checkpoint_path = os.path.join(self.model_dir, "checkpoints", "segmentation", "all_objs.ckpt")
        self.segmentation_models_cache.update(
            {"all_objs": SegLoader.load_from_checkpoint(segmentation_checkpoint_path)})

    def srv_cb(self, request, response):
        if len(request.object_ids) == 0:
            self.get_logger().warn("Received request with empty object_ids.")
            return response
        if len(request.cameras) < 3:
            self.get_logger().warn("Received request with insufficient cameras.")
            return response
        # try:
        cam_1 = Camera(request.cameras[0])
        cam_2 = Camera(request.cameras[1])
        cam_3 = Camera(request.cameras[2])
        photoneo = Camera(request.photoneo)
        response.pose_estimates = self.get_pose_estimates(request.object_ids, cam_1, cam_2, cam_3, photoneo)
        # except:
        #     self.get_logger().error("Error calling get_pose_estimates.")
        return response

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

        # rgb_cam1 = np.tile(cam_1.rgb[:, :, None], (1, 1, 3))
        # rgb_cam2 = np.tile(cam_2.rgb[:, :, None], (1, 1, 3))
        # rgb_cam3 = np.tile(cam_3.rgb[:, :, None], (1, 1, 3))

        # depth_cam1 = cam_1.depth * 0.1
        # depth_cam2 = cam_2.depth * 0.1
        # depth_cam3 = cam_3.depth * 0.1

        Rt_cam1 = cam_1.pose
        Rt_cam2 = cam_2.pose
        Rt_cam3 = cam_3.pose

        K_cam1 = cam_1.intrinsics
        K_cam2 = cam_2.intrinsics
        K_cam3 = cam_3.intrinsics
        # K_ref = photoneo.intrinsics

        RT1_to_ref = np.linalg.inv(Rt_cam1)
        # RT2_to_ref = np.linalg.inv(Rt_cam2)
        # RT3_to_ref = np.linalg.inv(Rt_cam3)


        for object_id in object_ids:
            global_time = time.time()


            masks_of_cam1 = self.match_three_cams_return_one(
                [np.tile(cam_1.rgb[:, :, None], (1, 1, 3)), np.tile(cam_2.rgb[:, :, None], (1, 1, 3)), np.tile(cam_3.rgb[:, :, None], (1, 1, 3))],
                [K_cam1, K_cam2, K_cam3], [Rt_cam1, Rt_cam2, Rt_cam3], object_id)

            self.get_logger().warn(f"Detected {len(masks_of_cam1)} masks for object {object_id} in camera 1")

            if (not masks_of_cam1) or len(masks_of_cam1) == 0:
                self.get_logger().warn(f"No valid masks for object {object_id}: ")
                continue

            st_time = time.time()
            all_object_xyzs, all_object_radii_maps, non_zero_indices = self.estimate_radii_maps_one_cam(np.tile(cam_1.rgb[:, :, None], (1, 1, 3)),
                                                                                                        cam_1.depth * 0.1,
                                                                                                        K_cam1,
                                                                                                        RT1_to_ref,
                                                                                                        masks_of_cam1,
                                                                                                        object_id)

            del masks_of_cam1
            gc.collect()

            self.get_logger().info("Time to run the regression model: {:.2f} seconds".format(time.time() - st_time))

            all_estimated_kpts = np.zeros((len(all_object_xyzs), 4, 3))
            ransac_time = 0
            refinement_time = 0
            for i in range(len(all_object_xyzs)):
                object_estimated_kpts = np.zeros((4, 3))
                object_xyz = all_object_xyzs[i]
                object_radii = all_object_radii_maps[i]
                # Only keeping the non-zero radii maps
                # u, v = np.where(object_radii[:, :, 0] != 0)
                object_radii = object_radii[non_zero_indices[i][0], non_zero_indices[i][1], :]

                self.get_logger().info("number of object points: {}".format(len(all_object_xyzs[i])))
                if object_radii.shape[0] != object_xyz.shape[0]:
                    self.get_logger().warn(f"Object {object_id} has different number of points and radii.")
                    continue
                if object_xyz.shape[0] < 4:
                    self.get_logger().warn(
                        f"Skipping object {object_id} because it has only {object_xyz.shape[0]} points.")
                    continue

                st_time = time.time()
                for keypoint_index in range(4):
                    center_mm_s = RANSAC_w_refinement_adaptive(object_xyz, object_radii[:, keypoint_index],
                                                               iterations=300, initial_epsilon=1, MAX_REFINEMENTS=1)
                    all_estimated_kpts[i, keypoint_index] = center_mm_s[0]
                    object_estimated_kpts[keypoint_index] = center_mm_s[0]
                ransac_time += time.time() - st_time

                estimated_pose = np.eye(4)
                horn_solver = HornPoseFitting()

                horn_solver.lmshorn(self.keypoints, object_estimated_kpts, 4, estimated_pose)

                mesh_points = (self.objects_model_points[str(object_id)]).copy()

                st_time = time.time()
                if self.perform_refinement:
                    transfered_mesh_with_icp, refined_transformation, final_error = refinement_by_rotation(
                        all_object_xyzs[i], estimated_pose, mesh_points)
                    # transfered_mesh_with_icp, refined_transformation = refinement_by_rotation(
                    #     all_object_xyzs[i], refined_transformation, mesh_points)
                else:
                    object_pc = o3d.geometry.PointCloud()
                    object_pc.points = o3d.utility.Vector3dVector(all_object_xyzs[i])
                    transfered_mesh_with_icp, refined_transformation = perform_icp(object_pc, mesh_points,
                                                                                   estimated_pose)

                refinement_time += time.time() - st_time

                final_transformation = refined_transformation.copy()
                pose_estimate = PoseEstimateMsg()
                pose_estimate.obj_id = object_id
                # TODO: calculate the score based on the two point clouds euclidean distance
                pose_estimate.score = self.calc_pose_score(final_error) if self.perform_refinement else 1.0
                pose_estimate.pose.position.x = final_transformation[0, 3]
                pose_estimate.pose.position.y = final_transformation[1, 3]
                pose_estimate.pose.position.z = final_transformation[2, 3]
                rot = rot_to_quat(final_transformation[0:3, 0:3])
                pose_estimate.pose.orientation.x = rot[0]
                pose_estimate.pose.orientation.y = rot[1]
                pose_estimate.pose.orientation.z = rot[2]
                pose_estimate.pose.orientation.w = rot[3]
                pose_estimates.append(pose_estimate)

            torch.cuda.empty_cache()
            # torch.cuda.ipc_collect()

            # self.get_logger().info(f"Detected {len(all_object_xyzs)} object with id {object_id} and estimated their pose")
            out = subprocess.check_output(['free', '-b']).decode().splitlines()[1].split()
            total, used, free = map(int, (out[1], out[2], out[3]))
            # self.get_logger().info("Ram Stats:")
            self.get_logger().info(
                f"Total: {total / (1024 ** 3):.2f} GiB, Used: {used / (1024 ** 3):.2f} GiB, Free: {free / (1024 ** 3):.2f} GiB")
            self.get_logger().info(
                f"RANSAC time for a scene with {len(all_object_xyzs)} objects detected: {ransac_time:.2f} seconds")
            self.get_logger().info(f"Here are the ICP thresholds: {ddd_utils.get_thresholds()}")
            self.get_logger().info(
                f"Here is the calculated pose score for the object with above thresholds: {self.calc_pose_score(final_error)}")
            self.get_logger().info(
                f"Refinement time for a scene with {len(all_object_xyzs)} objects detected: {refinement_time:.2f} seconds")
            self.get_logger().info(
                f"Total time for a scene with {len(all_object_xyzs)} objects detected: {time.time() - global_time:.2f} seconds")
            # print("\nHere are the estimated poses: \n", pose_estimates)
        self.call_counter += 1
        self.get_logger().info(
            f"Pose estimates called {self.call_counter} times in the totatl time of {time.time() - self.estimator_construction_time:.2f} seconds")

        return pose_estimates

    def load_regression_model(self, object_id):

        regression_checkpoint_path = os.path.join(self.model_dir, "checkpoints", "regression",
                                                  f"obj_{object_id}.ckpt")

        regression_model = RegLoader.load_from_checkpoint(regression_checkpoint_path)

        return regression_model

    def match_three_cams_return_one(self, rgbs, Ks, Rts, object_id):
        # rgb_cam1, rgb_cam2, rgb_cam3 = rgbs
        # K_cam1, K_cam2, K_cam3 = Ks
        # Rt_cam1, Rt_cam2, Rt_cam3 = Rts

        # segmentation_model = self.segmentation_models_cache["all_objs"]
        st_time = time.time()

        seg_preds = self.detect_three_cams([rgbs[0], rgbs[1], rgbs[2]], self.segmentation_models_cache["all_objs"])

        torch.cuda.empty_cache()

        self.get_logger().info(
            "Time to run the segmentation model: {:.2f} seconds".format(time.time() - st_time))

        segmentation_label_of_obj = object_id_to_segmentation_label(object_id)
        threshold_for_confidence = 0.93
        segmentation_masks = {}
        segmentation_boxes = {}
        # matched_masks = {}
        cam1_dets = []
        for i, cam_preds in enumerate(seg_preds):
            # cam_masks = cam_preds["masks"]
            # cam_scores = cam_preds["scores"]
            # cam_labels = cam_preds["labels"]
            # cam_boxes = cam_preds["boxes"]

            indices_of_interested_obj = np.where(cam_preds["labels"] == segmentation_label_of_obj)[0]
            indices_of_accepted_masks = np.where(cam_preds["scores"][indices_of_interested_obj] > threshold_for_confidence)[0]

            if len(indices_of_accepted_masks) == 0:
                print("At least one of the cameras did not detect the object")
                return cam1_dets

            masks_of_obj = cam_preds["masks"][indices_of_interested_obj][indices_of_accepted_masks]
            # boxes_of_obj = cam_preds["boxes"][indices_of_interested_obj][indices_of_accepted_masks]

            masks_of_obj = masks_of_obj.squeeze(axis=1)
            masks_of_obj = np.where(masks_of_obj > 0.6, 1, 0).astype(np.uint8)

            segmentation_masks.update({f"cam{i + 1}": masks_of_obj})
            segmentation_boxes.update({f"cam{i + 1}": cam_preds["boxes"][indices_of_interested_obj][indices_of_accepted_masks]})

        # delete_var(seg_preds)
        # torch.cuda.empty_cache()

        # matched_masks["cam1"], _, _ = multi_view_match(segmentation_boxes, segmentation_masks,[Ks[0], Ks[1], Ks[2]],
        #                                                 [Rts[0], Rts[1], Rts[2]])

        matches = multi_view_match(segmentation_boxes, segmentation_masks,[Ks[0], Ks[1], Ks[2]],
                                                        [Rts[0], Rts[1], Rts[2]])

        for i, j, k in matches:
            cam1_dets.append(segmentation_masks["cam1"][i])


        return cam1_dets

    def detect_three_cams(self, rgbs, segmentation_model):
        rgb_cam1, rgb_cam2, rgb_cam3 = rgbs

        if self.part_segmentation:
            seg_preds = []
            for rgb in [rgb_cam1, rgb_cam2, rgb_cam3]:
                with torch.no_grad():
                    rgb = rgb.transpose(2, 0, 1) / 255.0
                    rgb_tensor = torch.tensor(rgb).to(device="cuda").float()
                    rgb_tensor = rgb_tensor.unsqueeze(0)
                    seg_pred = segmentation_model(rgb_tensor)

                    seg_pred[0]["masks"] = seg_pred[0]["masks"].cpu().numpy()
                    seg_pred[0]["scores"] = seg_pred[0]["scores"].cpu().numpy()
                    seg_pred[0]["labels"] = seg_pred[0]["labels"].cpu().numpy()
                    seg_pred[0]["boxes"] = seg_pred[0]["boxes"].cpu().numpy()
                    seg_preds.append(seg_pred)


                    rgb_tensor = rgb_tensor.detach().cpu()
                    # seg_pred = seg_pred.detach().cpu()
                    rgb_tensor = None
                    seg_pred = None
                    del rgb_tensor
                    del seg_pred
                    gc.collect()
                    torch.cuda.empty_cache()

            seg_preds = np.concatenate(seg_preds, axis=0)

        else:
            rgbs = np.stack((rgb_cam1, rgb_cam2, rgb_cam3), axis=0).transpose((0, 3, 1, 2)) / 255.0

            with torch.no_grad():
                rgbs = torch.from_numpy(rgbs).to(device="cuda").float()
                seg_preds = segmentation_model(rgbs)

                rgbs = rgbs.detach().cpu()
                del rgbs

            for i in range(len(seg_preds)):
                seg_preds[i]["masks"] = seg_preds[i]["masks"].cpu().numpy()
                seg_preds[i]["scores"] = seg_preds[i]["scores"].cpu().numpy()
                seg_preds[i]["labels"] = seg_preds[i]["labels"].cpu().numpy()
                seg_preds[i]["boxes"] = seg_preds[i]["boxes"].cpu().numpy()

        return seg_preds

    def estimate_radii_maps_one_cam(self, rgb, depth, cam_K, RT1_to_ref, segmentation_masks, object_id):

        segmented_rgbs = np.tile(rgb, [len(segmentation_masks), 1, 1, 1]).astype(np.float32)

        segmented_rgbs[np.logical_not(segmentation_masks), :] = 0


        # Perform regression
        segmented_rgbs = segmented_rgbs.astype(np.float32)
        segmented_rgbs = segmented_rgbs / 255.0
        segmented_rgbs = segmented_rgbs.transpose(0, 3, 1, 2)

        if self.cache_models:
            regression_model = self.regression_models_cache[str(object_id)]
        else:
            regression_model = self.load_regression_model(object_id)

        # if self.part_regression:
        #     part_size = self.regression_part_size
        #     reg_preds = []
        #     for i in range(len(segmentation_masks) // part_size if len(segmentation_masks) % part_size == 0 else len(
        #             segmentation_masks) // part_size + 1):
        #         number_of_segmentations = part_size if i * part_size + part_size < len(segmentation_masks) else len(segmentation_masks) % part_size
        #
        #         parted_segmented_rgbs = np.tile(rgb, [number_of_segmentations, 1, 1, 1]).astype(np.float32)
        #         parted_segmented_rgbs[i * part_size:(i + 1) * part_size] if i * part_size < len(segmented_rgbs) else segmented_rgbs[i * part_size:] = 0
        #         # parted_segmented_rgbs = segmented_rgbs[i * part_size:(i + 1) * part_size] if i * part_size < len(
        #         #     segmented_rgbs) else segmented_rgbs[i * part_size:]
        #         parted_segmented_rgbs = parted_segmented_rgbs.astype(np.float32)
        #         parted_segmented_rgbs = parted_segmented_rgbs / 255.0
        #         parted_segmented_rgbs = parted_segmented_rgbs.transpose(0, 3, 1, 2)
        #
        #         with torch.cuda.amp.autocast(), torch.no_grad():
        #             parted_segmented_rgbs = torch.from_numpy(parted_segmented_rgbs).to(device="cuda").half()
        #             parted_reg_preds = regression_model(parted_segmented_rgbs)
        #             # self.get_logger().info(f"{torch.cuda.memory_summary()}")
        #
        #         parted_reg_preds = parted_reg_preds.cpu().numpy()
        #
        #         reg_preds.append(parted_reg_preds)
        #
        #         del parted_segmented_rgbs
        #         del parted_reg_preds
        #         gc.collect()
        #         # torch.cuda.empty_cache()
        # 
        #     reg_preds = np.concatenate(reg_preds, axis=0)
        if self.part_regression:
            part_size = self.regression_part_size
            reg_preds = []
            for i in range(len(segmented_rgbs) // part_size if len(segmented_rgbs) % part_size == 0 else len(
                    segmented_rgbs) // part_size + 1):
                parted_segmented_rgbs = segmented_rgbs[i * part_size:(i + 1) * part_size] if i * part_size < len(
                    segmented_rgbs) else segmented_rgbs[i * part_size:]

                with torch.cuda.amp.autocast(), torch.no_grad():
                    parted_segmented_rgbs = torch.from_numpy(parted_segmented_rgbs).to(device="cuda").half()
                    parted_reg_preds = regression_model(parted_segmented_rgbs)
                    # self.get_logger().info(f"{torch.cuda.memory_summary()}")

                parted_reg_preds = parted_reg_preds.cpu().numpy()

                reg_preds.append(parted_reg_preds)

                del parted_segmented_rgbs
                del parted_reg_preds
                gc.collect()
                # torch.cuda.empty_cache()

            reg_preds = np.concatenate(reg_preds, axis=0)

        else:
            with torch.cuda.amp.autocast(), torch.no_grad():
                # segmented_rgbs = torch.from_numpy(segmented_rgbs).to(device="cuda").float()
                segmented_rgbs = torch.from_numpy(segmented_rgbs).to(device="cuda").half()
                reg_preds = regression_model(segmented_rgbs).float().cpu().numpy()

        reg_preds = reg_preds.transpose(0, 2, 3, 1)

        # Segmenting the estimated radial maps using the zero values in the depth rather than the mask
        # n, u, v = np.where(segmented_depths == 0)
        # reg_preds[n, u, v, :] = 0
        # segmented_depths = np.tile(depth, [len(segmentation_masks), 1, 1]).astype(np.float32)
        # segmented_depths[np.logical_not(segmentation_masks)] = 0
        #
        # object_xyzs, non_zero_indices = depth_to_point_cloud(cam_K, segmented_depths)

        # B, H, W = batch_of_depths.shape
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]

        object_xyzs = []
        non_zero_indices = []

        for i in range(len(segmentation_masks)):
            segmented_depth = depth.copy()
            segmented_depth[segmentation_masks[i] == 0] = 0

            vs, us = np.nonzero(segmented_depth)
            zs = segmented_depth[vs, us]
            xs = ((us - cx) * zs) / fx
            ys = ((vs - cy) * zs) / fy
            pts = np.stack([xs, ys, zs], axis=-1)
            object_xyzs.append(pts)
            non_zero_indices.append((vs, us))


        object_xyzs = [transform_point_cloud(x, RT1_to_ref) for x in object_xyzs]

        return object_xyzs, reg_preds, non_zero_indices


    def calc_pose_score(self, error):
        scaling_factor = 20.0
        score = float(np.exp(-error / scaling_factor))
        score = max(min(score, 1.0), 1e-6)

        return score

def main(argv=sys.argv):
    rclpy.init(args=argv)

    pose_estimator = PoseEstimator()

    rclpy.spin(pose_estimator)

    rclpy.shutdown()


if __name__ == "__main__":
    main()

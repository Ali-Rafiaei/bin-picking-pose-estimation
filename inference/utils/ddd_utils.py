import open3d as o3d
import numpy as np

thresholds = []
def get_thresholds():
    return thresholds

def perform_icp(object_pc, mesh_points, estimated_pose):
    mesh_as_pc = o3d.geometry.PointCloud()
    mesh_as_pc.points = o3d.utility.Vector3dVector(mesh_points)

    transferred_mesh_points = np.dot(mesh_points, estimated_pose[0:3, :3].T) + estimated_pose[0:3, 3].T
    estimated_pc = o3d.geometry.PointCloud()
    estimated_pc.points = o3d.utility.Vector3dVector(transferred_mesh_points)

    euclidean_distance_bf_icp = np.mean(object_pc.compute_point_cloud_distance(estimated_pc))
    threshold = euclidean_distance_bf_icp
    global thresholds
    thresholds.append(threshold)

    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    initial_transformation = estimated_pose
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)

    p2p_registration = o3d.pipelines.registration.registration_icp(mesh_as_pc, object_pc, threshold,
                                                                   initial_transformation,
                                                                   estimation_method, criteria)

    transfered_mesh_with_icp = mesh_as_pc
    transfered_mesh_with_icp.transform(p2p_registration.transformation)

    return transfered_mesh_with_icp, p2p_registration.transformation

def rot_x(angle_deg):
    angle = np.radians(angle_deg)
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle),  np.cos(angle)]
    ])

def rot_y(angle_deg):
    angle = np.radians(angle_deg)
    return np.array([
        [ np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

def rot_z(angle_deg):
    angle = np.radians(angle_deg)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])

def refinement_by_rotation(object_points, initial_estimated_pose, mesh_points):
    global thresholds
    thresholds = []
    object_pc = o3d.geometry.PointCloud()
    object_pc.points = o3d.utility.Vector3dVector(object_points)

    initial_icp_mesh, transformation = perform_icp(object_pc, mesh_points, initial_estimated_pose)
    euclidean_distance_wout_refinement = np.mean(object_pc.compute_point_cloud_distance(initial_icp_mesh))
    thresholds.append(euclidean_distance_wout_refinement)


    R_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    R_y_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    R_z_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    R_x_90_pos = rot_x(90)
    R_x_90_neg = rot_x(-90)
    R_y_90_pos = rot_y(90)
    R_y_90_neg = rot_y(-90)
    R_z_90_pos = rot_z(90)
    R_z_90_neg = rot_z(-90)

    rotations = np.array([np.eye(3, 3), R_x_180, R_y_180, R_z_180, R_x_90_pos, R_y_90_pos, R_z_90_pos, R_x_90_neg, R_y_90_neg, R_z_90_neg])

    errors = np.zeros(len(rotations))
    icp_transformations = np.zeros((len(rotations), 4, 4))

    for i in range(len(rotations)):
        rotated_icp_transformation = np.eye(4)
        rotated_icp_transformation[0:3, 0:3] = np.dot(transformation[0:3, 0:3], rotations[i])
        rotated_icp_transformation[0:3, 3] = transformation[0:3, 3]

        second_icp_transferred_model_pc, second_transformation = perform_icp(object_pc, mesh_points,
                                                                             rotated_icp_transformation)
        second_icp_transferred_model_points = np.asarray(second_icp_transferred_model_pc.points)

        euclidean_distance = np.mean(object_pc.compute_point_cloud_distance(second_icp_transferred_model_pc))

        errors[i] = euclidean_distance
        icp_transformations[i] = second_transformation

    lowest_refinement_error_index = np.argmin(errors)

    if errors[lowest_refinement_error_index] < euclidean_distance_wout_refinement:
        best_transformation = icp_transformations[lowest_refinement_error_index]
        final_error = errors[lowest_refinement_error_index]
    else:
        best_transformation = transformation
        final_error = euclidean_distance_wout_refinement


    final_transferred_mesh_points = np.dot(mesh_points, best_transformation[0:3, 0:3].T) + best_transformation[0:3, 3].T
    final_transferred_mesh = o3d.geometry.PointCloud()
    final_transferred_mesh.points = o3d.utility.Vector3dVector(final_transferred_mesh_points)
    return final_transferred_mesh, best_transformation, final_error

def transform_point_cloud(points, RT):
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_h = np.concatenate([points, ones], axis=1)

    # Apply transformation
    points_transformed = (RT @ points_h.T).T
    return points_transformed[:, :3]

def depth_to_point_cloud(K, batch_of_depths):
    B, H, W = batch_of_depths.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    pts_batch = []
    indices = []

    for i in range(B):
        depth = batch_of_depths[i]
        vs, us = np.nonzero(depth)
        zs = depth[vs, us]
        xs = ((us - cx) * zs) / fx
        ys = ((vs - cy) * zs) / fy
        pts = np.stack([xs, ys, zs], axis=-1)
        pts_batch.append(pts)
        indices.append((vs, us))

    return pts_batch, indices
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


def multi_view_match(segmentation_boxes, segmentation_masks, Ks, RTs):
    """
    Use epipolar geometry to match detections from multiple cameras.
    Returns a list of PosePrediction instances.
    """
    predictions = []

    K1, K2, K3 = Ks
    R1, R2, R3 = [x[:3, :3] for x in RTs]
    t1, t2, t3 = [x[:3, 3] for x in RTs]
    F12 = compute_fundamental_matrix(K1, R1, t1, K2, R2, t2)
    F13 = compute_fundamental_matrix(K1, R1, t1, K3, R3, t3)
    F23 = compute_fundamental_matrix(K2, R2, t2, K3, R3, t3)

    cost_matrix = compute_cost_matrix(segmentation_boxes["cam1"], segmentation_boxes["cam2"], segmentation_boxes["cam3"], F12, F13, F23)

    # Hungarian matching + threshold
    matches = match_objects(cost_matrix, threshold=30)
    matches_sorted = sorted(matches, key=lambda t: cost_matrix[t[0], t[1], t[2]])

    cam1_dets, cam2_dets, cam3_dets = [], [], []
    cam1_boxes, cam2_boxes, cam3_boxes = [], [], []
    # for i, j, k in matches_sorted:
    #     cam1_dets.append(segmentation_masks["cam1"][i])
    #     # cam1_boxes.append(segmentation_boxes["cam1"][i])
    #     # cam2_dets.append(segmentation_masks["cam2"][j])
    #     # cam2_boxes.append(segmentation_boxes["cam2"][j])
    #     # cam3_dets.append(segmentation_masks["cam3"][k])
    #     # cam3_boxes.append(segmentation_boxes["cam3"][k])
    #
    # # return cam1_dets, cam2_dets, cam3_dets
    # return cam1_dets
    return matches_sorted

def multi_view_match_two_cams(segmentation_boxes, segmentation_masks, Ks, RTs):
    """
    Use epipolar geometry to match detections from multiple cameras.
    Returns a list of PosePrediction instances.
    """
    predictions = []

    K1, K2 = Ks
    R1, R2 = [x[:3, :3] for x in RTs]
    t1, t2 = [x[:3, 3] for x in RTs]
    F12 = compute_fundamental_matrix(K1, R1, t1, K2, R2, t2)

    cost_matrix = compute_cost_matrix_two_cams(segmentation_boxes["cam1"], segmentation_boxes["cam2"], F12)

    # Hungarian matching + threshold
    matches = match_objects_two_cams(cost_matrix, threshold=30)
    matches_sorted = sorted(matches, key=lambda t: cost_matrix[t[0], t[1]])

    cam1_dets, cam2_dets = [], []
    cam1_boxes, cam2_boxes = [], []
    for i, j in matches_sorted:
        cam1_dets.append(segmentation_masks["cam1"][i])
        cam1_boxes.append(segmentation_boxes["cam1"][i])
        cam2_dets.append(segmentation_masks["cam2"][j])
        cam2_boxes.append(segmentation_boxes["cam2"][j])


    return cam1_dets, cam2_dets, cam1_boxes, cam2_boxes

def compute_fundamental_matrix(K1, R1, t1, K2, R2, t2):
    """Compute the fundamental matrix between two cameras."""
    t1 = t1.flatten()
    t2 = t2.flatten()
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1

    # Skew-symmetric matrix
    tx = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ], dtype=np.float32)

    E = tx @ R_rel
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    F = (K2_inv.T @ E @ K1_inv)

    # Normalize
    if abs(F[2, 2]) > 1e-8:
        F /= F[2, 2]

    return F

def compute_cost_matrix(dets1, dets2, dets3, F12, F13, F23, img1=None, img2=None, img3=None):
    """
    NxMxP cost matrix from bounding-box centers.
    """
    N, M, P = len(dets1), len(dets2), len(dets3)
    cost = np.zeros((N, M, P), dtype=np.float32)

    for i in range(N):
        x1, y1, x2, y2 = dets1[i]
        pt1 = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        for j in range(M):
            x1, y1, x2, y2 = dets2[j]
            pt2 = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            for k in range(P):
                x1, y1, x2, y2 = dets3[k]
                pt3 = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                cost[i, j, k] = epipolar_error_full(pt1, pt2, pt3, F12, F13, F23)

    return cost

def compute_cost_matrix_two_cams(dets1, dets2, F12):
    """
    NxMxP cost matrix from bounding-box centers.
    """
    N, M = len(dets1), len(dets2)
    cost = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        x1, y1, x2, y2 = dets1[i]
        pt1 = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        for j in range(M):
            x1, y1, x2, y2 = dets2[j]
            pt2 = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            cost[i, j] = epipolar_error(pt1, pt2, F12)

    return cost


def epipolar_error(pt1, pt2, F, img1=None, img2=None):
    """
    Symmetric epipolar distance for points pt1 (cam1), pt2 (cam2),
    with visualization of epipolar lines and points.
    """
    pt1_h = np.array([pt1[0], pt1[1], 1.0])
    pt2_h = np.array([pt2[0], pt2[1], 1.0])

    l2 = F @ pt1_h  # Epipolar line in cam2
    l1 = F.T @ pt2_h  # Epipolar line in cam1

    # Normalize the lines
    norm_l1 = np.linalg.norm(l1[:2])
    norm_l2 = np.linalg.norm(l2[:2])

    if norm_l1 > 1e-8:
        l1 /= norm_l1
    if norm_l2 > 1e-8:
        l2 /= norm_l2

    d1 = abs(np.dot(l1, pt1_h)) if norm_l1 > 1e-8 else 9999
    d2 = abs(np.dot(l2, pt2_h)) if norm_l2 > 1e-8 else 9999

    error = 0.5 * (d1 + d2)

    return error


def epipolar_error_full(pt1, pt2, pt3, F12, F13, F23):
    """
    epipolar error across three cams:
      e12 + e13 + e23
    """
    e12 = epipolar_error(pt1, pt2, F12)
    e13 = epipolar_error(pt1, pt3, F13)
    e23 = epipolar_error(pt2, pt3, F23)
    return (e12 + e13 + e23) / 3


def match_objects(cost_matrix, threshold):
    """
    Flatten => Hungarian => keep matches < threshold => list of (i, j, k).
    """
    N, M, P = cost_matrix.shape
    matched = []
    flattened = cost_matrix.reshape(N*M, P)
    row_idx, col_idx = linear_sum_assignment(flattened)

    for r, c in zip(row_idx, col_idx):
        val = flattened[r, c]
        if val < threshold:
            i = r // M
            j = r % M
            k = c
            matched.append((i, j, k))
    return matched

def match_objects_two_cams(cost_matrix, threshold):
    matched = []
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    for i, j in zip(row_idx, col_idx):
        if cost_matrix[i, j] < threshold:
            matched.append((i, j))
    return matched
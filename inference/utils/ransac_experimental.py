from numba import jit, prange, njit
import numpy as np


@jit(nopython=False, parallel=False)
def center_est_numba(point_list, radius_list):
    assert len(point_list) == len(radius_list), 'different number of points and radii'
    assert len(point_list) >= 4, 'less than 4 points'

    A = np.zeros((len(point_list), 5))
    b = np.zeros((len(point_list), 5))

    for i in prange(len(point_list)):
        p = point_list[i]
        r = radius_list[i]
        x = p[0]
        y = p[1]
        z = p[2]
        A[i] = [-2*x, -2*y, -2*z, 1, x*x+y*y+z*z-r*r]
        b[i] = [0, 0, 0, 0, 0]

    U, S, Vh = np.linalg.svd(A)
    X = Vh[-1]
    X /= X[-1]

    return X[0], X[1], X[2]


# Random Sample Consensus looks for a good first guess, uses epsilon to determine inliers
@jit(nopython=False, parallel=False)
def random_center_est_numba(xyz, radial_list, epsilon, iterations=25):
    n = len(xyz)
    votes = np.zeros((iterations, 5))
    all_inlier_indices = np.zeros((iterations, n), dtype=np.uint8)


    for itr in prange(iterations):
        index = np.random.randint(0, n, 4)
        point_list = xyz[index]
        radius_list = radial_list[index]

        x, y, z = center_est_numba(point_list, radius_list)

        consensus = 0
        total_error = 0
        inlier_indices = np.zeros(len(xyz), dtype=np.uint8)
        true_condition_counter = 0

        for i in prange(n):
            p = xyz[i]
            r = radial_list[i]
            dist = np.sqrt((p[0]-x)**2 + (p[1]-y)**2 + (p[2]-z)**2)

            if abs(dist - r) <= epsilon:
                consensus += 1
                total_error += abs(dist-r)
                inlier_indices[true_condition_counter] = i + 1
                true_condition_counter += 1


        votes[itr, 0] = consensus
        votes[itr, 1] = x
        votes[itr, 2] = y
        votes[itr, 3] = z
        votes[itr, 4] = total_error
        all_inlier_indices[itr, :true_condition_counter] = inlier_indices[:true_condition_counter]



    return votes, all_inlier_indices

def select_best_vote(votes, all_inlier_indices):
    top_inlier_count = np.max(votes[:, 0])
    top_vote_indices = np.where(votes[:, 0] >= top_inlier_count * 0.9)[0]

    average_errors = votes[:, 4] / (votes[:, 0] + 1e-10)
    sorted_indices = sorted(top_vote_indices, key=lambda i: average_errors[i])

    best_vote_idx = sorted_indices[0]
    best_vote = votes[best_vote_idx]
    best_inlier_indices = all_inlier_indices[best_vote_idx]
    best_inlier_indices = list(best_inlier_indices[np.nonzero(best_inlier_indices)] - 1)

    return best_vote, best_vote[0] / len(all_inlier_indices[0]), best_inlier_indices

def accumulate_inliers(xyz, radial_list, center, epsilon, early_stop=None):
    if early_stop is None:
        early_stop = len(xyz)

    xyz_inliers = []
    radial_list_inliers = []

    indices = np.arange(len(xyz))
    np.random.shuffle(indices)

    count = 0
    for i in indices:
        if count >= early_stop:
            break
        p = xyz[i]
        r = radial_list[i]
        dist = np.linalg.norm(p - center)
        if abs(dist - r) <= epsilon:
            xyz_inliers.append(p)
            radial_list_inliers.append(r)
            count += 1

    return np.array(xyz_inliers), np.array(radial_list_inliers)

def estimate_adaptive_epsilon(xyz, radial_list, center_estimate, percentile=90, min_eps=0.1, max_eps=2.0):
    errors = np.abs(np.linalg.norm(xyz - center_estimate, axis=1) - radial_list)
    eps = np.percentile(errors, percentile)
    return np.clip(eps, min_eps, max_eps)

def RANSAC_w_refinement(xyz, radial_list, err, iterations=5000, epsilon=0.5, num_inliers_for_refinement=150):

    votes, all_inlier_indices = random_center_est_numba(xyz, radial_list, epsilon, iterations)
    best_vote, ratio, best_inlier_indices = select_best_vote(votes, all_inlier_indices)

    center_from_vote = np.array([best_vote[1], best_vote[2], best_vote[3]])

    # Accumulate inliers again based on best_vote's center
    xyz_inliers, radial_list_inliers = accumulate_inliers(
        xyz, radial_list, center_from_vote, epsilon, early_stop=best_vote[0]
    )

    if len(xyz_inliers) >= 4:
        refined_center = center_est_numba(xyz_inliers, radial_list_inliers)
        refined_center = np.array(refined_center).astype("float64")
        return refined_center, ratio
    else:
        return center_from_vote.astype("float64"), ratio

def RANSAC_w_refinement_adaptive(xyz, radial_list, iterations=5000, initial_epsilon=0.5, MAX_REFINEMENTS=1):

    votes, all_inlier_indices = random_center_est_numba(xyz, radial_list, initial_epsilon, iterations)
    best_vote, ratio, best_inlier_indices = select_best_vote(votes, all_inlier_indices)

    center_from_vote = np.array([best_vote[1], best_vote[2], best_vote[3]])

    # adaptive_epsilon = estimate_adaptive_epsilon(xyz, radial_list, center_from_vote)
    adaptive_epsilon = 0.5  # adaptive estimation disabled; fixed value performed better in competition

    refined_center = center_from_vote
    last_inlier_count = int(best_vote[0])

    for _ in range(MAX_REFINEMENTS):
        xyz_inliers, radial_list_inliers = accumulate_inliers(xyz, radial_list, refined_center, adaptive_epsilon, early_stop=last_inlier_count)
        if len(xyz_inliers) < 4:
            break

        last_inlier_count = len(xyz_inliers)
        refined_center = center_est_numba(xyz_inliers, radial_list_inliers)
        refined_center = np.array(refined_center).astype("float64")


    return refined_center, ratio
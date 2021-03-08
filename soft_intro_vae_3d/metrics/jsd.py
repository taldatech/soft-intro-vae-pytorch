import numpy as np
import torch
from numpy.linalg import norm
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors


__all__ = ['js_divercence_between_pc', 'jsd_between_point_cloud_sets']


#
# Compute JS divergence
#


def js_divercence_between_pc(pc1: torch.Tensor, pc2: torch.Tensor,
                             voxels: int = 64) -> float:
    """Method for computing JSD from 2 sets of point clouds."""
    pc1_ = _pc_to_voxel_distribution(pc1, voxels)
    pc2_ = _pc_to_voxel_distribution(pc2, voxels)
    jsd = _js_divergence(pc1_, pc2_)
    return jsd


def _js_divergence(P, Q):
    # Ensure probabilities.
    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    # Calculate JSD using scipy.stats.entropy()
    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res1 = e_sum - ((e1 + e2) / 2.0)

    # Calcujate JS-Div using manually defined KL divergence.
    # res2 = _jsdiv(P_, Q_)
    #
    # if not np.allclose(res1, res2, atol=10e-5, rtol=0):
    #     warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res1


def _jsdiv(P, Q):
    """Another way of computing JSD to check numerical stability."""
    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


def _pc_to_voxel_distribution(pc: torch.Tensor, n_voxels: int = 64) -> np.ndarray:
    pc_ = pc.clamp(-0.5, 0.4999) + 0.5
    # Because points are in range [0, 1], simple multiplication will bin them.
    pc_ = (pc_ * n_voxels).int()
    pc_ = pc_[:, :, 0] * n_voxels ** 2 + pc_[:, :, 1] * n_voxels + pc_[:, :, 2]

    B = np.zeros(n_voxels**3, dtype=np.int32)
    values, amounts = np.unique(pc_, return_counts=True)
    B[values] = amounts
    return B


#
# Stanford way to calculate JSD
#


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, voxels=28,
                                 in_unit_sphere=True):
    """Computes the JSD between two sets of point-clouds, as introduced in the
    paper ```Learning Representations And Generative Models For 3D Point
    Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        voxels: (int) grid-resolution. Affects granularity of measurements.
    """
    sample_grid_var = _entropy_of_occupancy_grid(sample_pcs, voxels,
                                                 in_unit_sphere)[1]
    ref_grid_var = _entropy_of_occupancy_grid(ref_pcs, voxels,
                                              in_unit_sphere)[1]
    return _js_divergence(sample_grid_var, ref_grid_var)


def _entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False):
    """Given a collection of point-clouds, estimate the entropy of the random
    variables corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    pclouds = pclouds.cpu().numpy()
    epsilon = 10e-4
    bound = 0.5 + epsilon
    # if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
    #     warnings.warn('Point-clouds are not in unit cube.')
    #
    # if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
    #     warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = _unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        p = 0.0
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def _unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing

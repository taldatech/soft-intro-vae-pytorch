# https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html

import numbers
import random
import math
import numpy as np
import torch


class LinearTransformation(object):
    r"""Transforms node positions with a square transformation matrix computed
    offline.

    Args:
        matrix (Tensor): tensor with shape :math:`[D, D]` where :math:`D`
            corresponds to the dimensionality of node positions.
    """

    def __init__(self, matrix):
        assert matrix.dim() == 2, (
            'Transformation matrix should be two-dimensional.')
        assert matrix.size(0) == matrix.size(1), (
            'Transformation matrix should be square. Got [{} x {}] rectangular'
            'matrix.'.format(*matrix.size()))

        self.matrix = matrix

    def __call__(self, data):
        pos = data.pos.view(-1, 1) if data.pos.dim() == 1 else data.pos

        assert pos.size(-1) == self.matrix.size(-2), (
            'Node position matrix and transformation matrix have incompatible '
            'shape.')

        data.pos = torch.matmul(pos, self.matrix.to(pos.dtype).to(pos.device))

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.matrix.tolist())


class RandomRotate(object):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, degrees, axis=0):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if data.pos.size(-1) == 2:
            matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        return LinearTransformation(torch.tensor(matrix))(data)

    def __repr__(self):
        return '{}({}, axis={})'.format(self.__class__.__name__, self.degrees,
                                        self.axis)


# https://github.com/hxdengBerkeley/PointCNN.Pytorch/blob/master/provider.py

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data

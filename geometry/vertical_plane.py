import numpy as np
from .surface import Surface
from .utils import expand_shape

class VerticalPlane(Surface):
    """
    Represents a vertical plane y=constant.
    """
    def __init__(self, constant):
        self._constant = constant

    def intersect(self,
        ray_start: np.array,
        ray_dir: np.array,
    ):
        # Enforce shape of (num_points, point_dimension)
        assert ray_start.shape == ray_dir.shape
        if len(ray_start.shape) == 1:
            ray_start = expand_shape(ray_start)
            ray_dir = expand_shape(ray_dir)

        # Compute intersections
        num_points, point_dimension = ray_start.shape
        constant = self._constant
        intersection_lambda = (constant - ray_start[:, 1]) / ray_dir[:, 1]

        is_valid = intersection_lambda >= 0
        intersection = ray_start + np.tile(intersection_lambda, (num_points, 1)) * ray_dir
        return intersection, is_valid

    def normal_at(self,
        point: np.array,
    ):
        normals = np.zeros(point.shape)
        normals[:, 1] = 1
        return normals

    def closest_point(self,
        point: np.array,
    ):
        if len(point.shape) == 1:
            point = expand_shape(point)

        closest_point = np.copy(point)
        closest_point[:, 1] = self._constant
        return closest_point

    def is_behind(self,
        point: np.array,
    ):
        if len(point.shape) == 1:
            point = expand_shape(point)
        return point[:, 1] < self._constant

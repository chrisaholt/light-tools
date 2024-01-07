import numpy as np
from .surface import Surface
from .utils import expand_shape
from .vertical_plane import VerticalPlane

class VerticalPlaneWithLimits(VerticalPlane):
    """
    Represents a vertical plane y=constant,
    defined within x in bounds=(x_min, x_max).
    """
    def __init__(self, constant, bounds):
        super().__init__(constant)

        if (len(bounds) != 2) or (bounds[0] >= bounds[1]):
            raise ValueError("Bounds should be a tuple of shape (x_min, x_max), where x_min < x_max.")
        self._bounds = bounds

    def intersect(self,
        ray_start: np.array,
        ray_dir: np.array,
    ):
        # Intersect the plane, independent of bounds.
        intersection, is_valid = super().intersect(ray_start, ray_dir)

        # Invalidate intersections outside of bounds
        is_outside_lower = intersection[:, 1] < self._bounds[0]
        is_outside_upper = intersection[:, 1] > self._bounds[1]
        is_valid[np.logical_or(is_outside_lower, is_outside_upper)] = False

        return intersection, is_valid
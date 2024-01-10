import numpy as np

class Surface:
    """
    Describes a surface in space.
    """
    def __init__(self):
        pass

    def intersect(self,
        ray_start: np.array,
        ray_dir: np.array,
    ):
        """
        Computes the intersection of a ray with this surface.

        Args:
            ray_start (np.array (num_points, point_dimension)): array of ray origins
            ray_dir (np.array (num_points, point_dimension)): array of ray directions

        Returns:
            intersection (np.array (num_points, point_dimension))
            is_valid (np.array[bool] (num_points,)): True iff the rays intersect along the positive ray direction.
        """
        raise NotImplementedError()

    def normal_at(self,
        point: np.array,
    ):
        """
        Computes the unit vector normal to the surface at the given point.
        If the point is not on the surface, then the returned value will be
        the normal at the closest point on the surface.
        """
        raise NotImplementedError()

    def closest_point(self,
        point: np.array,
    ):
        """
        Computes the closest point on the surface to the given point.
        """
        raise NotImplementedError()
    
    def is_inside(self,
        points: np.array,
    ):
        """Returns a boolean mask of points which are inside the Surface."""
        raise NotImplementedError()
        # is_inside = np.full_like(points, False)
        # return is_inside
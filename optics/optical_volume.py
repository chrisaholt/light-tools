import numpy as np

from geometry.surface import Surface
from geometry.utils import expand_shape

class OpticalVolume:
    """
    Describes a volume of space with refractive properties.
    An example would be a lens.
    Each volume has an entry surface and an exit surface,
    where light enters through the former and leaves throut the latter.
    """
    def __init__(self,
        entry_surface: Surface,
        exit_surface: Surface,
        index_of_refraction: float,
    ):
        self._entry_surface = entry_surface
        self._exit_surface = exit_surface
        self._index_of_refraction = index_of_refraction

    def additional_optical_distance_to_point(self,
        light_source: np.array,
        point: np.array,
    ):
        """
        Computes the additional optical distance induced by this volume
        from light_source to each point.

        Args:
            light_source: Location of point source.
            point (np.array (num_points, point_dimension))
        """
        if len(point.shape) == 1:
            point = expand_shape(point)

        num_points, point_dim = point.shape
        if light_source.shape != point.shape:
            assert light_source.shape == (point_dim,), f"light_source.shape = {light_source.shape}, point.shape = {point.shape}"
            light_source_expanded = np.tile(light_source, (num_points, 1))
        else:
            light_source_expanded = light_source

        # Ray directions from light source to the points.
        ray_dirs = point - light_source_expanded
        assert ray_dirs.shape == point.shape
        distance_light_to_point = np.linalg.norm(ray_dirs, axis=-1)

        ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=-1, keepdims=True)

        # Intersections with the entry surface of rays from light to point
        intersections_light_to_entry, is_entry_valid = self._entry_surface.intersect(
            light_source_expanded,
            ray_dirs,
        )

        # Intersections with the exit surface of rays from point.
        intersections_point_to_exit, is_exit_valid = self._exit_surface.intersect(
            point,
            ray_dirs,
        )

        # Intersections with the exit surface of rays from entry to exit.
        intersections_exit_from_entry, is_entry_from_exit_valid = self._exit_surface.intersect(
            intersections_light_to_entry,
            ray_dirs,
        )

        # Compute distances within the surface for three different cases.
        #   1) Point is before the surface
        #   2) Point is within the surface
        #   3) Point is after the surface.
        distance_light_to_entry = np.linalg.norm(intersections_light_to_entry - light_source_expanded, axis=-1)

        # Point is before surface (1)
        is_before_surface = is_entry_valid and (distance_light_to_point < distance_light_to_entry)
        distance_within_surface = np.zeros(num_points)

        # Point is within surface (2)
        is_within_surface = self.is_inside(point)
        distances_to_entry = np.linalg.norm(point - intersections_light_to_entry, axis=-1)
        distance_within_surface[is_within_surface] = distances_to_entry[is_within_surface]

        # Point is after surface (3)
        is_after_surface = is_entry_valid and ~is_within_surface and ~is_before_surface
        distances_exit_to_entry = np.linalg.norm(
            intersections_exit_from_entry - intersections_light_to_entry, axis=-1
        )
        distance_within_surface[is_after_surface] = distances_exit_to_entry[is_after_surface]

        # Adjust for index of refraction.
        # Ignore distances of length 0 to allow for infinite index of refraction.
        additional_optical_distance = distance_within_surface
        additional_optical_distance[additional_optical_distance > 0] *= (self._index_of_refraction - 1)

        return additional_optical_distance

    def is_inside(self,
        points: np.array,
    ):
        """Returns a boolean mask of points which are inside the volume."""
        return np.logical_and(
            self._entry_surface.is_inside(points),
            self._exit_surface.is_inside(points),
        )
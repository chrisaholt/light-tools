import numpy as np
from .surface import Surface
from .utils import expand_shape

class SphericalSurface(Surface):
    """
    Represents a spherical surface.
    """
    def __init__(self,
        center: np.array,
        radius: float,
        is_convex: bool,
        distance_from_apex: float,
    ):
        self._center = center
        self._radius = radius
        self._is_convex = is_convex

        self._cos_max_angle_from_apex = np.clip((radius - distance_from_apex) / radius, -1.0, 1.0)

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
        # Solve this equation for lambda:
        # |start + lambda*dir - center|^2 = radius^2
        num_points, point_dimension = ray_start.shape

        # From ChatGPT on 12/16/23
        center = self._center
        radius = self._radius
        # Adjust rays relative to the center of the sphere
        oc = ray_start - center

        # Coefficients of the quadratic equation
        a = np.sum(ray_dir**2, axis=1)
        b = 2 * np.sum(oc * ray_dir, axis=1)
        c = np.sum(oc**2, axis=1) - radius**2

        # Discriminant
        discriminant = b**2 - 4 * a * c

        # Find intersections where the discriminant is non-negative
        hits = discriminant >= 0

        # Compute the two solutions of the quadratic equation
        t0 = (-b - np.sqrt(discriminant)) / (2 * a)
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)

        # Select the smallest positive t
        # t = np.where((t0 < t1) & (t0 > 0), t0, t1)
        if self._is_convex:
            t = t0
        else:
            t = t1

        # If there's no positive t, there's no intersection
        is_valid = hits & (t > 0)
        t = np.where(is_valid, t, np.nan)

        # Calculate intersection points
        # intersection_points = ray_start + t[:, np.newaxis] * ray_dir
        intersection_points = ray_start + np.tile(t, (num_points, 1)) * ray_dir

        # Check how far they points are from the apex (in angle)
        vectors_from_center = intersection_points - self._center
        vectors_from_center = vectors_from_center / np.linalg.norm(vectors_from_center, axis=-1, keepdims=True)
        cos_angle_from_center = vectors_from_center[:, 1]
        if self._is_convex:
            cos_angle_from_center = -cos_angle_from_center
        is_valid = is_valid & (cos_angle_from_center >= self._cos_max_angle_from_apex)

        return intersection_points, is_valid

# Example of waves propagating in 2D.

import cv2
import itertools
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go


def expand_shape(array: np.array):
    """Expand array's shape to be one more dimension."""
    return array.reshape((1,) + array.shape)

def indicator_func(
        t: np.array,
        thresh: np.array):
    """
    Returns nan if t < thresh and 1 if t >= thresh.
    """
    indicator_array = np.ones(t.shape )
    indicator_array[t < thresh] = np.nan
    return indicator_array

def scale_func(
    optical_distance_to_point: np.array,
):
    return 1 / (1 + optical_distance_to_point)

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
        is_within_surface = ~is_before_surface and is_entry_valid and is_exit_valid
        distances_to_entry = np.linalg.norm(point - intersections_light_to_entry, axis=-1)
        distance_within_surface[is_within_surface] = distances_to_entry[is_within_surface]

        # Point is after surface (3)
        is_after_surface = is_entry_valid and ~is_exit_valid
        distances_exit_to_entry = np.linalg.norm(
            intersections_exit_from_entry - intersections_light_to_entry, axis=-1
        )
        distance_within_surface[is_after_surface] = distances_exit_to_entry[is_after_surface]

        # Adjust for index of refraction.
        additional_optical_distance = distance_within_surface * (self._index_of_refraction - 1)

        return additional_optical_distance

def compute_optical_path_length(start, end):
    """
    Computes the optical path length between two points.
    For now, this function assumes a fixed scene geometry.
    """
    entry_surface = SphericalSurface(
        center=np.array([0, 1.0]),
        radius=0.75,
        is_convex=True,
        distance_from_apex=0.3,
    )
    exit_surface = SphericalSurface(
        center=np.array([0, 0]),
        radius=0.75,
        is_convex=False,
        distance_from_apex=0.3,
    )
    # entry_surface = VerticalPlane(0.25)
    # exit_surface = VerticalPlane(0.75)
    index_of_refraction = 1.5
    optical_volume = OpticalVolume(
        entry_surface,
        exit_surface,
        index_of_refraction,
    )

    distance = np.linalg.norm(end - start)
    additional_optical_distance = optical_volume.additional_optical_distance_to_point(
        start, end
    )
    optical_distance_to_point = distance + additional_optical_distance
    return optical_distance_to_point

class Wave2D:
    """
    Describes a 2D wave.
    """
    def __init__(self):
        pass

class PointSourceWave(Wave2D):
    """
    Describes a 2D wave traveling along the +x-axis.
    """
    def __init__(self, wavelength, phase=0, center=np.array([0, 0])):
        self._phase = phase
        self._wavelength = wavelength
        self._inverse_wavelength = 1 / wavelength
        self._center = center

    def field_at_time(self, t):
        """
        Amplitude of the wave field at normalized time t.
        Normalized time means time assuming speed of light c in a vacuum is 1.
        """
        return np.cos(2 * np.pi * self._inverse_wavelength * t + self._phase)

    def wave_at_point(self, p):
        start = self._center
        optical_distance_to_point = compute_optical_path_length(start, p)

        def wave_func(t):
            return \
                indicator_func(t, optical_distance_to_point) * \
                scale_func(optical_distance_to_point) * \
                self.field_at_time(t-optical_distance_to_point)

        return wave_func

class PlaneWave(Wave2D):
    """
    Describes a 2D plane wave.
    """
    def __init__(self,
        wavelength,
        emitter=VerticalPlane(0),
        phase=0,
        # wavelength: float,
        # emitter: VerticalPlane = VerticalPlane(0),
        # phase: float = 0,
    ):
        self._emitter = emitter
        self._phase = phase
        self._wavelength = wavelength
        self._inverse_wavelength = 1 / wavelength

    def field_at_time(self, t):
        """
        Amplitude of the wave field at normalized time t.
        """
        return np.cos(2 * np.pi * self._inverse_wavelength * t + self._phase)

    def wave_at_point(self, p):
        start = self._emitter.closest_point(p)
        optical_distance_to_point = compute_optical_path_length(start, p)

        def wave_func(t):
            return \
                indicator_func(t, optical_distance_to_point) * \
                scale_func(optical_distance_to_point) * \
                self.field_at_time(t-optical_distance_to_point)

        return wave_func

def save_images_to_video(images, video_filename):
    frame_rate = 10
    _, height, width = images.shape
    forcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        video_filename,
        forcc,
        frame_rate,
        (width, height),
        isColor=False,
        )

    # Write the images to the video writer
    for image in images:
        writer.write((255*image).astype(np.uint8))

    # Close the video writer
    writer.release()

def point_source_over_time_experiment():
    # wavelengths = [0.3, 0.3]
    # centers = np.array([
    #     [0.4, -0.2],
    #     [-0.5, -0.1],
    # ])
    wavelengths = [0.3]
    centers = np.array([
        # [0.0, 0.0],
        [0.0, -0.2],
    ])
    plane_wave_emitters = [
        VerticalPlane(-0.5),
    ]
    # waves = [
    #     PointSourceWave(wavelength, center=center)
    #     for wavelength, center in zip(wavelengths, centers)
    # ]
    waves = [
        PlaneWave(wavelength, emitter=emitter)
        for wavelength, emitter in zip(wavelengths, plane_wave_emitters)
    ]

    grid_size = 200 # 10, 200
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    points = np.array(list(itertools.product(x, y)))

    N = 200 # 200
    t = np.linspace(0, 2, N)

    amplitudes = np.stack([
        np.array([
            wave.wave_at_point(p)(t) for p in points
        ]) for wave in waves
    ])
    amplitudes[np.isnan(amplitudes)] = 0
    combined_amplitudes = np.sum(amplitudes, axis=0)
    combined_amplitudes = combined_amplitudes

    images = combined_amplitudes.reshape([len(x), len(y), -1]) ** 2
    images = images.transpose([2, 0, 1])

    # Create a video writer
    video_filename = os.path.expanduser(r"~/Desktop/waves.mp4")
    save_images_to_video(images, video_filename)

if __name__ == "__main__":
    point_source_over_time_experiment()

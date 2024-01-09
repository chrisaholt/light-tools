# Example of waves propagating in 2D.

import cv2
import itertools
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

import sys
sys.path.append("..")

from geometry.spherical_surface import SphericalSurface
from geometry.surface import Surface
from geometry.vertical_plane import VerticalPlane
from geometry.vertical_plane_with_limits import VerticalPlaneWithLimits
from geometry.utils import expand_shape
from waves.plane_wave import PlaneWave
from waves.point_source_wave import PointSourceWave

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
        # Ignore distances of length 0 to allow for infinite index of refraction.
        additional_optical_distance = distance_within_surface
        additional_optical_distance[additional_optical_distance > 0] *= (self._index_of_refraction - 1)

        return additional_optical_distance

def create_lens():
    entry_surface = SphericalSurface(
        center=np.array([0, 0.25]),
        radius=0.75,
        is_convex=True,
        distance_from_apex=0.3,
    )
    exit_surface = SphericalSurface(
        center=np.array([0, -0.75]),
        radius=1.0,
        is_convex=False,
        distance_from_apex=0.3,
    )
    index_of_refraction = 1.5
    lens = OpticalVolume(
        entry_surface,
        exit_surface,
        index_of_refraction=index_of_refraction,
    )
    return lens

def create_light_blockers():
    blocker_index_of_refraction = np.inf
    blocker_location = -0.2
    blocker_width = 0.5
    bounds_for_upper_blocker = (0.6, np.inf)
    bounds_for_lower_blocker = (-np.inf, -0.6)
    blocker_upper = OpticalVolume(
        VerticalPlaneWithLimits(
            blocker_location,
            bounds=bounds_for_upper_blocker,
        ),
        VerticalPlaneWithLimits(
            blocker_location + blocker_width,
            bounds=bounds_for_upper_blocker,
        ),
        index_of_refraction=blocker_index_of_refraction,
    )
    blocker_lower = OpticalVolume(
        VerticalPlaneWithLimits(
            blocker_location,
            bounds=bounds_for_lower_blocker,
        ),
        VerticalPlaneWithLimits(
            blocker_location + blocker_width,
            bounds=bounds_for_lower_blocker,
        ),
        index_of_refraction=blocker_index_of_refraction,
    )
    return (blocker_upper, blocker_lower)

def compute_optical_path_length(start, end):
    """
    Computes the optical path length between two points.
    For now, this function assumes a fixed scene geometry.
    """

    lens = create_lens()
    light_blockers = create_light_blockers()

    distance = np.linalg.norm(end - start)
    additional_optical_distance = lens.additional_optical_distance_to_point(
        start, end
    )
    for blocker in light_blockers:
        additional_optical_distance += blocker.additional_optical_distance_to_point(
            start, end
        )

    optical_distance_to_point = distance + additional_optical_distance
    return optical_distance_to_point

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
        VerticalPlane(-1.0),
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

    time_resolution = 200 # 200
    total_time = 3 # 2
    t = np.linspace(0, total_time, time_resolution)

    all_waves = [
        np.array([
            wave.wave_at_point(p, compute_optical_path_length)(t) for p in points
        ]) for wave in waves
    ]

    # Diffractive amplitudes
    diffraction_points = np.array([
        [-0.59, 0.3] #[-0.6, -0.1], [0.0, 0.0]
    ])
    for diffraction_point in diffraction_points:
        start = waves[0]._emitter.closest_point(diffraction_point)
        optical_distance_to_diffractive_point = compute_optical_path_length(start, diffraction_point)
        diffractive_wave = PointSourceWave(wavelengths[0], center=diffraction_point)

        def compute_optical_path_for_diffractive_point(start, end):
            # First verify that propagation is in the +y-direction.
            if end[1] < start[1]:
                return np.nan
            return optical_distance_to_diffractive_point + compute_optical_path_length(start, end)
        
        all_waves.append(
            np.array([
                diffractive_wave.wave_at_point(
                    p,
                    compute_optical_path_for_diffractive_point,
                )(t) for p in points
            ]))
        
    amplitudes = np.stack(all_waves)
    amplitudes[np.isnan(amplitudes)] = 0
    combined_amplitudes = np.sum(amplitudes, axis=0)
    combined_amplitudes = combined_amplitudes

    images = combined_amplitudes.reshape([len(x), len(y), -1]) ** 2
    images = images.transpose([2, 0, 1])

    # Create a video writer
    video_filename = os.path.expanduser(r"~/OneDrive/Desktop/waves.mp4")
    save_images_to_video(images, video_filename)

if __name__ == "__main__":
    point_source_over_time_experiment()

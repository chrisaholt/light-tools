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
from geometry.vertical_plane import VerticalPlane
from geometry.vertical_plane_with_limits import VerticalPlaneWithLimits
from optics.optical_volume import OpticalVolume
from waves.plane_wave import PlaneWave
from waves.point_source_wave import PointSourceWave

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

def create_diffractive_wave_amplitudes_at_points(emitter, wavelength, points, time):
    diffraction_points = np.array([
        [-0.59, 0.3],
        [0.59, 0.3],
    ])
    diffraction_amplitudes = []
    for diffraction_point in diffraction_points:
        start = emitter.closest_point(diffraction_point)
        optical_distance_to_diffractive_point = compute_optical_path_length(start, diffraction_point)
        diffractive_wave = PointSourceWave(wavelength, center=diffraction_point)
        diffractive_scale = 0.5

        def compute_optical_path_for_diffractive_point(start, end):
            # First verify that propagation is in the +y-direction.
            if end[1] < start[1]:
                return np.nan
            return optical_distance_to_diffractive_point + compute_optical_path_length(start, end)
        
        diffraction_amplitudes.append(
            diffractive_scale * np.array([
                diffractive_wave.wave_at_point(
                    p,
                    compute_optical_path_for_diffractive_point,
                )(time) for p in points
            ]))
    return diffraction_amplitudes

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
    wavelengths = [0.1]
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

    all_amplitudes = [
        np.array([
            wave.wave_at_point(p, compute_optical_path_length)(t) for p in points
        ]) for wave in waves
    ]

    # Diffractive amplitudes
    diffraction_amplitudes = create_diffractive_wave_amplitudes_at_points(
        plane_wave_emitters[0], wavelengths[0], points, t
    )
        
    amplitudes = np.stack(all_amplitudes + diffraction_amplitudes)
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

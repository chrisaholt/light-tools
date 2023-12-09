# Example of waves propagating in 2D.

import cv2
import itertools
import numpy as np
import plotly.express as px
import plotly.graph_objects as go



def indicator_func(
        t: np.array,
        thresh: np.array):
    """
    Returns nan if t < thresh and 1 if t >= thresh.
    """
    indicator_array = np.ones(t.shape )
    indicator_array[t < thresh] = np.nan
    return indicator_array

class Wave2D:
    """
    Describes a 2D wave.
    """
    def __init__(self):
        pass

class PlanarWave(Wave2D):
    """
    Describes a 2D wave traveling along the +x-axis.
    """
    def __init__(self, wavelength, phase=0):
        self._phase = phase
        self._wavelength = wavelength
        self._inverse_wavelength = 1 / wavelength
    
    def field_at_time(self, t):
        """
        Amplitude of the wave field at normalized time t.
        Normalized time means time assuming speed of light c in a vacuum is 1.
        """
        return np.cos(2 * np.pi * self._inverse_wavelength * t + self._phase)

    def wave_at_point(self, p):
        start = np.array([0, 0])
        distance = np.linalg.norm(p - start)
        optical_distance_to_point = distance

        # Define an (arbitrary) optical surface at y=interface_y,
        # with specificied index_of_refraction. Then compute the 
        # optical path length including that interface.
        interface_y = 0.25
        index_of_refraction = 1.5
        if p[1] >= interface_y:
            p_intersection_lambda = (interface_y - start[1]) / p[1]
            p_intersection = start + p_intersection_lambda * p
            optical_distance_to_point = \
                np.linalg.norm(p_intersection - start) + \
                np.linalg.norm(p - p_intersection) * index_of_refraction

        def wave_func(t):
            return \
                indicator_func(t, optical_distance_to_point) * \
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
    wavelength = 0.3 #700e-9
    wave = PlanarWave(wavelength)
    
    grid_size = 200 # 10, 200
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    points = np.array(list(itertools.product(x, y)))

    N = 100 # 100
    t = np.linspace(0, 2, N)

    amplitudes = np.array([
        wave.wave_at_point(p)(t) for p in points
    ])
    images = amplitudes.reshape([len(x), len(y), -1]) ** 2
    images = images.transpose([2, 0, 1])
    images[np.isnan(images)] = 0

    # Create a video writer
    video_filename = r"C:\Users\chris\OneDrive\Desktop\waves.mp4"
    save_images_to_video(images, video_filename)

if __name__ == "__main__":
    point_source_over_time_experiment()

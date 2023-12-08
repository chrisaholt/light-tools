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
        def wave_func(t):
            return indicator_func(t, distance) * self.field_at_time(t-distance)
        return wave_func

def light_path_intensity(wave, start, end):
    distance = np.linalg.norm(end - start)
    
    # Compute intensities.
    N = 100
    t = np.linspace(0, distance, N)
    intensity = wave.field_at_time(t)**2

    # TEMPORARY TO CHECK INDEX OF REFRACTION CHANGE AT X==1.
    eta = 2.5
    # All xy points between start and end.
    xy = np.tile(start, [N, 1]) + np.tile(t/distance, [2,1]).T * np.tile(end, [N, 1])
    refracted_indices = np.where(xy[:, 0] > 1)[0]
    if len(refracted_indices) > 0:
        first_refracted_idx = refracted_indices[0]
        t[first_refracted_idx:] = t[first_refracted_idx] + (t[first_refracted_idx:] - t[first_refracted_idx])/eta


    # intensity = wave.field_at_time(t)**2

    
    # All xy points between start and end.
    xy = np.tile(start, [N, 1]) + np.tile(t/distance, [2,1]).T * np.tile(end, [N, 1])

    return intensity, xy

def planar_wave_experiment():
    wavelength = 0.3 #700e-9
    wave = PlanarWave(wavelength)
    
    N = 100
    start = np.array([0, 0])
    theta = np.linspace(-np.pi/4, np.pi/4, N)
    radius = 2
    ends = radius * np.vstack([
        np.cos(theta),
        np.sin(theta)
    ]).T

    plots = []
    for end in ends:
        intensity, xy = light_path_intensity(wave, start, end)

        plots += [go.Scatter(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="markers",
            marker=dict(
                # color=[f"hsva(255,{int(val*100)},{int(val*100)},1.0)" for val in intensity],
                color=[f"hsva(255,100,100,{val:0.2f})" for val in intensity],
            ),
            showlegend=False,
        )]

    fig = go.Figure(data=plots)
    fig.show()

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

    plots = []

    amplitudes = np.array([
        wave.wave_at_point(p)(t) for p in points
    ])
    images = amplitudes.reshape([len(x), len(y), -1]) ** 2
    images = images.transpose([2, 0, 1])
    images[np.isnan(images)] = 0

    # current_image = images[N//2, :, :] ** 2
    # fig = px.imshow(current_image)
    # fig.show()

    # Create a video writer
    video_filename = r"C:\Users\chris\OneDrive\Desktop\waves.mp4"
    # video_filename = r"C:\Users\chris\OneDrive\Desktop\waves.avi"
    save_images_to_video(images, video_filename)

    should_plot = False
    if should_plot:
        for amp in amplitudes:
            plots += [go.Scatter(
                x=t,
                y=amp,
                mode="lines+markers",
            )]
        fig = go.Figure(data=plots)
        fig.show()

if __name__ == "__main__":
    # planar_wave_experiment()

    point_source_over_time_experiment()

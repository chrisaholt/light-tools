# Example of waves propagating in 2D.

import numpy as np
import plotly.graph_objects as go

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
    
    def field(self, t):
        """
        Amplitude of the wave field at normalized time t.
        Normalized time means time assuming speed of light c in a vacuum is 1.
        """
        return np.cos(2 * np.pi * self._inverse_wavelength * t + self._phase)

def light_path_intensity(wave, start, end):
    distance = np.linalg.norm(end - start)
    
    # Compute intensities.
    N = 100
    t = np.linspace(0, distance, N)
    intensity = wave.field(t)**2

    # TEMPORARY TO CHECK INDEX OF REFRACTION CHANGE AT X==1.
    eta = 2.5
    # All xy points between start and end.
    xy = np.tile(start, [N, 1]) + np.tile(t/distance, [2,1]).T * np.tile(end, [N, 1])
    refracted_indices = np.where(xy[:, 0] > 1)[0]
    if len(refracted_indices) > 0:
        first_refracted_idx = refracted_indices[0]
        t[first_refracted_idx:] = t[first_refracted_idx] + (t[first_refracted_idx:] - t[first_refracted_idx])/eta


    # intensity = wave.field(t)**2

    
    # All xy points between start and end.
    xy = np.tile(start, [N, 1]) + np.tile(t/distance, [2,1]).T * np.tile(end, [N, 1])

    return intensity, xy

if __name__ == "__main__":
    wavelength = 0.3 #700e-9
    wave = PlanarWave(wavelength)
    
    N = 100
    start = np.array([0,0])
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
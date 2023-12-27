import numpy as np

class Wave:
    """
    Describes a light wave.
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

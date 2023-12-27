import numpy as np

from .utils import (
    indicator_func,
    scale_func,
)
from .wave import Wave

class PointSourceWave(Wave):
    """
    Describes a 2D wave traveling along the +x-axis.
    """
    def __init__(self, wavelength, phase=0, center=np.array([0, 0])):
        super().__init__(wavelength, phase)
        self._center = center

    def wave_at_point(self, p, compute_optical_path_length):
        start = self._center
        optical_distance_to_point = compute_optical_path_length(start, p)

        def wave_func(t):
            return \
                indicator_func(t, optical_distance_to_point) * \
                scale_func(optical_distance_to_point) * \
                self.field_at_time(t-optical_distance_to_point)

        return wave_func

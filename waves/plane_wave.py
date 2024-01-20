import numpy as np

from geometry.vertical_plane import VerticalPlane
from .utils import (
    indicator_func,
    scale_func,
)
from .wave import Wave

class PlaneWave(Wave):
    """
    Describes a 2D plane wave.
    """
    def __init__(self,
        wavelength: float,
        emitter: VerticalPlane = VerticalPlane(0, 1),
        phase: float = 0,
    ):
        super().__init__(wavelength, phase)
        self._emitter = emitter

    def wave_at_point(self, p, compute_optical_path_length):
        start = self._emitter.closest_point(p)
        optical_distance_to_point = compute_optical_path_length(start, p)
        optical_distance_to_point[self._emitter.is_behind(p)] = np.nan

        def wave_func(t):
            return \
                indicator_func(t, optical_distance_to_point) * \
                scale_func(optical_distance_to_point) * \
                self.field_at_time(t-optical_distance_to_point)

        return wave_func

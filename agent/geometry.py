from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CameraModel:
    hfov_deg: float
    vfov_deg: float
    height_cm: float
    tilt_down_deg: float = 0.0

    def bearing_deg(self, x_norm: float) -> float:
        u = (x_norm / 1000.0 - 0.5) * 2.0
        return math.degrees(math.atan(u * math.tan(math.radians(self.hfov_deg / 2))))

    def ground_distance_cm(self, y_norm: float) -> float | None:
        v = (y_norm / 1000.0 - 0.5) * 2.0
        below_axis = math.atan(v * math.tan(math.radians(self.vfov_deg / 2)))
        depression = math.radians(self.tilt_down_deg) + below_axis
        if depression <= math.radians(1.0):
            return None
        return self.height_cm / math.tan(depression)

    def point_to_motion(self, y_norm: float, x_norm: float) -> tuple[float, float | None]:
        turn = self.bearing_deg(x_norm)
        straight_ahead = self.ground_distance_cm(y_norm)
        if straight_ahead is None:
            return turn, None
        return turn, straight_ahead / max(math.cos(math.radians(turn)), 0.2)


    def project(self, forward_cm: float, right_cm: float) -> tuple[float, float] | None:
        if forward_cm <= 0.5:
            return None
        u = (right_cm / forward_cm) / math.tan(math.radians(self.hfov_deg / 2))
        below_axis = math.atan(self.height_cm / forward_cm) - math.radians(self.tilt_down_deg)
        v = math.tan(below_axis) / math.tan(math.radians(self.vfov_deg / 2))
        if abs(u) > 0.97 or abs(v) > 0.97:
            return None
        return (v / 2 + 0.5) * 1000, (u / 2 + 0.5) * 1000


def vfov_from_hfov(hfov_deg: float, width: int, height: int) -> float:
    return math.degrees(2 * math.atan(math.tan(math.radians(hfov_deg) / 2) * height / width))

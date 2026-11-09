"""Per-phase timing for smooth Cartesian assembly motion."""
import math
def motion_steps(config, phase, nominal_seconds, distance_m, angle_rad, duration_variation):
    limits = {**config["default"], **config["phases"].get(phase, {})}
    seconds = max(
        nominal_seconds * limits["duration_scale"] * duration_variation,
        1.875 * distance_m / (limits["peak_linear_mm_s"] / 1000.0),
        math.sqrt((10.0 / math.sqrt(3.0)) * distance_m /
                  (limits.get("peak_linear_acceleration_mm_s2", float("inf")) / 1000.0)),
        1.875 * abs(angle_rad) / limits["peak_angular_rad_s"],
        1.0 / 60.0,
    )
    return math.ceil(seconds * 60)

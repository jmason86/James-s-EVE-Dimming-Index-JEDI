from make_light_curve import make_light_curve, normalized_irradiance_in_percent_units
from determine_dimming_duration import determine_dimming_duration
import numpy as np


def test_duration():
    light_curve = make_light_curve()
    assert no_points_below_zero_returns_nan(light_curve)

    light_curve = normalized_irradiance_in_percent_units(light_curve)
    duration_seconds, duration_start_time, duration_end_time = determine_dimming_duration(light_curve.copy())
    print(duration_seconds)
    print(duration_start_time)
    print(duration_end_time)


def no_points_below_zero_returns_nan(light_curve):
    duration_seconds, duration_start_time, duration_end_time = determine_dimming_duration(light_curve.copy())
    if (duration_seconds is np.nan) and (duration_start_time is np.nan) and (duration_end_time is np.nan):
        return True

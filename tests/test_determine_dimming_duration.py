from make_light_curve import make_light_curve, normalized_irradiance_in_percent_units
from determine_dimming_duration import determine_dimming_duration
import numpy as np
import pandas as pd


def test_duration():
    light_curve = make_light_curve()
    assert no_points_below_zero_returns_nan(light_curve)

    light_curve = normalized_irradiance_in_percent_units(light_curve)
    assert nominal_case_returns_expected_values(light_curve)
    assert early_cutoff_works(light_curve)


def no_points_below_zero_returns_nan(light_curve):
    duration_seconds, duration_start_time, duration_end_time = determine_dimming_duration(light_curve.copy())
    if (duration_seconds is np.nan) and (duration_start_time is np.nan) and (duration_end_time is np.nan):
        return True


def nominal_case_returns_expected_values(light_curve):
    duration_seconds, duration_start_time, duration_end_time = determine_dimming_duration(light_curve.copy(),
                                                                                          smooth_points=50)
    if (duration_seconds == 9780) and \
            (duration_start_time == pd.Timestamp('2010-08-07 18:52:11')) and \
            (duration_end_time == pd.Timestamp('2010-08-07 21:35:11')):
        return True


def early_cutoff_works(light_curve):
    time_early = pd.Timestamp('2010-08-07 19:40:00')
    duration_seconds, duration_start_time, duration_end_time = determine_dimming_duration(light_curve.copy(),
                                                                                          smooth_points=50,
                                                                                          earliest_allowed_time=time_early)
    if (duration_seconds is np.nan) and (duration_start_time is np.nan) and (duration_end_time is np.nan):
        return True

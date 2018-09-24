from make_light_curve import make_light_curve, normalized_irradiance_in_percent_units
from determine_dimming_depth import determine_dimming_depth
import numpy as np
import pandas as pd


def test_depth():
    light_curve = make_light_curve()
    assert no_points_below_zero_returns_nan(light_curve)

    light_curve = normalized_irradiance_in_percent_units(light_curve)
    assert nominal_case_returns_expected_values(light_curve)
    assert early_cutoff_works(light_curve)
    assert late_cutoff_works(light_curve)


def no_points_below_zero_returns_nan(light_curve):
    depth, time = determine_dimming_depth(light_curve.copy())
    if (depth is np.nan) and (time is np.nan):
        return True


def nominal_case_returns_expected_values(light_curve):
    depth, time = determine_dimming_depth(light_curve.copy(), smooth_points=50)
    if ('{0:.2f}'.format(depth) == '1.66') and (time == pd.Timestamp('2010-08-07 19:36:11')):
        return True


def early_cutoff_works(light_curve):
    time_early = pd.Timestamp('2010-08-07 19:40:00')
    depth, time = determine_dimming_depth(light_curve.copy(),
                                          smooth_points=50,
                                          earliest_allowed_time=time_early)
    if ('{0:.2f}'.format(depth) == '1.79') and (time == pd.Timestamp('2010-08-07 20:10:11')):
        return True


def late_cutoff_works(light_curve):
    time_late = pd.Timestamp('2010-08-07 19:40:00')
    depth, time = determine_dimming_depth(light_curve.copy(),
                                          smooth_points=50,
                                          latest_allowed_time=time_late)
    if ('{0:.2f}'.format(depth) == '0.84') and (time == pd.Timestamp('2010-08-07 19:15:11')):
        return True

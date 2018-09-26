from make_light_curve import make_light_curve, normalized_irradiance_in_percent_units
from determine_dimming_depth import determine_dimming_depth
import numpy as np
from numpy.testing import assert_approx_equal
import pandas as pd


def test_depth():
    light_curve = make_light_curve()
    assert no_points_below_zero_returns_nan(light_curve)

    light_curve = normalized_irradiance_in_percent_units(light_curve)
    nominal_case_returns_expected_values(light_curve)
    early_cutoff_works(light_curve)
    late_cutoff_works(light_curve)


def no_points_below_zero_returns_nan(light_curve):
    depth, time = determine_dimming_depth(light_curve.copy())
    if (depth is np.nan) and (time is np.nan):
        return True


def nominal_case_returns_expected_values(light_curve):
    depth, time = determine_dimming_depth(light_curve.copy(), smooth_points=50)
    assert_approx_equal(depth, 1.66, significant=3)
    assert time == pd.Timestamp('2010-08-07 19:36:11')


def early_cutoff_works(light_curve):
    time_early = pd.Timestamp('2010-08-07 19:40:00')
    depth, time = determine_dimming_depth(light_curve.copy(),
                                          smooth_points=50,
                                          earliest_allowed_time=time_early)
    assert_approx_equal(depth, 1.79, significant=3)
    assert time == pd.Timestamp('2010-08-07 20:10:11')


def late_cutoff_works(light_curve):
    time_late = pd.Timestamp('2010-08-07 19:40:00')
    depth, time = determine_dimming_depth(light_curve.copy(),
                                          smooth_points=50,
                                          latest_allowed_time=time_late)
    assert_approx_equal(depth, 0.841, significant=3)
    assert time == pd.Timestamp('2010-08-07 19:15:11')

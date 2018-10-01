from make_light_curve import make_light_curve, normalized_irradiance_in_percent_units
from determine_dimming_slope import determine_dimming_slope
import numpy as np
from numpy.testing import assert_approx_equal
import pandas as pd


class TestSlope():
    def test_slope(self):
        light_curve = make_light_curve()
        self.light_curve = normalized_irradiance_in_percent_units(light_curve)
        self.depth_time = pd.Timestamp('2010-08-07 19:36:11')

        self.nominal_case_returns_expected_values()
        self.early_cutoff_returns_expected_values()
        self.late_cutoff_returns_expected_values()

    def nominal_case_returns_expected_values(self):
        slope_min, slope_max, slope_mean = determine_dimming_slope(self.light_curve,
                                                                   latest_allowed_time=self.depth_time,
                                                                   smooth_points=50)
        assert_approx_equal(slope_min, 2.70e-5, significant=3)
        assert_approx_equal(slope_max, 1.39e-3, significant=3)
        assert_approx_equal(slope_mean, 5.53e-4, significant=3)

    def early_cutoff_returns_expected_values(self):
        time_early = pd.Timestamp('2010-08-07 19:00:00')
        slope_min, slope_max, slope_mean = determine_dimming_slope(self.light_curve,
                                                                   earliest_allowed_time=time_early,
                                                                   latest_allowed_time=self.depth_time,
                                                                   smooth_points=50)
        assert_approx_equal(slope_min, 4.29e-4, significant=3)
        assert_approx_equal(slope_max, 6.34e-4, significant=3)
        assert_approx_equal(slope_mean, 5.64e-4, significant=3)

    def late_cutoff_returns_expected_values(self):
        time_late = pd.Timestamp('2010-08-07 21:00:00')
        slope_min, slope_max, slope_mean = determine_dimming_slope(self.light_curve,
                                                                   latest_allowed_time=time_late,
                                                                   smooth_points=50)
        assert_approx_equal(slope_min, -1.92e-4, significant=3)
        assert_approx_equal(slope_max, 5.48e-4, significant=3)
        assert_approx_equal(slope_mean, 1.91e-4, significant=3)

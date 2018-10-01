from make_light_curve import make_light_curve, make_light_curve_284nm, normalized_irradiance_in_percent_units
from light_curve_peak_match_subtract import light_curve_peak_match_subtract
from numpy.testing import assert_approx_equal
import pandas as pd


class TestPeakMatchSubtract:

    def test_peak_match_subtract(self):
        self.subtract_from = normalized_irradiance_in_percent_units(make_light_curve())
        self.subtract_with = normalized_irradiance_in_percent_units(make_light_curve_284nm())
        self.flare_peak_time = pd.Timestamp('2010-08-07 18:24:00')

        self.nominal_case_returns_expected_values()
        self.different_peak_estimate_returns_different_but_similar_values()

    def nominal_case_returns_expected_values(self):
        corrected, seconds_shift, scale_factor = light_curve_peak_match_subtract(self.subtract_from,
                                                                                 self.subtract_with,
                                                                                 self.flare_peak_time)
        assert isinstance(corrected, pd.DataFrame)
        assert len(corrected) == 300
        assert seconds_shift == 180.0
        assert_approx_equal(scale_factor, 0.070, significant=3)

    def different_peak_estimate_returns_different_but_similar_values(self):
        time_other_peak = pd.Timestamp('2010-08-07 20:00:00')
        corrected, seconds_shift, scale_factor = light_curve_peak_match_subtract(self.subtract_from,
                                                                                 self.subtract_with,
                                                                                 estimated_time_of_peak=time_other_peak)
        assert isinstance(corrected, pd.DataFrame)
        assert len(corrected) == 300
        assert seconds_shift == 360.0
        assert_approx_equal(scale_factor, 0.0687, significant=3)

from make_light_curve import make_preflare_light_curve, normalized_irradiance_in_percent_units
from determine_preflare_irradiance import determine_preflare_irradiance
import pandas as pd
import numpy as np
from numpy.testing import assert_approx_equal


class TestBaselineDetermination:

    def test_baseline_determination(self):
        self.light_curve = make_preflare_light_curve()
        self.flare_peak_time = pd.Timestamp('2010-08-07 18:24:00')

        self.nominal_case_returns_expected_values()
        self.low_median_diff_threshold_fails()
        self.low_std_threshold_fails()
        assert self.too_early_time_of_peak_start_fails()

    def nominal_case_returns_expected_values(self):
        preflare_irradiance = determine_preflare_irradiance(self.light_curve.copy(),
                                                            estimated_time_of_peak_start=self.flare_peak_time)
        assert_approx_equal(preflare_irradiance, 5.85e-5, significant=3)

    def low_median_diff_threshold_fails(self):
        preflare_irradiance = determine_preflare_irradiance(self.light_curve.copy(),
                                                            estimated_time_of_peak_start=self.flare_peak_time,
                                                            max_median_diff_threshold=0.5)
        assert preflare_irradiance is np.nan

    def low_std_threshold_fails(self):
        preflare_irradiance = determine_preflare_irradiance(self.light_curve.copy(),
                                                            estimated_time_of_peak_start=self.flare_peak_time,
                                                            std_threshold=0.39)
        assert preflare_irradiance is np.nan

    def too_early_time_of_peak_start_fails(self):
        too_early_time = pd.Timestamp('2010-08-07 09:00:00')
        try:
            determine_preflare_irradiance(self.light_curve.copy(), estimated_time_of_peak_start=too_early_time)
        except ValueError as error:
            if len(error.args) == 2:
                if error.args[1] == 'too_early_time_of_peak_start':
                    return True
        else:
            return False

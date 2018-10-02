import jedi_config
import pandas as pd
from astropy.time import Time
import astropy.units as u

jedi_config.init()  # Takes about 60 seconds


def make_light_curve():
    light_curve = pd.DataFrame(jedi_config.eve_lines.loc['2010-08-07 17:00:00':'2010-08-07 22:00:00', '17.1'])
    light_curve.columns = ['irradiance']
    return light_curve


def make_light_curve_284nm():
    light_curve = pd.DataFrame(jedi_config.eve_lines.loc['2010-08-07 17:00:00':'2010-08-07 22:00:00', '28.4'])
    light_curve.columns = ['irradiance']
    return light_curve


def normalized_irradiance_in_percent_units(light_curve):
    return pd.DataFrame((light_curve['irradiance'] - light_curve['irradiance'].iloc[0]) / light_curve['irradiance'].iloc[0] * 100.0)


def make_preflare_light_curve():
    flare_peak_time = '2010-08-07 18:24:00'
    preflare_start_time = (Time(flare_peak_time, precision=0) - (jedi_config.threshold_time_prior_flare_minutes * u.minute)).iso
    light_curve = pd.DataFrame(jedi_config.eve_lines.loc[preflare_start_time:flare_peak_time, '17.1'])
    light_curve.columns = ['irradiance']
    return light_curve

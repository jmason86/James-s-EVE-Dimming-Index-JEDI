import jedi_config
import pandas as pd

jedi_config.init()  # Takes about 60 seconds


def make_light_curve():
    light_curve = pd.DataFrame(jedi_config.eve_lines.loc['2010-08-07 17:00:00':'2010-08-07 22:00:00', '17.1'])
    light_curve.columns = ['irradiance']
    return light_curve


def normalized_irradiance_in_percent_units(light_curve):
    return (light_curve['irradiance'] - light_curve['irradiance'].iloc[0]) / light_curve['irradiance'].iloc[0] * 100.0

import jedi_config
from determine_dimming_depth import determine_dimming_depth
import pandas as pd

jedi_config.init()  # Configures and loads everything - takes about 60 seconds


def test_depth():
    light_curve = make_light_curve()

    depth, time = determine_dimming_depth(light_curve, smooth_points=50)
    assert '{0:.2f}'.format(depth) == '1.66'
    assert time == pd.Timestamp('2010-08-07 19:36:11')


def make_light_curve():
    light_curve = pd.DataFrame(jedi_config.eve_lines.loc['2010-08-07 17:00:00':'2010-08-07 22:00:00', '17.1'])
    light_curve.columns = ['irradiance']
    light_curve['irradiance'] = (light_curve['irradiance'] - light_curve['irradiance'].iloc[0]) / light_curve['irradiance'].iloc[0] * 100
    return light_curve


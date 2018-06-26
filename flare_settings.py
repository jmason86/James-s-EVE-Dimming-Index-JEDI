import os
from jpm_logger import JpmLogger
from astropy.time import Time
from scipy.io.idl import readsav
import numpy as np
import pandas as pd


eve_data_path = ''
goes_data_path = ''
output_path = ''
logger_filename = ''  # 'generate_jedi_catalog'

threshold_time_prior_flare_minutes = 480.0
dimming_window_relative_to_flare_minutes_left = -1.0
dimming_window_relative_to_flare_minutes_right = 1440.0
threshold_minimum_dimming_window_minutes = 120.0


n_jobs = 1
verbose = True

eve_lines = None
goes_flare_events = None
logger = None
csv_filename = None
preflare_hdf_filename = None

jedi_columns = ['Event #',
                'GOES Flare Start Time',
                'GOES Flare Peak Time',
                'GOES Flare Class',
                'Pre-Flare Start Time',
                'Pre-Flare End Time',
                'Flare Interrupt',
                'Pre-Flare Irradiance [W/m2]',
                'Slope Start Time',
                'Slope End Time',
                'Slope Min [%/s]',
                'Slope Max [%/s]',
                'Slope Mean [%/s]',
                'Slope Uncertainty [%/s]',
                'Depth Time',
                'Depth [%]',
                'Depth Uncertainty [%]',
                'Duration Start Time',
                'Duration End Time',
                'Duration [s]',
                'Fitting Gamma',
                'Fitting Score',
                'Slope Start Time',
                'Slope End Time',
                'Slope Min [%/s]',
                'Slope Max [%/s]',
                'Slope Mean [%/s]',
                'Slope Uncertainty [%/s]',
                'Depth Time',
                'Depth [%]',
                'Depth Uncertainty [%]',
                'Duration Start Time',
                'Duration End Time',
                'Duration [s]',
                'Correction Time Shift [s]',
                'Correction Scale Factor',
                'Fitting Gamma',
                'Fitting Score']


def init():

    global eve_lines, goes_flare_events, logger, csv_filename, preflare_hdf_filename

    csv_filename = output_path + 'jedi_{0}.csv'.format(Time.now().iso)
    preflare_hdf_filename = os.path.join(output_path, 'preflare_df.hdf5')

    logger = JpmLogger(filename=logger_filename, path=output_path, console=False)

    eve_readsav = readsav(eve_data_path)

    # Load up the actual irradiance data into a pandas DataFrame
    # TODO: Replace this shortcut method with the method I'm building into sunpy
    irradiance = eve_readsav['irradiance'].byteswap().newbyteorder()  # pandas doesn't like big endian
    irradiance[irradiance == -1] = np.nan
    wavelengths = eve_readsav['wavelength']
    wavelengths_str = []
    [wavelengths_str.append('{0:1.1f}'.format(wavelength)) for wavelength in wavelengths]
    eve_lines = pd.DataFrame(irradiance, columns=wavelengths_str)
    eve_lines.index = pd.to_datetime(eve_readsav.iso.astype(str))
    eve_lines.sort_index(inplace=True)
    eve_lines = eve_lines.drop_duplicates()

    # Get GOES flare events above C1 within date range corresponding to EVE data
    # flares = get_goes_flare_events(eve_lines.index[0], eve_lines.index[-1], verbose=verbose)  # TODO: The method in sunpy needs fixing, issue 2434

    # Load GOES events from IDL saveset instead of directly through sunpy
    goes_flare_events = readsav(goes_data_path)
    goes_flare_events['class'] = goes_flare_events['class'].astype(str)
    goes_flare_events['event_peak_time_human'] = goes_flare_events['event_peak_time_human'].astype(str)
    goes_flare_events['event_start_time_human'] = goes_flare_events['event_start_time_human'].astype(str)
    goes_flare_events['peak_time'] = Time(goes_flare_events['event_peak_time_jd'], format='jd', scale='utc')
    goes_flare_events['start_time'] = Time(goes_flare_events['event_start_time_jd'], format='jd', scale='utc')
    #t = pd.to_datetime(goes_flare_events['event_start_time_jd'], unit='D', origin='julian')






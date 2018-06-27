from importlib import reload
import os
import numpy as np
import pandas as pd
import itertools
from astropy.time import Time
import eve_jedi
import time
import flare_settings
import matplotlib as mpl
mpl.use('agg')
from collections import OrderedDict

__author__ = 'James Paul Mason'
__contact__ = 'jmason86@gmail.com'

# Main

flare_settings.eve_data_path = '/Users/rattie/Data/James/eve_lines_2010121-2014146 MEGS-A Mission Bare Bones.sav'
flare_settings.goes_data_path = '/Users/rattie/Data/James/GoesEventsC1MinMegsAEra.sav'
flare_settings.output_path = '/Users/rattie/Data/James/output/'
flare_settings.csv_filename = flare_settings.output_path + 'jedi_{0}.csv'.format(Time.now().iso)
flare_settings.verbose = True
flare_settings.logger_filename = 'generate_jedi_catalog'

flare_settings.init()
flare_settings.make_jedi_df()

wavelengths = flare_settings.eve_lines.columns

if flare_settings.verbose:
    flare_settings.logger.info('Created JEDI row definition.')

# Load the pre-flare irradiance data
preflare_df = pd.read_hdf(flare_settings.preflare_hdf_filename)

# Prepare a hold-over pre-flare irradiance value,
# which will normally have one element for each of the 39 emission lines
preflare_irradiance = np.nan
# precalculate the series of minutes_since_last_flare which otherwise is calculated at each loop index
peak_time = flare_settings.goes_flare_events['peak_time']
all_minutes_since_last_flare = (peak_time[1:] - peak_time[0:-1]).sec / 60.0
tmask = all_minutes_since_last_flare > flare_settings.threshold_time_prior_flare_minutes
# flare_index_range = range(0, len(all_minutes_since_last_flare))
# TODO: parallelization will work if we create series of ranges where tmask[current_range[0]] = True
# <-> minutes_since_last_flare > flare_settings.threshold_time_prior_flare_minutes

# Map all flare indices to their pre-flare event index
preflare_map_idx = eve_jedi.map_true_indices2(tmask, range(0, tmask.size))

flare_index_range = range(1, 4)

verbose = flare_settings.verbose
# Start loop through all flares


for flare_index in flare_index_range:

    # Flare index relative to the preflare index map
    flare_index_rel = flare_index - 1
    print('Processing flare index %d' % flare_index)
    # Profile 0 starts
    loop_time_start = time.time()

    print('Running on event {0}'.format(flare_index))

    # If haven't already done all pre-parameterization processing
    processed_jedi_non_params_filename = flare_settings.output_path + 'Processed Pre-Parameterization Data/Event {0} Pre-Parameterization.h5'.format(flare_index)
    processed_lines_filename = flare_settings.output_path + 'Processed Lines Data/Event {0} Lines.h5'.format(flare_index)

    # Fill the GOES flare information into the dataframe at all wavelength

    flare_settings.jedi_df.at[flare_index, 'GOES Flare Start Time'] = flare_settings.goes_flare_events['start_time'][flare_index].iso
    flare_settings.jedi_df.at[flare_index, 'GOES Flare Peak Time'] = flare_settings.goes_flare_events['peak_time'][flare_index].iso
    flare_settings.jedi_df.at[flare_index, 'GOES Flare Class'] = flare_settings.goes_flare_events['class'][flare_index]

    if verbose:
        flare_settings.logger.info("Event {0} GOES flare details stored to JEDI row.".format(flare_index))

    if not os.path.isfile(processed_lines_filename) or not os.path.isfile(processed_jedi_non_params_filename):

        # Map current event to preflare data at all wavelengths.
        flare_settings.jedi_df.at[flare_index, 'Pre-Flare Start Time'] = preflare_df['window start'][preflare_map_idx[flare_index_rel]]
        flare_settings.jedi_df.at[flare_index, 'Pre-Flare End Time'] = preflare_df['window end'][preflare_map_idx[flare_index_rel]]

        preflare_irradiance_cols = [col for col in flare_settings.jedi_df.columns if 'Pre-Flare Irradiance' in col]
        flare_settings.jedi_df.at[flare_index, preflare_irradiance_cols] = preflare_df.iloc[preflare_map_idx[flare_index_rel], 2:]

        if verbose:
            flare_settings.logger.info("Event {0} pre-flare determination complete.".format(flare_index))

        eve_lines_event = eve_jedi.clip_eve_data_to_dimming_window(flare_index)
        if eve_lines_event is False:
            continue

        # # Convert irradiance units to percent
        # # (in place, don't care about absolute units from this point forward)
        # eve_lines_event = (eve_lines_event - preflare_irradiance) / preflare_irradiance * 100.0
        # if verbose:
        #     flare_settings.logger.info(
        #         "Event {0} irradiance converted from absolute to percent units.".format(flare_index))
        #
        # # Do flare removal in the light curves and add the results to the DataFrame
        # # Profile 1 starts
        # peak_match_subtract_time_start = time.time()
        # eve_jedi.peak_match_subtract(flare_settings, eve_lines_event, jedi_row, flare_index, ion_tuples, ion_permutations)
        # # Profile 1 ends
        # peak_match_subtract_elapsed_time = time.time() - peak_match_subtract_time_start
        # print('Peak match subtract elapsed time: %d s'%peak_match_subtract_elapsed_time)

    #     # TODO: Update calculate_eve_fe_line_precision to compute for all emission lines, not just selected
    #     uncertainty = np.ones(len(eve_lines_event)) * 0.002545
    #
    #     # TODO: Propagate uncertainty through light_curve_peak_match_subtract and store in eve_lines_event
    #
    #     # Fit the light curves to reduce influence of noise on the parameterizations to come later
    #
    #     # Profile 2 starts
    #     light_curve_fitting_start = time.time()
    #     eve_jedi.light_curve_fitting(flare_settings, eve_lines_event, jedi_row, flare_index, uncertainty)
    #     # Profile 2 ends
    #     light_curve_fitting_elapsed_time = time.time() - light_curve_fitting_start
    #     print('Light curve fitting elapsed time: %d' % light_curve_fitting_elapsed_time)
    #
    #     # Save the dimming event data to disk for quicker restore
    #     # jedi_row.to_hdf(processed_jedi_non_params_filename, 'jedi_row')
    #     eve_lines_event.to_hdf(processed_lines_filename, 'eve_lines_event')
    # else:
    #     # jedi_row = pd.read_hdf(processed_jedi_non_params_filename, 'jedi_row')
    #     eve_lines_event = pd.read_hdf(processed_lines_filename, 'eve_lines_event')
    #     if verbose:
    #         flare_settings.logger.info(
    #             'Loading files {0} and {1} rather than processing again.'.format(processed_jedi_non_params_filename,
    #                                                                              processed_lines_filename))

#     # Parameterize the light curves for dimming
#     eve_jedi.determine_dimming(flare_settings, eve_lines_event, jedi_row, flare_index)
#
#     # Profile 0 stops
#     loop_elapsed_time = time.time() - loop_time_start
#     print('Loop elapsed time: %d' % loop_elapsed_time)
#
#
# # Profile total stops
# total_elapsed_time = time.time() - total_time_start

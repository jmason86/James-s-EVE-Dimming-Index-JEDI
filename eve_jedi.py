import os
import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.io.idl import readsav
from sunpy.util.metadata import MetaDict
import astropy.units as u
import itertools
from collections import OrderedDict
from jpm_logger import JpmLogger
from jpm_number_printing import latex_float
import matplotlib as mpl
mpl.use('agg')
from matplotlib import dates
import matplotlib.pyplot as plt
import multiprocessing as mp

from determine_preflare_irradiance import determine_preflare_irradiance
from light_curve_peak_match_subtract import light_curve_peak_match_subtract
from automatic_fit_light_curve import automatic_fit_light_curve
from determine_dimming_depth import determine_dimming_depth
from determine_dimming_slope import determine_dimming_slope
from determine_dimming_duration import determine_dimming_duration

import flare_settings

# class FlareSettings:
#     # For passing an instance as global.
#     # See https://stackoverflow.com/questions/13034496/using-global-variables-between-files
#     # and https://docs.python.org/2/faq/programming.html#how-do-i-share-global-variables-across-modules
#     def __init__(self, eve_data_path, goes_data_path, output_path,
#                  threshold_time_prior_flare_minutes=480.0,
#                  dimming_window_relative_to_flare_minutes_left=-1.0,
#                  dimming_window_relative_to_flare_minutes_right=1440.0,
#                  threshold_minimum_dimming_window_minutes=120.0,
#                  verbose=True):
#
#         self.eve_data_path = eve_data_path
#         self.goes_data_path = goes_data_path
#         self.output_path = output_path
#         self.threshold_time_prior_flare_minutes = threshold_time_prior_flare_minutes
#         self.dimming_window_relative_to_flare_minutes_left = dimming_window_relative_to_flare_minutes_left
#         self.dimming_window_relative_to_flare_minutes_right = dimming_window_relative_to_flare_minutes_right
#         self.threshold_minimum_dimming_window_minutes = threshold_minimum_dimming_window_minutes
#         self.verbose = verbose
#
#         self.logger = JpmLogger(filename='generate_jedi_catalog', path=output_path, console=False)
#         self.logger.info("Starting JEDI processing pipeline.")
#         #self.logger.info("Processing events {0} - {1}".format(flare_index_range[0], flare_index_range[-1]))
#
#         self.csv_filename = output_path + 'jedi_{0}.csv'.format(Time.now().iso)
#         self.n_jobs = 1
#
#         self.eve_lines, self.goes_flare_events = load_eve_goes(self.eve_data_path, self.goes_data_path)
#
#         if verbose:
#             self.logger.info('Loaded GOES flare events.')


# def init(eve_data_path, goes_data_path, output_path, verbose=False):
#     global flare_settings
#     flare_settings = FlareSettings(eve_data_path, goes_data_path, output_path, verbose=verbose)


def merge_jedi_catalog_files(
        file_path='/Users/jmason86/Dropbox/Research/Postdoc_NASA/Analysis/Coronal Dimming Analysis/JEDI Catalog/',
        verbose=False):
    """Function for merging the .csv output files of generate_jedi_catalog()

    Inputs:
        None.

    Optional Inputs:
        file_path [str]: Set to a path for saving the JEDI catalog table.
                         Default is '/Users/jmason86/Dropbox/Research/Postdoc_NASA/Analysis/Coronal Dimming Analysis/JEDI Catalog/'.
        verbose [bool]:  Set to log the processing messages to console. Default is False.

    Outputs:
        No direct return, but writes a csv to disk with the dimming paramerization results.

    Optional Outputs:
        None

    Example:
        generate_jedi_catalog(output_path='/Users/jmason86/Dropbox/Research/Postdoc_NASA/Analysis/Coronal Dimming Analysis/JEDI Catalog/',
                              verbose=True)
    """
    # Create one sorted, clean dataframe from all of the csv files
    list_dfs = []
    for file in os.listdir(file_path):
        if file.endswith(".csv") and "merged" not in file:
            jedi_rows = pd.read_csv(os.path.join(file_path, file), index_col=None, header=0, low_memory=False)
            list_dfs.append(jedi_rows)
    jedi_catalog_df = pd.concat(list_dfs, ignore_index=True)
    jedi_catalog_df.dropna(axis=0, how='all', inplace=True)
    jedi_catalog_df.drop_duplicates(inplace=True)
    jedi_catalog_df.sort_values(by=['Event #'], inplace=True)
    jedi_catalog_df.reset_index(drop=True, inplace=True)
    if verbose:
        print("Read files, sorted, dropped empty and duplicate rows, and reset index.")

    # Write the catalog to disk
    csv_filename = file_path + 'jedi_merged_{0}.csv'.format(Time.now().iso)
    jedi_catalog_df.to_csv(csv_filename, header=True, index=False, mode='w')
    if verbose:
        print("Wrote merged file to {0}".format(csv_filename))

    return 1


def make_metadata(eve_readsav):
    metadata = MetaDict()
    metadata['ion'] = eve_readsav['name']
    metadata['temperature_ion_peak_formation'] = np.power(10.0, eve_readsav['logt']) * u.Kelvin
    metadata['extracted_wavelength_center'] = eve_readsav['wavelength'] * u.nm
    metadata['extracted_wavelength_min'] = metadata['extracted_wavelength_center']
    metadata['extracted_wavelength_max'] = metadata['extracted_wavelength_center']
    metadata['emission_line_blends'] = ['none', 'yay', 'poop', 'Fe vi']  # etc
    metadata['exposure_time'] = 60.0 * u.second  # These example EVE data are already binned down to 1 minute
    metadata['precision'] = ['Not implemented in prototype']
    metadata['accuracy'] = ['Not implemented in prototype']
    metadata['flags'] = ['Not implemented in prototype']
    metadata['flags_description'] = '1 = MEGS-A data is missing, ' \
                                    '2 = MEGS-B data is missing, ' \
                                    '4 = ESP data is missing, ' \
                                    '8 = MEGS-P data is missing, ' \
                                    '16 = Possible clock adjust in MEGS-A, ' \
                                    '32 = Possible clock adjust in MEGS-B, ' \
                                    '64 = Possible clock adjust in ESP, ' \
                                    '128 = Possible clock adjust in MEGS-P'
    metadata['flags_spacecraft'] = ['Not implemented in prototype']
    metadata['flags_spacecraft_description'] = '0 = No obstruction, ' \
                                               '1 = Warm up from Earth eclipse, ' \
                                               '2 = Obstruction atmosphere penumbra, ' \
                                               '3 = Obstruction atmosphere umbra, ' \
                                               '4 = Obstruction penumbra of Mercury, ' \
                                               '5 = Obstruction penumbra of Mercury, ' \
                                               '6 = Obstruction penumbra of Venus, ' \
                                               '7 = Obstruction umbra of Venus, ' \
                                               '8 = Obstruction penumbra of Moon, ' \
                                               '9 = Obstruction umbra of Moon, ' \
                                               '10 = Obstruction penumbra of solid Earth, ' \
                                               '11 = Obstruction umbra of solid Earth, ' \
                                               '16 = Observatory is off-pointed by more than 1 arcmin'
    metadata['data_version'] = ['Not implemented in prototype']
    metadata['data_reprocessed_revision'] = ['Not implemented in prototype']
    metadata['filename'] = ['Not implemented in prototype']

    return metadata


def make_jedi_row(eve_lines, csv_filename):
    # Define the columns of the JEDI catalog
    ion_tuples = list(itertools.permutations(eve_lines.columns.values, 2))
    ion_permutations = pd.Index([' by '.join(ion_tuples[i]) for i in range(len(ion_tuples))])

    # Define the columns of the JEDI catalog
    jedi_row = pd.DataFrame([OrderedDict([
        ('Event #', np.nan),
        ('GOES Flare Start Time', np.nan),
        ('GOES Flare Peak Time', np.nan),
        ('GOES Flare Class', np.nan),
        ('Pre-Flare Start Time', np.nan),
        ('Pre-Flare End Time', np.nan),
        ('Flare Interrupt', np.nan)])])

    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Pre-Flare Irradiance [W/m2]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Slope Start Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Slope End Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Slope Min [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Slope Max [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Slope Mean [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Slope Uncertainty [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Depth Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Depth [%]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Depth Uncertainty [%]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Duration Start Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Duration End Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Duration [s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Fitting Gamma'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=eve_lines.columns + ' Fitting Score'))

    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Slope Start Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Slope End Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Slope Min [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Slope Max [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Slope Mean [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Slope Uncertainty [%/s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Depth Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Depth [%]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Depth Uncertainty [%]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Duration Start Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Duration End Time'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Duration [s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Correction Time Shift [s]'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Correction Scale Factor'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Fitting Gamma'))
    jedi_row = jedi_row.join(pd.DataFrame(columns=ion_permutations + ' Fitting Score'))

    jedi_row.to_csv(csv_filename, header=True, index=False, mode='w')

    return jedi_row, ion_tuples, ion_permutations


def map_true_indices(tmask, irange):
    """
    Given a mask of True/False values and an array of indices, create a new array of same size as irange,
    with the index of the nearest preceding True value.

    :param tmask: logical numpy array. False wherever we need the location of the nearest preceding True.
    :param irange: array of tmask indices. Must satisfy len(irange) <= tmask.size()
    :return: Array of indices of True in tmask that are immediately preceding the irange indices.

    Example:
    tmask = np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
    irange = range(0,8)
    returns: array([0, 0, 0, 3, 4, 5, 5, 7])
    """

    true_indices = np.where(tmask)[0]
    mapped_indices = np.empty(len(irange), dtype=np.int)

    for i, index in enumerate(irange):
        index_loc = np.where(true_indices <= index)[0][-1]
        mapped_indices[i] = true_indices[index_loc]

    return mapped_indices


def map_true_indices2(tmask, irange):
    """ Fast version of map_true_indices

    Given a mask of True/False values and an array of indices, create a new array of same size as irange,
    with the index of the nearest preceding True value.

    :param tmask: logical numpy array. False wherever we need the location of the nearest preceding True.
    :param irange: array of tmask indices. Must satisfy len(irange) <= tmask.size()
    :return: Array of indices of True in tmask that are immediately preceding the irange indices.

    Example:
    tmask = np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
    irange = range(0,8)
    returns: array([0, 0, 0, 3, 4, 5, 5, 7])
    """

    invalid_index = -1
    idx = np.where(tmask)[0]
    sidx = np.searchsorted(idx, irange, 'right')-1
    indices = np.where(sidx == -1, invalid_index, idx[sidx])
    return indices





def loop_preflare_irradiance(flare_index):

    print('Processing event at flare_index = %d' % flare_index)
    # Clip EVE data from threshold_time_prior_flare_minutes prior to flare up to peak flare time
    preflare_window_start = (flare_settings.goes_flare_events['peak_time'][flare_index] - (flare_settings.threshold_time_prior_flare_minutes * u.minute)).iso
    preflare_window_end = (flare_settings.goes_flare_events['peak_time'][flare_index]).iso
    eve_lines_preflare_time = flare_settings.eve_lines[preflare_window_start:preflare_window_end]

    # Loop through the emission lines and get pre-flare irradiance for each
    preflare_irradiance = []
    for column in eve_lines_preflare_time:
        #print("column %s" % column)
        eve_line_preflare_time = pd.DataFrame(eve_lines_preflare_time[column])
        eve_line_preflare_time.columns = ['irradiance']

        preflare_temp = determine_preflare_irradiance(eve_line_preflare_time,
                                                      pd.Timestamp(flare_settings.goes_flare_events['start_time'][flare_index].iso),
                                                      plot_path_filename=os.path.join(flare_settings.output_path, 'Preflare_Determination', 'Event_%d_%s.png' % (flare_index, column)),
                                                      verbose=flare_settings.verbose,
                                                      logger=flare_settings.logger)

        preflare_irradiance.append(preflare_temp)

    return preflare_irradiance, preflare_window_start, preflare_window_end


def multiprocess_preflare_irradiance(preflare_indices, nworkers):

    if nworkers == 1:
        preflare_irradiances, preflare_windows_start, preflare_windows_end = zip(*map(loop_preflare_irradiance, preflare_indices))
        print('Preparing export of dataframe')
    else:
        pool = mp.Pool(processes=nworkers)
        preflare_irradiances, preflare_windows_start, preflare_windows_end = zip(*pool.map(loop_preflare_irradiance, preflare_indices))
        pool.close()
        print('Pool closed. Preparing export of dataframe')

    preflare_irradiances = np.array(preflare_irradiances)
    preflare_windows_start = preflare_windows_start
    preflare_windows_end = preflare_windows_end

    return preflare_irradiances, preflare_windows_start, preflare_windows_end


def clip_eve_data_to_dimming_window(jedi_df, flare_index):

    flare_interrupt = False

    # Clip EVE data to dimming window
    bracket_time_left = (flare_settings.goes_flare_events['peak_time'][flare_index] + (flare_settings.dimming_window_relative_to_flare_minutes_left * u.minute))
    next_flare_time = Time((flare_settings.goes_flare_events['peak_time'][flare_index + 1]).iso)
    user_choice_time = (flare_settings.goes_flare_events['peak_time'][flare_index] + (flare_settings.dimming_window_relative_to_flare_minutes_right * u.minute))
    bracket_time_right = min(next_flare_time, user_choice_time)

    # If flare is shortening the window, set the flare_interrupt flag
    if bracket_time_right == next_flare_time:
        flare_interrupt = True
        if flare_settings.verbose:
            flare_settings.logger.info('Flare interrupt for event at {0} by flare at {1}'.format(flare_settings.goes_flare_events['peak_time'][flare_index].iso, next_flare_time))

    # Write flare_interrupt to JEDI row
    jedi_df.loc[:, 'Flare Interrupt'] = flare_interrupt

    if ((bracket_time_right - bracket_time_left).sec / 60.0) < flare_settings.threshold_minimum_dimming_window_minutes:
        # Leave all dimming parameters as NaN and write this null result to the CSV on disk
        jedi_df.to_csv(flare_settings.csv_filename, header=False, index=False, mode='a')

        # Log message
        if flare_settings.verbose:
            flare_settings.logger.info(
                'The dimming window duration of {0} minutes is shorter than the minimum threshold of {1} minutes. Skipping this event ({2})'
                    .format(((bracket_time_right - bracket_time_left).sec / 60.0),
                            flare_settings.threshold_minimum_dimming_window_minutes,
                            flare_settings.goes_flare_events['peak_time'][flare_index]))

        eve_lines_event = False

    else:
        eve_lines_event = flare_settings.eve_lines[bracket_time_left.iso:bracket_time_right.iso]
        if flare_settings.verbose:
            flare_settings.logger.info("Event {0} EVE data clipped to dimming window.".format(flare_index))

    return eve_lines_event


def peak_match_subtract(flare_settings, eve_lines_event, jedi_row, flare_index, ion_tuples, ion_permutations):

    for i in range(len(ion_tuples)):
        light_curve_to_subtract_from_df = pd.DataFrame(eve_lines_event[ion_tuples[i][0]])
        light_curve_to_subtract_from_df.columns = ['irradiance']
        light_curve_to_subtract_with_df = pd.DataFrame(eve_lines_event[ion_tuples[i][1]])
        light_curve_to_subtract_with_df.columns = ['irradiance']

        if (light_curve_to_subtract_from_df.isnull().all().all()) or (light_curve_to_subtract_with_df.isnull().all().all()):
            if flare_settings.verbose:
                flare_settings.logger.info(
                    'Event {0} {1} correction skipped because all irradiances are NaN.'.format(flare_index,
                                                                                               ion_permutations[
                                                                                                   i]))
        else:
            light_curve_corrected, seconds_shift, scale_factor = light_curve_peak_match_subtract(
                light_curve_to_subtract_from_df,
                light_curve_to_subtract_with_df,
                pd.Timestamp((flare_settings.goes_flare_events['peak_time'][flare_index]).iso),
                plot_path_filename=flare_settings.output_path + 'Peak Subtractions/Event {0} {1}.png'.format(flare_index,
                                                                                              ion_permutations[i]),
                verbose=flare_settings.verbose, logger=flare_settings.logger)

            eve_lines_event[ion_permutations[i]] = light_curve_corrected
            jedi_row[ion_permutations[i] + ' Correction Time Shift [s]'] = seconds_shift
            jedi_row[ion_permutations[i] + ' Correction Scale Factor'] = scale_factor

            plt.close('all')

            if flare_settings.verbose:
                flare_settings.logger.info('Event {0} flare removal correction complete'.format(flare_index))


def light_curve_fitting(flare_settings, eve_lines_event, jedi_row, flare_index, uncertainty):

    for i, column in enumerate(eve_lines_event):
        if eve_lines_event[column].isnull().all().all():
            if flare_settings.verbose:
                flare_settings.logger.info(
                    'Event {0} {1} fitting skipped because all irradiances are NaN.'.format(flare_index, column))
        else:
            eve_line_event = pd.DataFrame(eve_lines_event[column])
            eve_line_event.columns = ['irradiance']
            eve_line_event['uncertainty'] = uncertainty

            fitting_path = flare_settings.output_path + 'Fitting/'
            if not os.path.exists(fitting_path):
                os.makedirs(fitting_path)

            plt.close('all')
            light_curve_fit, best_fit_gamma, best_fit_score = automatic_fit_light_curve(eve_line_event,
                                                                                        gamma=np.array([5e-8]),
                                                                                        plots_save_path='{0}Event {1} {2} '.format(
                                                                                            fitting_path,
                                                                                            flare_index, column),
                                                                                        verbose=flare_settings.verbose,
                                                                                        logger=flare_settings.logger,
                                                                                        n_jobs=flare_settings.n_jobs)
            eve_lines_event[column] = light_curve_fit
            jedi_row[column + ' Fitting Gamma'] = best_fit_gamma
            jedi_row[column + ' Fitting Score'] = best_fit_score

            if flare_settings.verbose:
                flare_settings.logger.info('Event {0} {1} light curves fitted.'.format(flare_index, column))


def determine_dimming(flare_settings, eve_lines_event, jedi_row, flare_index):

    for column in eve_lines_event:

        # Null out all parameters
        depth_percent, depth_time = np.nan, np.nan
        slope_start_time, slope_end_time = np.nan, np.nan
        slope_min, slope_max, slope_mean = np.nan, np.nan, np.nan
        duration_seconds, duration_start_time, duration_end_time = np.nan, np.nan, np.nan

        # Determine whether to do the parameterizations or not
        if eve_lines_event[column].isnull().all().all():
            if flare_settings.verbose:
                flare_settings.logger.info(
                    'Event {0} {1} parameterization skipped because all irradiances are NaN.'.format(flare_index,
                                                                                                     column))
        else:
            eve_line_event = pd.DataFrame(eve_lines_event[column])
            eve_line_event.columns = ['irradiance']

            # Determine dimming depth (if any)
            depth_path = flare_settings.output_path + 'Depth/'
            if not os.path.exists(depth_path):
                os.makedirs(depth_path)

            plt.close('all')
            depth_percent, depth_time = determine_dimming_depth(eve_line_event,
                                                                plot_path_filename='{0}Event {1} {2} Depth.png'.format(
                                                                    depth_path, flare_index, column),
                                                                verbose=flare_settings.verbose, logger=flare_settings.logger)

            jedi_row[column + ' Depth [%]'] = depth_percent
            # jedi_row[column + ' Depth Uncertainty [%]'] = depth_uncertainty  # TODO: make determine_dimming_depth return the propagated uncertainty
            jedi_row[column + ' Depth Time'] = depth_time

            # Determine dimming slope (if any)
            slope_path = flare_settings.output_path + 'Slope/'
            if not os.path.exists(slope_path):
                os.makedirs(slope_path)

            slope_start_time = pd.Timestamp((flare_settings.goes_flare_events['peak_time'][flare_index]).iso)
            slope_end_time = depth_time

            if (pd.isnull(slope_start_time)) or (pd.isnull(slope_end_time)):
                if flare_settings.verbose:
                    flare_settings.logger.warning('Cannot compute slope or duration because slope bounding times NaN.')
            else:
                plt.close('all')
                slope_min, slope_max, slope_mean = determine_dimming_slope(eve_line_event,
                                                                           earliest_allowed_time=slope_start_time,
                                                                           latest_allowed_time=slope_end_time,
                                                                           plot_path_filename='{0}Event {1} {2} Slope.png'.format(
                                                                               slope_path, flare_index, column),
                                                                           verbose=flare_settings.verbose, logger=flare_settings.logger)

                jedi_row[column + ' Slope Min [%/s]'] = slope_min
                jedi_row[column + ' Slope Max [%/s]'] = slope_max
                jedi_row[column + ' Slope Mean [%/s]'] = slope_mean
                # jedi_row[column + ' Slope Uncertainty [%]'] = slope_uncertainty  # TODO: make determine_dimming_depth return the propagated uncertainty
                jedi_row[column + ' Slope Start Time'] = slope_start_time
                jedi_row[column + ' Slope End Time'] = slope_end_time

                # Determine dimming duration (if any)
                duration_path = flare_settings.output_path + 'Duration/'
                if not os.path.exists(duration_path):
                    os.makedirs(duration_path)

                plt.close('all')
                duration_seconds, duration_start_time, duration_end_time = determine_dimming_duration(eve_line_event,
                                                                                                      earliest_allowed_time=slope_start_time,
                                                                                                      plot_path_filename='{0}Event {1} {2} Duration.png'.format(
                                                                                                          duration_path,
                                                                                                          flare_index,
                                                                                                          column),
                                                                                                      verbose=flare_settings.verbose,
                                                                                                      logger=flare_settings.logger)

                jedi_row[column + ' Duration [s]'] = duration_seconds
                jedi_row[column + ' Duration Start Time'] = duration_start_time
                jedi_row[column + ' Duration End Time'] = duration_end_time

            if flare_settings.verbose:
                flare_settings.logger.info("Event {0} {1} parameterizations complete.".format(flare_index, column))

            # Produce a summary plot for each light curve
            # plt.style.use('jpm-transparent-light')

            ax = eve_line_event['irradiance'].plot(color='black')
            plt.axhline(linestyle='dashed', color='grey')
            start_date = eve_line_event.index.values[0]
            start_date_string = pd.to_datetime(str(start_date))
            plt.xlabel(start_date_string.strftime('%Y-%m-%d %H:%M:%S'))
            plt.ylabel('Irradiance [%]')
            fmtr = dates.DateFormatter("%H:%M:%S")
            ax.xaxis.set_major_formatter(fmtr)
            ax.xaxis.set_major_locator(dates.HourLocator())
            plt.title('Event {0} {1} nm Parameters'.format(flare_index, column))

            if not np.isnan(depth_percent):
                plt.annotate('', xy=(depth_time, -depth_percent), xycoords='data',
                             xytext=(depth_time, 0), textcoords='data',
                             arrowprops=dict(facecolor='limegreen', edgecolor='limegreen', linewidth=2))
                mid_depth = -depth_percent / 2.0
                plt.annotate('{0:.2f} %'.format(depth_percent), xy=(depth_time, mid_depth), xycoords='data',
                             ha='right', va='center', rotation=90, size=18, color='limegreen')

            if not np.isnan(slope_mean):
                p = plt.plot(eve_line_event[slope_start_time:slope_end_time]['irradiance'], c='tomato')

                inverse_str = '$^{-1}$'
                plt.annotate('slope_min={0} % s{1}'.format(latex_float(slope_min), inverse_str),
                             xy=(0.98, 0.12), xycoords='axes fraction', ha='right',
                             size=12, color=p[0].get_color())
                plt.annotate('slope_max={0} % s{1}'.format(latex_float(slope_max), inverse_str),
                             xy=(0.98, 0.08), xycoords='axes fraction', ha='right',
                             size=12, color=p[0].get_color())
                plt.annotate('slope_mean={0} % s{1}'.format(latex_float(slope_mean), inverse_str),
                             xy=(0.98, 0.04), xycoords='axes fraction', ha='right',
                             size=12, color=p[0].get_color())

            if not np.isnan(duration_seconds):
                plt.annotate('', xy=(duration_start_time, 0), xycoords='data',
                             xytext=(duration_end_time, 0), textcoords='data',
                             arrowprops=dict(facecolor='dodgerblue', edgecolor='dodgerblue', linewidth=5,
                                             arrowstyle='<->'))
                mid_time = duration_start_time + (duration_end_time - duration_start_time) / 2
                plt.annotate(str(duration_seconds) + ' s', xy=(mid_time, 0), xycoords='data', ha='center', va='bottom',
                             size=18, color='dodgerblue')

            summary_path = flare_settings.output_path + 'Summary Plots/'
            if not os.path.exists(summary_path):
                os.makedirs(summary_path)
            summary_filename = '{0}Event {1} {2} Parameter Summary.png'.format(summary_path, flare_index, column)
            plt.savefig(summary_filename)
            plt.close('all')

            if flare_settings.verbose:
                flare_settings.logger.info("Summary plot saved to %s" % summary_filename)




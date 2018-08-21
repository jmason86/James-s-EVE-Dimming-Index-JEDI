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
from light_curve_fit import light_curve_fit
from determine_dimming_depth import determine_dimming_depth
from determine_dimming_slope import determine_dimming_slope
from determine_dimming_duration import determine_dimming_duration

import jedi_config

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


def map_true_indices2(tmask, irange):
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

    invalid_index = -1
    idx = np.where(tmask)[0]
    sidx = np.searchsorted(idx, irange, 'right')-1
    indices = np.where(sidx == -1, invalid_index, idx[sidx])
    return indices


def loop_preflare_irradiance(flare_index):

    print('Processing event at flare_index = %d' % flare_index)
    # Clip EVE data from threshold_time_prior_flare_minutes prior to flare up to peak flare time
    preflare_window_start = (jedi_config.goes_flare_events['peak_time'][flare_index] - (jedi_config.threshold_time_prior_flare_minutes * u.minute)).iso
    preflare_window_end = (jedi_config.goes_flare_events['peak_time'][flare_index]).iso
    eve_lines_preflare_time = jedi_config.eve_lines[preflare_window_start:preflare_window_end]

    # Loop through the emission lines and get pre-flare irradiance for each
    preflare_irradiance = []
    for column in eve_lines_preflare_time:
        #print("column %s" % column)
        eve_line_preflare_time = pd.DataFrame(eve_lines_preflare_time[column])
        eve_line_preflare_time.columns = ['irradiance']

        preflare_temp = determine_preflare_irradiance(eve_line_preflare_time,
                                                      pd.Timestamp(jedi_config.goes_flare_events['start_time'][flare_index].iso),
                                                      plot_path_filename=os.path.join(jedi_config.output_path, 'Preflare_Determination', 'Event_%d_%s.png' % (flare_index, column)),
                                                      verbose=jedi_config.verbose,
                                                      logger=jedi_config.logger)

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


def clip_eve_data_to_dimming_window(flare_index):

    flare_interrupt = False

    # Clip EVE data to dimming window
    bracket_time_left = (jedi_config.goes_flare_events['peak_time'][flare_index] + (jedi_config.dimming_window_relative_to_flare_minutes_left * u.minute))
    next_flare_time = Time((jedi_config.goes_flare_events['peak_time'][flare_index + 1]).iso)
    user_choice_time = (jedi_config.goes_flare_events['peak_time'][flare_index] + (jedi_config.dimming_window_relative_to_flare_minutes_right * u.minute))
    bracket_time_right = min(next_flare_time, user_choice_time)

    # If flare is shortening the window, set the flare_interrupt flag
    if bracket_time_right == next_flare_time:
        flare_interrupt = True
        if jedi_config.verbose:
            jedi_config.logger.info('Flare interrupt for event at {0} by flare at {1}'.format(jedi_config.goes_flare_events['peak_time'][flare_index].iso, next_flare_time))

    # Write flare_interrupt to JEDI row
    jedi_config.jedi_df.at[flare_index, 'Flare Interrupt'] = flare_interrupt

    if ((bracket_time_right - bracket_time_left).sec / 60.0) < jedi_config.threshold_minimum_dimming_window_minutes:
        # Leave all dimming parameters as NaN and write this null result to the CSV on disk

        # TODO: TO BE REVIEWED IF USING jedi_df as a ~5k x 24k dataframe!!!!!
        jedi_config.jedi_df.to_csv(jedi_config.jedi_csv_filename, header=False, index=False, mode='a')

        # Log message
        if jedi_config.verbose:
            jedi_config.logger.info(
                'The dimming window duration of {0} minutes is shorter than the minimum threshold of {1} minutes. Skipping this event ({2})'
                    .format(((bracket_time_right - bracket_time_left).sec / 60.0),
                            jedi_config.threshold_minimum_dimming_window_minutes,
                            jedi_config.goes_flare_events['peak_time'][flare_index]))

        eve_lines_event = False

    else:
        eve_lines_event = jedi_config.eve_lines[bracket_time_left.iso:bracket_time_right.iso]
        if jedi_config.verbose:
            jedi_config.logger.info("Event {0} EVE data clipped to dimming window.".format(flare_index))

    return eve_lines_event


def peak_match_subtract(jedi_config, eve_lines_event, jedi_row, flare_index, ion_tuples, ion_permutations):

    for i in range(len(ion_tuples)):
        light_curve_to_subtract_from_df = pd.DataFrame(eve_lines_event[ion_tuples[i][0]])
        light_curve_to_subtract_from_df.columns = ['irradiance']
        light_curve_to_subtract_with_df = pd.DataFrame(eve_lines_event[ion_tuples[i][1]])
        light_curve_to_subtract_with_df.columns = ['irradiance']

        if (light_curve_to_subtract_from_df.isnull().all().all()) or (light_curve_to_subtract_with_df.isnull().all().all()):
            if jedi_config.verbose:
                jedi_config.logger.info(
                    'Event {0} {1} correction skipped because all irradiances are NaN.'.format(flare_index,
                                                                                               ion_permutations[
                                                                                                   i]))
        else:
            light_curve_corrected, seconds_shift, scale_factor = light_curve_peak_match_subtract(
                light_curve_to_subtract_from_df,
                light_curve_to_subtract_with_df,
                pd.Timestamp((jedi_config.goes_flare_events['peak_time'][flare_index]).iso),
                plot_path_filename=jedi_config.output_path + 'Peak Subtractions/Event {0} {1}.png'.format(flare_index,
                                                                                              ion_permutations[i]),
                verbose=jedi_config.verbose, logger=jedi_config.logger)

            eve_lines_event[ion_permutations[i]] = light_curve_corrected
            jedi_row[ion_permutations[i] + ' Correction Time Shift [s]'] = seconds_shift
            jedi_row[ion_permutations[i] + ' Correction Scale Factor'] = scale_factor

            plt.close('all')

            if jedi_config.verbose:
                jedi_config.logger.info('Event {0} flare removal correction complete'.format(flare_index))


def light_curve_fitting(jedi_config, eve_lines_event, jedi_row, flare_index, uncertainty):

    for i, column in enumerate(eve_lines_event):
        if eve_lines_event[column].isnull().all().all():
            if jedi_config.verbose:
                jedi_config.logger.info(
                    'Event {0} {1} fitting skipped because all irradiances are NaN.'.format(flare_index, column))
        else:
            eve_line_event = pd.DataFrame(eve_lines_event[column])
            eve_line_event.columns = ['irradiance']
            eve_line_event['uncertainty'] = uncertainty

            fitting_path = jedi_config.output_path + 'Fitting/'
            if not os.path.exists(fitting_path):
                os.makedirs(fitting_path)

            plt.close('all')
            light_curve_fit_df, best_fit_gamma, best_fit_score = light_curve_fit(eve_line_event,
                                                                                        gamma=np.array([5e-8]),
                                                                                        plots_save_path='{0}Event {1} {2} '.format(
                                                                                            fitting_path,
                                                                                            flare_index, column),
                                                                                        verbose=jedi_config.verbose,
                                                                                        logger=jedi_config.logger,
                                                                                        n_jobs=jedi_config.n_threads)
            eve_lines_event[column] = light_curve_fit_df
            jedi_row[column + ' Fitting Gamma'] = best_fit_gamma
            jedi_row[column + ' Fitting Score'] = best_fit_score

            if jedi_config.verbose:
                jedi_config.logger.info('Event {0} {1} light curves fitted.'.format(flare_index, column))


def determine_dimming(jedi_config, eve_lines_event, jedi_row, flare_index):

    for column in eve_lines_event:

        # Null out all parameters
        depth_percent, depth_time = np.nan, np.nan
        slope_start_time, slope_end_time = np.nan, np.nan
        slope_min, slope_max, slope_mean = np.nan, np.nan, np.nan
        duration_seconds, duration_start_time, duration_end_time = np.nan, np.nan, np.nan

        # Determine whether to do the parameterizations or not
        if eve_lines_event[column].isnull().all().all():
            if jedi_config.verbose:
                jedi_config.logger.info(
                    'Event {0} {1} parameterization skipped because all irradiances are NaN.'.format(flare_index,
                                                                                                     column))
        else:
            eve_line_event = pd.DataFrame(eve_lines_event[column])
            eve_line_event.columns = ['irradiance']

            # Determine dimming depth (if any)
            depth_path = jedi_config.output_path + 'Depth/'
            if not os.path.exists(depth_path):
                os.makedirs(depth_path)

            plt.close('all')
            depth_percent, depth_time = determine_dimming_depth(eve_line_event,
                                                                plot_path_filename='{0}Event {1} {2} Depth.png'.format(
                                                                    depth_path, flare_index, column),
                                                                verbose=jedi_config.verbose, logger=jedi_config.logger)

            jedi_row[column + ' Depth [%]'] = depth_percent
            # jedi_row[column + ' Depth Uncertainty [%]'] = depth_uncertainty  # TODO: make determine_dimming_depth return the propagated uncertainty
            jedi_row[column + ' Depth Time'] = depth_time

            # Determine dimming slope (if any)
            slope_path = jedi_config.output_path + 'Slope/'
            if not os.path.exists(slope_path):
                os.makedirs(slope_path)

            slope_start_time = pd.Timestamp((jedi_config.goes_flare_events['peak_time'][flare_index]).iso)
            slope_end_time = depth_time

            if (pd.isnull(slope_start_time)) or (pd.isnull(slope_end_time)):
                if jedi_config.verbose:
                    jedi_config.logger.warning('Cannot compute slope or duration because slope bounding times NaN.')
            else:
                plt.close('all')
                slope_min, slope_max, slope_mean = determine_dimming_slope(eve_line_event,
                                                                           earliest_allowed_time=slope_start_time,
                                                                           latest_allowed_time=slope_end_time,
                                                                           plot_path_filename='{0}Event {1} {2} Slope.png'.format(
                                                                               slope_path, flare_index, column),
                                                                           verbose=jedi_config.verbose, logger=jedi_config.logger)

                jedi_row[column + ' Slope Min [%/s]'] = slope_min
                jedi_row[column + ' Slope Max [%/s]'] = slope_max
                jedi_row[column + ' Slope Mean [%/s]'] = slope_mean
                # jedi_row[column + ' Slope Uncertainty [%]'] = slope_uncertainty  # TODO: make determine_dimming_depth return the propagated uncertainty
                jedi_row[column + ' Slope Start Time'] = slope_start_time
                jedi_row[column + ' Slope End Time'] = slope_end_time

                # Determine dimming duration (if any)
                duration_path = jedi_config.output_path + 'Duration/'
                if not os.path.exists(duration_path):
                    os.makedirs(duration_path)

                plt.close('all')
                duration_seconds, duration_start_time, duration_end_time = determine_dimming_duration(eve_line_event,
                                                                                                      earliest_allowed_time=slope_start_time,
                                                                                                      plot_path_filename='{0}Event {1} {2} Duration.png'.format(
                                                                                                          duration_path,
                                                                                                          flare_index,
                                                                                                          column),
                                                                                                      verbose=jedi_config.verbose,
                                                                                                      logger=jedi_config.logger)

                jedi_row[column + ' Duration [s]'] = duration_seconds
                jedi_row[column + ' Duration Start Time'] = duration_start_time
                jedi_row[column + ' Duration End Time'] = duration_end_time

            if jedi_config.verbose:
                jedi_config.logger.info("Event {0} {1} parameterizations complete.".format(flare_index, column))

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

            summary_path = jedi_config.output_path + 'Summary Plots/'
            if not os.path.exists(summary_path):
                os.makedirs(summary_path)
            summary_filename = '{0}Event {1} {2} Parameter Summary.png'.format(summary_path, flare_index, column)
            plt.savefig(summary_filename)
            plt.close('all')

            if jedi_config.verbose:
                jedi_config.logger.info("Summary plot saved to %s" % summary_filename)




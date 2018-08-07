# Standard modules
import os
import itertools
from collections import OrderedDict
import numpy as np
import matplotlib as mpl
#mpl.use('macosx') # For interactive plotting
mpl.use('agg')
from matplotlib import dates
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
import time
#import progressbar
import multiprocessing as mp
from functools import partial

# Custom modules
from jpm_number_printing import latex_float
# from get_goes_flare_events import get_goes_flare_events  # TODO: Uncomment once sunpy method implemented
from determine_preflare_irradiance import multiprocess_preflare_irradiance
from light_curve_peak_match_subtract import light_curve_peak_match_subtract
from light_curve_fit import light_curve_fit
from determine_dimming_depth import determine_dimming_depth
from determine_dimming_slope import determine_dimming_slope
from determine_dimming_duration import determine_dimming_duration

# Configuration
import jedi_config

__author__ = 'James Paul Mason'
__contact__ = 'jmason86@gmail.com'


def generate_jedi_catalog(flare_index_range=range(0, 5052),
                          compute_new_preflare_irradiances=False,
                          threshold_time_prior_flare_minutes=480.0,
                          dimming_window_relative_to_flare_minutes_left=-1.0,
                          dimming_window_relative_to_flare_minutes_right=1440.0,
                          threshold_minimum_dimming_window_minutes=120.0,
                          output_path='/Users/jmason86/Dropbox/Research/Postdoc_NASA/Analysis/Coronal Dimming Analysis/JEDI Catalog/',
                          verbose=True):
    """Wrapper code for creating James's Extreme Ultraviolet Variability Experiment (EVE) Dimming Index (JEDI) catalog.

    Inputs:
        flare_index_range [range]: The range of GOES flare indices to process. Default is range(0, 5052).

    Optional Inputs:
        compute_new_preflare_irradiances [bool]:                Set to force reprocessing of pre-flare irradiances. Will also occur if preflare file doesn't exist on disk.
        threshold_time_prior_flare_minutes [float]:             How long before a particular event does the last one need to have
                                                                occurred to be considered independent. If the previous one was too
                                                                recent, will use that event's pre-flare irradiance.
                                                                The mean dimming time of 100 dimming events in
                                                                Reinard and Biesecker (2008, 2009) is the default.
                                                                Default is 480 (8 hours).
        dimming_window_relative_to_flare_minutes_left [float]:  Defines the left side of the time window to search for dimming
                                                                relative to the GOES/XRS flare peak. Negative numbers mean
                                                                minutes prior to the flare peak. Default is 0.
        dimming_window_relative_to_flare_minutes_right [float]: Defines the right side of the time window to search for dimming
                                                                relative to the GOES/XRS flare peak. If another flare
                                                                occurs before this, that time will define the end of the
                                                                window instead. The time that "most" of the 100 dimming events had recovered
                                                                by in Reinard and Biesecker (2008, 2009) is the default.
                                                                Default is 1440 (24 hours).
        threshold_minimum_dimming_window_minutes [float]:       The smallest allowed time window in which to search for dimming.
                                                                Default is 120 (2 hours).
        output_path [str]:                                      Set to a path for saving the JEDI catalog table and processing
                                                                summary plots. Default is '/Users/jmason86/Dropbox/Research/Postdoc_NASA/Analysis/Coronal Dimming Analysis/JEDI Catalog/'.
        verbose [bool]:                                         Set to log the processing messages to disk and console. Default is False.

    Outputs:
        No direct return, but writes a csv to disk with the dimming paramerization results.
        Subroutines also optionally save processing plots to disk in output_path.

    Optional Outputs:
        None

    Example:
        generate_jedi_catalog(output_path='/Users/jmason86/Dropbox/Research/Postdoc_NASA/Analysis/Coronal Dimming Analysis/JEDI Catalog/',
                              verbose=True)
    """

    # Force flare_index_range to be an array type so it can be indexed in later code
    if isinstance(flare_index_range, int):
        flare_index_range = np.array([flare_index_range])

    # Load data
    jedi_config.init()

    # Define the columns of the JEDI catalog
    jedi_row = jedi_config.init_jedi_row()

    if verbose:
        jedi_config.logger.info('Created JEDI row definition.')

    # Start a progress bar
    #widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.Timer(), ' ', progressbar.AdaptiveETA()]
    #progress_bar = progressbar.ProgressBar(widgets=[progressbar.FormatLabel('Flare event loop: ')] + widgets,
    #                                       min_value=flare_index_range[0], max_value=flare_index_range[-1]).start()

    # Compute all pre-flare irradiances if needed or because kwarg set
    preflare_irradiance_filename = output_path + 'Processed Pre-Flare Irradiances/preflare_irradiances.csv'
    os.makedirs(os.path.dirname(preflare_irradiance_filename), exist_ok=True)  # Create folder if it doesn't exist already
    if compute_new_preflare_irradiances or (os.path.isfile(preflare_irradiance_filename) is False):
        jedi_config.logger.info('Recomputing pre-flare irradiances.')
        preflare_irradiances, \
            preflare_windows_start, \
            preflare_windows_end = multiprocess_preflare_irradiance(jedi_config.preflare_indices,
                                                                    nworkers=5,
                                                                    verbose=verbose)
        preflare_df = pd.DataFrame(columns=[preflare_irradiances, preflare_windows_start, preflare_windows_end])
        preflare_df.to_csv(preflare_irradiance_filename, index=False, mode='w')
    else:
        preflare_df = pd.read_csv(preflare_irradiance_filename, index_col=None, low_memory=False)  # TODO: is index_col=None right?

    # Start loop through all flares
    for flare_index in flare_index_range:
        loop_time = time.time()

        # Skip event 0 to avoid problems with referring to earlier indices
        if flare_index == 0:
            continue

        jedi_config.logger.info('Running on event {0}'.format(flare_index))

        # Reset jedi_row
        jedi_row[:] = np.nan

        # Reset the flare interrupt flag
        flare_interrupt = False

        # Fill the GOES flare information into the JEDI row
        jedi_row['Event #'] = flare_index
        jedi_row['GOES Flare Start Time'] = goes_flare_events['start_time'][flare_index].iso
        jedi_row['GOES Flare Peak Time'] = goes_flare_events['peak_time'][flare_index].iso
        jedi_row['GOES Flare Class'] = goes_flare_events['class'][flare_index]
        if verbose:
            jedi_config.logger.info("Event {0} GOES flare details stored to JEDI row.".format(flare_index))

        # If haven't already done all pre-parameterization processing
        processed_jedi_non_params_filename = output_path + 'Processed Pre-Parameterization Data/Event {0} Pre-Parameterization.h5'.format(flare_index)
        processed_lines_filename = output_path + 'Processed Lines Data/Event {0} Lines.h5'.format(flare_index)
        if not os.path.isfile(processed_lines_filename) or not os.path.isfile(processed_jedi_non_params_filename):
            jedi_row["Pre-Flare Start Time"] = preflare_df['preflare_windows_start'].iloc[map_flare_index_to_preflare_index(flare_index)]
            jedi_row["Pre-Flare End Time"] = preflare_df['preflare_windows_end'].iloc[map_flare_index_to_preflare_index(flare_index)]
            preflare_irradiance_cols = [col for col in jedi_row.columns if 'Pre-Flare Irradiance' in col]
            jedi_row[preflare_irradiance_cols] = preflare_df['preflare_irradiance'].iloc[map_flare_index_to_preflare_index(flare_index)]  # TODO: Is n_element preflare_irradiance handled right here?

            if verbose:
                jedi_config.logger.info("Event {0} pre-flare determination complete.".format(flare_index))

            # Clip EVE data to dimming window
            bracket_time_left = (goes_flare_events['peak_time'][flare_index] + (dimming_window_relative_to_flare_minutes_left * u.minute))
            next_flare_time = Time((goes_flare_events['peak_time'][flare_index + 1]).iso)
            user_choice_time = (goes_flare_events['peak_time'][flare_index] + (dimming_window_relative_to_flare_minutes_right * u.minute))
            bracket_time_right = min(next_flare_time, user_choice_time)

            # If flare is shortening the window, set the flare_interrupt flag
            if bracket_time_right == next_flare_time:
                flare_interrupt = True
                if verbose:
                    jedi_config.logger.info('Flare interrupt for event at {0} by flare at {1}'.format(goes_flare_events['peak_time'][flare_index].iso, next_flare_time))

            # Write flare_interrupt to JEDI row
            jedi_row['Flare Interrupt'] = flare_interrupt

            # Skip event if the dimming window is too short
            if ((bracket_time_right - bracket_time_left).sec / 60.0) < threshold_minimum_dimming_window_minutes:
                # Leave all dimming parameters as NaN and write this null result to the CSV on disk
                jedi_row.to_csv(csv_filename, header=False, index=False, mode='a')

                # Log message
                if verbose:
                    jedi_config.logger.info('The dimming window duration of {0} minutes is shorter than the minimum threshold of {1} minutes. Skipping this event ({2})'
                                .format(((bracket_time_right - bracket_time_left).sec / 60.0), threshold_minimum_dimming_window_minutes, goes_flare_events['peak_time'][flare_index]))

                # Skip the rest of the processing in the flare_index loop
                continue
            else:
                eve_lines_event = eve_lines[bracket_time_left.iso:bracket_time_right.iso]

            if verbose:
                jedi_config. \
                    jedi_config. \
                    jedi_config.logger.info("Event {0} EVE data clipped to dimming window.".format(flare_index))

            # Convert irradiance units to percent
            # (in place, don't care about absolute units from this point forward)
            eve_lines_event = (eve_lines_event - preflare_irradiance) / preflare_irradiance * 100.0

            if verbose:
                jedi_config.logger.info("Event {0} irradiance converted from absolute to percent units.".format(flare_index))

            # Do flare removal in the light curves and add the results to the DataFrame
            #progress_bar_correction = progressbar.ProgressBar(widgets=[progressbar.FormatLabel('Peak match subtract: ')] + widgets,
            #                                                  max_value=len(ion_tuples)).start()
            time_correction = time.time()

            for i in range(len(ion_tuples)):
                light_curve_to_subtract_from_df = pd.DataFrame(eve_lines_event[ion_tuples[i][0]])
                light_curve_to_subtract_from_df.columns = ['irradiance']
                light_curve_to_subtract_with_df = pd.DataFrame(eve_lines_event[ion_tuples[i][1]])
                light_curve_to_subtract_with_df.columns = ['irradiance']

                if (light_curve_to_subtract_from_df.isnull().all().all()) or (light_curve_to_subtract_with_df.isnull().all().all()):
                    if verbose:
                        jedi_config.logger.info('Event {0} {1} correction skipped because all irradiances are NaN.'.format(flare_index, ion_permutations[i]))
                else:
                    light_curve_corrected, seconds_shift, scale_factor = light_curve_peak_match_subtract(light_curve_to_subtract_from_df,
                                                                                                         light_curve_to_subtract_with_df,
                                                                                                         pd.Timestamp((goes_flare_events['peak_time'][flare_index]).iso),
                                                                                                         plot_path_filename=output_path + 'Peak Subtractions/Event {0} {1}.png'.format(flare_index, ion_permutations[i]),
                                                                                                         verbose=verbose)

                    eve_lines_event[ion_permutations[i]] = light_curve_corrected
                    jedi_row[ion_permutations[i] + ' Correction Time Shift [s]'] = seconds_shift
                    jedi_row[ion_permutations[i] + ' Correction Scale Factor'] = scale_factor

                    plt.close('all')

                    if verbose:
                        jedi_config.logger.info('Event {0} flare removal correction complete'.format(flare_index))
                    #progress_bar_correction.update(i)

            #progress_bar_correction.finish()
            print('Time to do peak match subtract [s]: {0}'.format(time.time() - time_correction))

            # TODO: Update calculate_eve_fe_line_precision to compute for all emission lines, not just selected
            uncertainty = np.ones(len(eve_lines_event)) * 0.002545

            # TODO: Propagate uncertainty through light_curve_peak_match_subtract and store in eve_lines_event

            # Fit the light curves to reduce influence of noise on the parameterizations to come later
            #progress_bar_fitting = progressbar.ProgressBar(widgets=[progressbar.FormatLabel('Light curve fitting: ')] + widgets,
            #                                               max_value=len(eve_lines_event.columns)).start()
            time_fitting = time.time()

            for i, column in enumerate(eve_lines_event):
                if eve_lines_event[column].isnull().all().all():
                    if verbose:
                        jedi_config.logger.info('Event {0} {1} fitting skipped because all irradiances are NaN.'.format(flare_index, column))
                else:
                    eve_line_event = pd.DataFrame(eve_lines_event[column])
                    eve_line_event.columns = ['irradiance']
                    eve_line_event['uncertainty'] = uncertainty

                    fitting_path = output_path + 'Fitting/'
                    if not os.path.exists(fitting_path):
                        os.makedirs(fitting_path)

                    plt.close('all')
                    light_curve_fit_df, best_fit_gamma, best_fit_score = light_curve_fit(eve_line_event,
                                                                                         gamma=np.array([5e-8]),
                                                                                         plots_save_path='{0}Event {1} {2} '.format(fitting_path, flare_index, column),
                                                                                         verbose=verbose)
                    eve_lines_event[column] = light_curve_fit_df
                    jedi_row[column + ' Fitting Gamma'] = best_fit_gamma
                    jedi_row[column + ' Fitting Score'] = best_fit_score

                    if verbose:
                        jedi_config.logger.info('Event {0} {1} light curves fitted.'.format(flare_index, column))
                    #progress_bar_fitting.update(i)

            #progress_bar_fitting.finish()
            print('Time to do fitting [s]: {0}'.format(time.time() - time_fitting))

            # Save the dimming event data to disk for quicker restore
            jedi_row.to_hdf(processed_jedi_non_params_filename, 'jedi_row')
            eve_lines_event.to_hdf(processed_lines_filename, 'eve_lines_event')
        else:
            jedi_row = pd.read_hdf(processed_jedi_non_params_filename, 'jedi_row')
            eve_lines_event = pd.read_hdf(processed_lines_filename, 'eve_lines_event')
            if verbose:
                jedi_config.logger.info('Loading files {0} and {1} rather than processing again.'.format(processed_jedi_non_params_filename, processed_lines_filename))

        # Parameterize the light curves for dimming
        for column in eve_lines_event:

            # Null out all parameters
            depth_percent, depth_time = np.nan, np.nan
            slope_start_time, slope_end_time = np.nan, np.nan
            slope_min, slope_max, slope_mean = np.nan, np.nan, np.nan
            duration_seconds, duration_start_time, duration_end_time = np.nan, np.nan, np.nan

            # Determine whether to do the parameterizations or not
            if eve_lines_event[column].isnull().all().all():
                if verbose:
                    jedi_config.logger.info('Event {0} {1} parameterization skipped because all irradiances are NaN.'.format(flare_index, column))
            else:
                eve_line_event = pd.DataFrame(eve_lines_event[column])
                eve_line_event.columns = ['irradiance']

                # Determine dimming depth (if any)
                depth_path = output_path + 'Depth/'
                if not os.path.exists(depth_path):
                    os.makedirs(depth_path)

                plt.close('all')
                depth_percent, depth_time = determine_dimming_depth(eve_line_event,
                                                                    plot_path_filename='{0}Event {1} {2} Depth.png'.format(depth_path, flare_index, column),
                                                                    verbose=verbose, logger=jedi_config.logger)

                jedi_row[column + ' Depth [%]'] = depth_percent
                # jedi_row[column + ' Depth Uncertainty [%]'] = depth_uncertainty  # TODO: make determine_dimming_depth return the propagated uncertainty
                jedi_row[column + ' Depth Time'] = depth_time

                # Determine dimming slope (if any)
                slope_path = output_path + 'Slope/'
                if not os.path.exists(slope_path):
                    os.makedirs(slope_path)

                slope_start_time = pd.Timestamp((goes_flare_events['peak_time'][flare_index]).iso)
                slope_end_time = depth_time

                if (pd.isnull(slope_start_time)) or (pd.isnull(slope_end_time)):
                    if verbose:
                        jedi_config.logger.warning('Cannot compute slope or duration because slope bounding times NaN.')
                else:
                    plt.close('all')
                    slope_min, slope_max, slope_mean = determine_dimming_slope(eve_line_event,
                                                                               earliest_allowed_time=slope_start_time,
                                                                               latest_allowed_time=slope_end_time,
                                                                               plot_path_filename='{0}Event {1} {2} Slope.png'.format(slope_path, flare_index, column),
                                                                               verbose=verbose, logger=jedi_config.logger)

                    jedi_row[column + ' Slope Min [%/s]'] = slope_min
                    jedi_row[column + ' Slope Max [%/s]'] = slope_max
                    jedi_row[column + ' Slope Mean [%/s]'] = slope_mean
                    # jedi_row[column + ' Slope Uncertainty [%]'] = slope_uncertainty  # TODO: make determine_dimming_depth return the propagated uncertainty
                    jedi_row[column + ' Slope Start Time'] = slope_start_time
                    jedi_row[column + ' Slope End Time'] = slope_end_time

                    # Determine dimming duration (if any)
                    duration_path = output_path + 'Duration/'
                    if not os.path.exists(duration_path):
                        os.makedirs(duration_path)

                    plt.close('all')
                    duration_seconds, duration_start_time, duration_end_time = determine_dimming_duration(eve_line_event,
                                                                                                          earliest_allowed_time=slope_start_time,
                                                                                                          plot_path_filename='{0}Event {1} {2} Duration.png'.format(duration_path, flare_index, column),
                                                                                                          verbose=verbose, logger=jedi_config.logger)

                    jedi_row[column + ' Duration [s]'] = duration_seconds
                    jedi_row[column + ' Duration Start Time'] = duration_start_time
                    jedi_row[column + ' Duration End Time'] = duration_end_time

                if verbose:
                    jedi_config.logger.info("Event {0} {1} parameterizations complete.".format(flare_index, column))

                # Produce a summary plot for each light curve
                plt.style.use('jpm-transparent-light')

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
                                 arrowprops=dict(facecolor='dodgerblue', edgecolor='dodgerblue', linewidth=5, arrowstyle='<->'))
                    mid_time = duration_start_time + (duration_end_time - duration_start_time) / 2
                    plt.annotate(str(duration_seconds) + ' s', xy=(mid_time, 0), xycoords='data', ha='center', va='bottom', size=18, color='dodgerblue')

                summary_path = output_path + 'Summary Plots/'
                if not os.path.exists(summary_path):
                    os.makedirs(summary_path)
                summary_filename = '{0}Event {1} {2} Parameter Summary.png'.format(summary_path, flare_index, column)
                plt.savefig(summary_filename)
                plt.close('all')
                if verbose:
                    jedi_config.logger.info("Summary plot saved to %s" % summary_filename)

        # Write to the JEDI catalog on disk
        jedi_row.to_csv(csv_filename, header=False, index=False, mode='a')
        if verbose:
            jedi_config.logger.info('Event {0} JEDI row written to {1}.'.format(flare_index, csv_filename))

        # Update progress bar
        #progress_bar.update(flare_index)
        print('Total time for loop [s]: {0}'.format(time.time() - loop_time))

    #progress_bar.finish()

def map_flare_index_to_preflare_index(flare_index):
    """Internal-use function for translating the <5k preflare_indices to the ~5k flare_indices
    Why?
    When flares occur close together in time, they can't be considered independent. In these cases, the pre-flare
    irradiance for flare #2 can't be computed because flare #1 was still in progress. Instead, we use the pre-flare
    irradiance determined for flare #1. This can be true for flare #3, 4, 5, etc until a sufficient time gap occurs
    before the next flare and a "fresh" pre-flare irradiance can be computed. Since that time gap is fixed (it's a
    tuneable parameter in jedi_config) we can determine a priori which flares are independent and only compute the
    pre-flare irradiance for those. Thus, the array we end up with is smaller than the total number of flares. This
    function tells you which pre-flare irradiance index to access for the flare index you are currently processing.
    Thanks to Raphael Attie for conceiving of and implementing this logic, which can be easily parallelized and
    processed just a single time to speed up code execution.

    Inputs:
        flare_index [int]: The event identifier from the main loop.

    Optional Inputs:
        None

    Outputs:
        preflare_map_indices [numpy array]: Map of each flare_index to which preflare irradiance index to use.

    Optional Outputs:
        None

    Example:
        preflare_index = map_flare_index_to_preflare_index(flare_index)
    """
    is_independent_flare = jedi_config.all_minutes_since_last_flare > jedi_config.threshold_time_prior_flare_minutes
    irange = range(is_independent_flare)
    invalid_index = -1
    idx = np.where(is_independent_flare)[0]
    sidx = np.searchsorted(idx, irange, 'right')-1
    preflare_map_indices = np.where(sidx == -1, invalid_index, idx[sidx])
    return preflare_map_indices[flare_index]

def clip_eve_data_to_dimming_window(flare_index,
                                    verbose=False):
    """Clip all EVE data (4+ years) down to just the time range of interest for this particular event (~hours)

    Inputs:
        flare_index [int]: The identifier for which event in JEDI to process.

    Optional Inputs:
        verbose [bool]:     Set to log the processing messages to disk and console. Default is False.

    Outputs:
        eve_lines_event [pandas DataFrame]: The (39) EVE extracted emission lines (columns) trimmed in time (rows).

    Optional Outputs:
        None

    Example:
        eve_lines_event = clip_eve_data_to_dimming_window(flare_index,
                                                          verbose=verbose)
    """
    if verbose:
        jedi_config.logger.info("Clipping EVE data in time for event {0}.".format(flare_index))

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
    jedi_config.jedi_row.at[flare_index, 'Flare Interrupt'] = flare_interrupt

    if ((bracket_time_right - bracket_time_left).sec / 60.0) < jedi_config.threshold_minimum_dimming_window_minutes:
        # Leave all dimming parameters as NaN and write this null result to the CSV on disk

        # TODO: TO BE REVIEWED IF USING jedi_df as a ~5k x 24k dataframe!!!!!
        jedi_config.jedi_row.to_csv(jedi_config.jedi_csv_filename, header=False, index=False, mode='a')

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


def loop_light_curve_peak_match_subtract(eve_lines_event, jedi_row, flare_index, ion_tuples, ion_permutations,
                                         verbose=False):
    """Loop through all of the ion permutations, match the pairs of light curves at the flare peak and subtract them

    Inputs:
        eve_lines_event [pandas DataFrame]: The (39) EVE extracted emission lines (columns) trimmed in time (rows).
        jedi_row [pandas DataFrame]:        A ~24k column DataFrame with only a single row.
        flare_index [int]:                  The identifier for which event in JEDI to process.
        ion_tuples [list of tuples]:        Every pair-permutation of the 39 extracted EVE emission lines.
        ion_permutations [string array]:    Every pair-permutation of the 39 extracted EVE emission lines with "by" between them.

    Optional Inputs:
        verbose [bool]:     Set to log the processing messages to disk and console. Default is False.

    Outputs:
        No new outputs; appends to eve_lines_event and fills in jedi_row

    Optional Outputs:
        None

    Example:
        loop_light_curve_peak_match_subtract(eve_lines_event, jedi_row, flare_index, ion_tuples, ion_permutations,
                                             verbose=verbose)
    """
    if verbose:
        jedi_config.logger.info("Clipping EVE data in time for event {0}.".format(flare_index))

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

            eve_lines_event[ion_permutations[i]] = light_curve_corrected  # TODO: Verify that I don't have to pass this back, i.e., this changes the df outside of this function
            jedi_row[ion_permutations[i] + ' Correction Time Shift [s]'] = seconds_shift
            jedi_row[ion_permutations[i] + ' Correction Scale Factor'] = scale_factor

            plt.close('all')

            if jedi_config.verbose:
                jedi_config.logger.info('Event {0} flare removal correction complete'.format(flare_index))


def loop_light_curve_fit(eve_lines_event, jedi_row, flare_index, uncertainty,
                         verbose=False):
    """Loop through all of the light curves for an event (flare_index) and fit them

    Inputs:
        eve_lines_event [pandas DataFrame]: The (39) EVE extracted emission lines (columns) trimmed in time (rows).
        jedi_row [pandas DataFrame]:        A ~24k column DataFrame with only a single row.
        flare_index [int]:                  The identifier for which event in JEDI to process.
        uncertainty [numpy array]:          An array containing the uncertainty of each irradiance value. TODO: Needs to be properly populated.

    Optional Inputs:
        verbose [bool]:     Set to log the processing messages to disk and console. Default is False.

    Outputs:
        No new outputs; appends to eve_lines_event and fills in jedi_row

    Optional Outputs:
        None

    Example:
        loop_light_curve_fit(eve_lines_event, jedi_row, flare_index, uncertainty,
                             verbose=verbose)
    """
    if verbose:
        jedi_config.logger.info("Fitting light curves for event {0}.".format(flare_index))

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
                                                                                 plots_save_path='{0}Event {1} {2} '.format(fitting_path, flare_index, column),
                                                                                 verbose=jedi_config.verbose,
                                                                                 logger=jedi_config.logger,
                                                                                 n_jobs=jedi_config.n_jobs)
            eve_lines_event[column] = light_curve_fit_df  # TODO: Verify that I don't have to pass this back, i.e., this changes the df outside of this function
            jedi_row[column + ' Fitting Gamma'] = best_fit_gamma
            jedi_row[column + ' Fitting Score'] = best_fit_score

            if jedi_config.verbose:
                jedi_config.logger.info('Event {0} {1} light curves fitted.'.format(flare_index, column))


def determine_dimming_parameters(eve_lines_event, jedi_row, flare_index,
                                 verbose=False):
    """For every light curve, determine the dimming parameters (depth, slope, duration) wherever possible

    Inputs:
        eve_lines_event [pandas DataFrame]: The (39) EVE extracted emission lines (columns) trimmed in time (rows).
        jedi_row [pandas DataFrame]:        A ~24k column DataFrame with only a single row.
        flare_index [int]:                  The identifier for which event in JEDI to process.

    Optional Inputs:
        verbose [bool]:     Set to log the processing messages to disk and console. Default is False.

    Outputs:
        No new outputs; appends to eve_lines_event and fills in jedi_row

    Optional Outputs:
        None

    Example:
        determine_dimming_parameters(eve_lines_event, jedi_row, flare_index,
                                     verbose=verbose)
    """
    if verbose:
        jedi_config.logger.info("Fitting light curves for event {0}.".format(flare_index))

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
                                                                plot_path_filename='{0}Event {1} {2} Depth.png'.format(depth_path, flare_index, column),
                                                                verbose=jedi_config.verbose, logger=jedi_config.logger)

            jedi_row[column + ' Depth [%]'] = depth_percent  # TODO: Verify that I don't have to pass this back, i.e., this changes the df outside of this function
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
                                                                           plot_path_filename='{0}Event {1} {2} Slope.png'.format(slope_path, flare_index, column),
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
                                                                                                      plot_path_filename='{0}Event {1} {2} Duration.png'.format(duration_path, flare_index, column),
                                                                                                      verbose=jedi_config.verbose,
                                                                                                      logger=jedi_config.logger)

                jedi_row[column + ' Duration [s]'] = duration_seconds
                jedi_row[column + ' Duration Start Time'] = duration_start_time
                jedi_row[column + ' Duration End Time'] = duration_end_time

            if jedi_config.verbose:
                jedi_config.logger.info("Event {0} {1} parameterizations complete.".format(flare_index, column))


def produce_summary_plot(eve_lines_event, jedi_row, flare_index,
                         verbose=False):
    """Make a plot of the fitted light curve, annotated with every dimming parameter that could be determined

    Inputs:
        eve_lines_event [pandas DataFrame]: The (39) EVE extracted emission lines (columns) trimmed in time (rows).
        jedi_row [pandas DataFrame]:        A ~24k column DataFrame with only a single row.
        flare_index [int]:                  The identifier for which event in JEDI to process.

    Optional Inputs:
        verbose [bool]:     Set to log the processing messages to disk and console. Default is False.

    Outputs:
        Creates a .png file on disk for the plot

    Optional Outputs:
        None

    Example:
        produce_summary_plot(eve_lines_event, jedi_row, flare_index,
                             verbose=verbose)
    """
    if verbose:
        jedi_config.logger.info("Fitting light curves for event {0}.".format(flare_index))

    # Produce a summary plot for each light curve
    # plt.style.use('jpm-transparent-light')
    for column in eve_lines_event:
        eve_line_event = pd.DataFrame(eve_lines_event[column])
        eve_line_event.columns = ['irradiance']

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

        if not np.isnan(jedi_row[column + ' Depth [%]']):
            plt.annotate('', xy=(jedi_row[column + ' Depth Time'], -jedi_row[column + ' Depth [%]']), xycoords='data',
                         xytext=(jedi_row[column + ' Depth Time'], 0), textcoords='data',
                         arrowprops=dict(facecolor='limegreen', edgecolor='limegreen', linewidth=2))
            mid_depth = -jedi_row[column + ' Depth [%]'] / 2.0
            plt.annotate('{0:.2f} %'.format(jedi_row[column + ' Depth [%]']), xy=(jedi_row[column + ' Depth Time'], mid_depth), xycoords='data',
                         ha='right', va='center', rotation=90, size=18, color='limegreen')

        if not np.isnan(jedi_row[column + ' Slope Mean [%/s]']):
            p = plt.plot(eve_line_event[jedi_row[column + ' Slope Start Time']:jedi_row[column + ' Slope End Time']]['irradiance'], c='tomato')

            inverse_str = '$^{-1}$'
            plt.annotate('slope_min={0} % s{1}'.format(latex_float(jedi_row[column + ' Slope Min [%/s]']), inverse_str),
                         xy=(0.98, 0.12), xycoords='axes fraction', ha='right',
                         size=12, color=p[0].get_color())
            plt.annotate('slope_max={0} % s{1}'.format(latex_float(jedi_row[column + ' Slope Max [%/s]']), inverse_str),
                         xy=(0.98, 0.08), xycoords='axes fraction', ha='right',
                         size=12, color=p[0].get_color())
            plt.annotate('slope_mean={0} % s{1}'.format(latex_float(jedi_row[column + ' Slope Mean [%/s]']), inverse_str),
                         xy=(0.98, 0.04), xycoords='axes fraction', ha='right',
                         size=12, color=p[0].get_color())

        if not np.isnan(jedi_row[column + ' Duration [s]']):
            plt.annotate('', xy=(jedi_row[column + ' Duration Start Time'], 0), xycoords='data',
                         xytext=(jedi_row[column + ' Duration End Time'], 0), textcoords='data',
                         arrowprops=dict(facecolor='dodgerblue', edgecolor='dodgerblue', linewidth=5,
                                         arrowstyle='<->'))
            mid_time = jedi_row[column + ' Duration Start Time'] + (jedi_row[column + ' Duration End Time'] - jedi_row[column + ' Duration Start Time']) / 2
            plt.annotate(str(jedi_row[column + ' Duration [s]']) + ' s', xy=(mid_time, 0), xycoords='data', ha='center', va='bottom',
                         size=18, color='dodgerblue')

        summary_path = jedi_config.output_path + 'Summary Plots/'
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        summary_filename = '{0}Event {1} {2} Parameter Summary.png'.format(summary_path, flare_index, column)
        plt.savefig(summary_filename)
        plt.close('all')

        if jedi_config.verbose:
            jedi_config.logger.info("Summary plot saved to %s" % summary_filename)


def merge_jedi_catalog_files(file_path='/Users/jmason86/Dropbox/Research/Postdoc_NASA/Analysis/Coronal Dimming Analysis/JEDI Catalog/',
                             verbose=False):
    """Function for merging the .csv output files of generate_jedi_catalog()

    Inputs:
        None.

    Optional Inputs:
        file_path [str]: Set to a path for saving the JEDI catalog table.
                         Default is '/Users/jmason86/Dropbox/Research/Postdoc_NASA/Analysis/Coronal Dimming Analysis/JEDI Catalog/'.
        verbose [bool]:  Set to log the processing messages to console. Default is False.

    Outputs:
        No direct return, but writes a csv to disk with the dimming parameterization results.

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


if __name__ == '__main__':
    # Parallel processing method
    # with mp.Pool(processes=6) as pool:
    #     for events in range(78, 148, 5):
    #         generate_jedi_catalog_function_1_varying_input = partial(generate_jedi_catalog, verbose=True)
    #         print('Should be running from {0} to {1}'.format(events, events + 5))
    #         pool.map(generate_jedi_catalog, range(events, events + 5))
    
    # Just run code over some range
    generate_jedi_catalog(range(1, 4))

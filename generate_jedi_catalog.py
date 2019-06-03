# Standard modules
import os
import numpy as np
import matplotlib as mpl
#mpl.use('macosx')  # For interactive plotting
mpl.use('agg')
from matplotlib import dates
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
import time

# Custom modules
from jpm_number_printing import latex_float
from lat_lon_to_position_angle import lat_lon_to_position_angle
from determine_preflare_irradiance import multiprocess_preflare_irradiance
from light_curve_peak_match_subtract import light_curve_peak_match_subtract
from light_curve_fit import light_curve_fit
from determine_dimming_depth import determine_dimming_depth
from determine_dimming_slope import determine_dimming_slope
from determine_dimming_duration import determine_dimming_duration

# Configuration
import jedi_config

# Global variables
jedi_row = pd.DataFrame()

__author__ = 'James Paul Mason'
__contact__ = 'jmason86@gmail.com'


def generate_jedi_catalog(flare_index_range=range(0, 5052),
                          compute_new_preflare_irradiances=False):
    """Wrapper code for creating James's Extreme Ultraviolet Variability Experiment (EVE) Dimming Index (JEDI) catalog.

    Inputs:
        flare_index_range [range]: The range of GOES flare indices to process. Default is range(0, 5052).

    Optional Inputs:
        compute_new_preflare_irradiances [bool]: Set to force reprocessing of pre-flare irradiances. Will also occur if preflare file doesn't exist on disk.

    Outputs:
        No direct return, but writes a csv to disk with the dimming parameterization results.
        Subroutines also optionally save processing plots to disk in jedi_config.output_path.

    Optional Outputs:
        None

    Example:
        generate_jedi_catalog()
    """

    # Force flare_index_range to be an array type so it can be indexed in later code
    if isinstance(flare_index_range, int):
        flare_index_range = np.array([flare_index_range])

    # Set up folders, load and clean data
    jedi_config.init()

    # Define the columns of the JEDI catalog
    global jedi_row
    jedi_row = jedi_config.init_jedi_row()
    jedi_config.write_new_jedi_file_to_disk(jedi_row)

    if jedi_config.verbose:
        jedi_config.logger.info('Created JEDI row definition.')

    # Compute all pre-flare irradiances if needed or because kwarg set
    if compute_new_preflare_irradiances or (os.path.isfile(jedi_config.preflare_csv_filename) is False):
        jedi_config.logger.info('Recomputing pre-flare irradiances.')
        preflare_irradiances, \
            preflare_windows_start, \
            preflare_windows_end = multiprocess_preflare_irradiance()
        jedi_config.logger.info('Finished processing pre-flare irradiances. Writing them to disk.')
        preflare_df = pd.DataFrame()
        preflare_df['Pre-Flare Start Time'] = preflare_windows_start
        preflare_df['Pre-Flare End Time'] = preflare_windows_end
        preflare_irradiance_column_names = jedi_config.eve_lines.columns + ' Pre-Flare Irradiance [W/m2]'
        preflare_df = preflare_df.join(pd.DataFrame(columns=preflare_irradiance_column_names))
        preflare_df[preflare_irradiance_column_names] = preflare_irradiances
        preflare_df.to_csv(jedi_config.preflare_csv_filename, index=None, mode='w')
        jedi_config.logger.info('Finished writing pre-flare irradiances to disk.')
    else:
        preflare_df = pd.read_csv(jedi_config.preflare_csv_filename, index_col=None)

    # Start loop through all flares
    for flare_index in flare_index_range:
        loop_time = time.time()

        # Skip event 0 to avoid problems with referring to earlier indices
        if flare_index == 0:
            continue

        jedi_config.logger.info('Running on event {0}'.format(flare_index))

        # Reinitalize jedi_row (faster and less buggy than setting all values to np.nan)
        jedi_row = jedi_config.init_jedi_row()

        # Fill the GOES flare information into the JEDI row
        jedi_row['Event #'] = flare_index
        jedi_row['GOES Flare Start Time'] = jedi_config.goes_flare_events['start_time'][flare_index].iso
        jedi_row['GOES Flare Peak Time'] = jedi_config.goes_flare_events['peak_time'][flare_index].iso
        jedi_row['GOES Flare Class'] = jedi_config.goes_flare_events['class'][flare_index]
        jedi_row['Flare Latitude [deg]'] = jedi_config.goes_flare_events.latitude[flare_index][0]
        jedi_row['Flare Longitude [deg]'] = jedi_config.goes_flare_events.longitude[flare_index][0]
        jedi_row['Flare Position Angle [deg]'] = lat_lon_to_position_angle(jedi_row['Flare Latitude [deg]'].values[0], jedi_row['Flare Longitude [deg]'].values[0])
        if jedi_config.verbose:
            jedi_config.logger.info("Event {0} GOES flare details stored to JEDI row.".format(flare_index))

        # Only do pre-parameterization processing if it hasn't been done already (check if files exist on disk)
        processed_jedi_non_params_filename = jedi_config.output_path + 'Processed Pre-Parameterization Data/Event {0} Pre-Parameterization.h5'.format(flare_index)
        processed_lines_filename = jedi_config.output_path + 'Processed Lines Data/Event {0} Lines.h5'.format(flare_index)
        if not os.path.isfile(processed_lines_filename) or not os.path.isfile(processed_jedi_non_params_filename):
            jedi_row["Pre-Flare Start Time"] = preflare_df['Pre-Flare Start Time'].iloc[map_flare_index_to_preflare_index(flare_index)]
            jedi_row["Pre-Flare End Time"] = preflare_df['Pre-Flare End Time'].iloc[map_flare_index_to_preflare_index(flare_index)]
            preflare_irradiance_cols = [col for col in jedi_row.columns if 'Pre-Flare Irradiance' in col]
            jedi_row[preflare_irradiance_cols] = preflare_df[preflare_irradiance_cols].iloc[map_flare_index_to_preflare_index(flare_index)].values

            if jedi_config.verbose:
                jedi_config.logger.info("Event {0} pre-flare irradiances stored to JEDI row.".format(flare_index))

            # Clip EVE data to dimming window
            eve_lines_event = clip_eve_data_to_dimming_window(flare_index)
            if eve_lines_event is False:
                continue

            # Convert irradiance units to percent (in place, don't care about absolute units from this point forward)
            preflare_irradiances = preflare_df.iloc[map_flare_index_to_preflare_index(flare_index)].filter(regex="\d").values
            eve_lines_event = (eve_lines_event - preflare_irradiances) / preflare_irradiances * 100.0

            if jedi_config.verbose:
                jedi_config.logger.info("Event {0} irradiance converted from absolute to percent units.".format(flare_index))

            # Do flare removal in the light curves and add the results to the DataFrame
            time_correction = time.time()
            loop_light_curve_peak_match_subtract(eve_lines_event, flare_index)
            print('Time to do peak match subtract [s]: {0}'.format(time.time() - time_correction))

            # TODO: Update calculate_eve_fe_line_precision to compute for all emission lines, not just selected
            uncertainty = np.ones(len(eve_lines_event)) * 0.002545

            # TODO: Propagate uncertainty through light_curve_peak_match_subtract and store in eve_lines_event

            # Fit the light curves to reduce influence of noise on the parameterizations to come later
            time_fitting = time.time()
            loop_light_curve_fit(eve_lines_event, flare_index, uncertainty)
            print('Time to do fitting [s]: {0}'.format(time.time() - time_fitting))

            # Save the dimming event data to disk for quicker restore
            jedi_row.to_hdf(processed_jedi_non_params_filename, 'jedi_row')
            eve_lines_event.to_hdf(processed_lines_filename, 'eve_lines_event')
        else:
            jedi_row = pd.read_hdf(processed_jedi_non_params_filename, 'jedi_row')
            eve_lines_event = pd.read_hdf(processed_lines_filename, 'eve_lines_event')
            if jedi_config.verbose:
                jedi_config.logger.info('Loading files {0} and {1} rather than processing again.'.format(processed_jedi_non_params_filename, processed_lines_filename))

        # Parameterize the light curves for dimming
        determine_dimming_parameters(eve_lines_event, flare_index)

        # Produce a summary plot for each light curve
        produce_summary_plot(eve_lines_event, flare_index)

        # Write to the JEDI catalog on disk
        jedi_row.to_hdf('{0} Event {1}.h5'.format(jedi_config.jedi_hdf_filename, flare_index), key='jedi_row', mode='w')
        if jedi_config.verbose:
            jedi_config.logger.info('Event {0} JEDI row written to {1}.'.format(jedi_config.jedi_hdf_filename, flare_index))

        print('Total time for loop [s]: {0}'.format(time.time() - loop_time))


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
        preflare_index [np.int64]: The index in the pre-flare irradiance array to use for the given flare_index.

    Optional Outputs:
        None

    Example:
        preflare_index = map_flare_index_to_preflare_index(flare_index)
    """
    is_independent_flare = jedi_config.all_minutes_since_last_flare > jedi_config.threshold_time_prior_flare_minutes
    all_flare_indices = range(0, is_independent_flare.size)
    independent_flare_indices = np.where(is_independent_flare)[0]
    sidx = np.searchsorted(independent_flare_indices, all_flare_indices, 'right') - 1
    return sidx[flare_index]


def clip_eve_data_to_dimming_window(flare_index):
    """Clip all EVE data (4+ years) down to just the time range of interest for this particular event (~hours)

    Inputs:
        flare_index [int]: The identifier for which event in JEDI to process.

    Optional Inputs:
        None.

    Outputs:
        eve_lines_event [pandas DataFrame]: The (39) EVE extracted emission lines (columns) trimmed in time (rows).

    Optional Outputs:
        None.

    Example:
        eve_lines_event = clip_eve_data_to_dimming_window(flare_index)
    """
    if jedi_config.verbose:
        jedi_config.logger.info("Clipping EVE data in time for event {0}.".format(flare_index))

    flare_interrupt = False

    # Clip EVE data to dimming window
    bracket_time_left = (jedi_config.goes_flare_events['peak_time'][flare_index] + (jedi_config.dimming_window_relative_to_flare_minutes_left * u.minute))
    next_flare_time = jedi_config.goes_flare_events['peak_time'][flare_index + 1]
    user_choice_time = (jedi_config.goes_flare_events['peak_time'][flare_index] + (jedi_config.dimming_window_relative_to_flare_minutes_right * u.minute))
    bracket_time_right = min(next_flare_time, user_choice_time)

    # If flare is shortening the window, set the flare_interrupt flag
    if bracket_time_right == next_flare_time:
        flare_interrupt = True
        if jedi_config.verbose:
            jedi_config.logger.info('Flare interrupt for event at {0} by flare at {1}'.format(jedi_config.goes_flare_events['peak_time'][flare_index].iso, next_flare_time))

    # Write flare_interrupt to JEDI row
    jedi_row['Flare Interrupt'] = flare_interrupt

    if ((bracket_time_right - bracket_time_left).sec / 60.0) < jedi_config.threshold_minimum_dimming_window_minutes:
        # Leave all dimming parameters as NaN and write this null result to the disk
        jedi_row.to_hdf('{0} Event {1}.h5'.format(jedi_config.jedi_hdf_filename, flare_index), key='jedi_row', mode='w')

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


def loop_light_curve_peak_match_subtract(eve_lines_event, flare_index):
    """Loop through all of the ion permutations, match the pairs of light curves at the flare peak and subtract them

    Inputs:
        eve_lines_event [pandas DataFrame]: The (39) EVE extracted emission lines (columns) trimmed in time (rows).
        flare_index [int]:                  The identifier for which event in JEDI to process.

    Optional Inputs:
        None.

    Outputs:
        No new outputs; appends to eve_lines_event and fills in jedi_row

    Optional Outputs:
        None.

    Example:
        loop_light_curve_peak_match_subtract(eve_lines_event, flare_index)
    """
    if jedi_config.verbose:
        jedi_config.logger.info("Clipping EVE data in time for event {0}.".format(flare_index))

    for i in range(len(jedi_config.ion_tuples)):
        light_curve_to_subtract_from_df = pd.DataFrame(eve_lines_event[jedi_config.ion_tuples[i][0]])
        light_curve_to_subtract_from_df.columns = ['irradiance']
        light_curve_to_subtract_with_df = pd.DataFrame(eve_lines_event[jedi_config.ion_tuples[i][1]])
        light_curve_to_subtract_with_df.columns = ['irradiance']

        if (light_curve_to_subtract_from_df.isnull().all().all()) or (light_curve_to_subtract_with_df.isnull().all().all()):
            if jedi_config.verbose:
                jedi_config.logger.warning(
                    'Event {0} {1} correction skipped because all irradiances are NaN.'.format(flare_index,
                                                                                               jedi_config.ion_permutations[i]))
        else:
            light_curve_corrected, seconds_shift, scale_factor = light_curve_peak_match_subtract(light_curve_to_subtract_from_df,
                                                                                                 light_curve_to_subtract_with_df,
                                                                                                 pd.Timestamp((jedi_config.goes_flare_events['peak_time'][flare_index]).iso),
                                                                                                 plot_path_filename=jedi_config.output_path + 'Peak Subtractions/Event {0} {1}.png'.format(flare_index, jedi_config.ion_permutations[i]))

            eve_lines_event[jedi_config.ion_permutations[i]] = light_curve_corrected
            jedi_row[jedi_config.ion_permutations[i] + ' Correction Time Shift [s]'] = seconds_shift
            jedi_row[jedi_config.ion_permutations[i] + ' Correction Scale Factor'] = scale_factor

            plt.close('all')

            if jedi_config.verbose:
                jedi_config.logger.info('Event {0} flare removal correction complete'.format(flare_index))


def loop_light_curve_fit(eve_lines_event, flare_index, uncertainty):
    """Loop through all of the light curves for an event (flare_index) and fit them

    Inputs:
        eve_lines_event [pandas DataFrame]: The (39) EVE extracted emission lines (columns) trimmed in time (rows).
        flare_index [int]:                  The identifier for which event in JEDI to process.
        uncertainty [numpy array]:          An array containing the uncertainty of each irradiance value. TODO: Needs to be properly populated.

    Optional Inputs:
        None.

    Outputs:
        No new outputs; appends to eve_lines_event and fills in jedi_row

    Optional Outputs:
        None.

    Example:
        loop_light_curve_fit(eve_lines_event, jedi_row, flare_index, uncertainty)
    """
    if jedi_config.verbose:
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

            plt.close('all')
            light_curve_fit_df, best_fit_gamma, best_fit_score = light_curve_fit(eve_line_event,
                                                                                 gamma=np.array([5e-8]),
                                                                                 plots_save_path='{0}Event {1} {2}'.format(fitting_path, flare_index, column))
            eve_lines_event[column] = light_curve_fit_df
            jedi_row[column + ' Fitting Gamma'] = best_fit_gamma
            jedi_row[column + ' Fitting Score'] = best_fit_score

            if jedi_config.verbose:
                jedi_config.logger.info('Event {0} {1} light curves fitted.'.format(flare_index, column))


def determine_dimming_parameters(eve_lines_event, flare_index):
    """For every light curve, determine the dimming parameters (depth, slope, duration) wherever possible

    Inputs:
        eve_lines_event [pandas DataFrame]: The (39) EVE extracted emission lines (columns) trimmed in time (rows).
        flare_index [int]:                  The identifier for which event in JEDI to process.

    Optional Inputs:
        None.

    Outputs:
        No new outputs; appends to eve_lines_event and fills in jedi_row

    Optional Outputs:
        None.

    Example:
        determine_dimming_parameters(eve_lines_event, flare_index)
    """
    if jedi_config.verbose:
        jedi_config.logger.info("Fitting light curves for event {0}.".format(flare_index))

    for column in eve_lines_event:
        # Null out all parameters
        depth_first, depth_first_time, depth_max, depth_max_time = np.nan, np.nan, np.nan, np.nan
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

            plt.close('all')
            depth_first, depth_first_time, depth_max, depth_max_time = determine_dimming_depth(eve_line_event,
                                                                                               plot_path_filename='{0}Event {1} {2} Depth.png'.format(depth_path, flare_index, column))

            # Make sure times haven't become NaT instead of NaN
            depth_first_time = valid_time(depth_first_time)
            depth_max_time = valid_time(depth_max_time)

            jedi_row[column + ' Depth First [%]'] = depth_first
            jedi_row[column + ' Depth First Time'] = depth_first_time
            jedi_row[column + ' Depth Max [%]'] = depth_max
            jedi_row[column + ' Depth Max Time'] = depth_max_time
            # jedi_row[column + ' Depth Uncertainty [%]'] = depth_uncertainty  # TODO: make determine_dimming_depth return the propagated uncertainty

            # Determine dimming slope (if any)
            slope_path = jedi_config.output_path + 'Slope/'

            slope_start_time = pd.Timestamp((jedi_config.goes_flare_events['peak_time'][flare_index]).iso)
            slope_end_time = depth_first_time

            if (pd.isnull(slope_start_time)) or (pd.isnull(slope_end_time)):
                if jedi_config.verbose:
                    jedi_config.logger.warning('Cannot compute slope or duration because slope bounding times NaN.')
            else:
                plt.close('all')
                slope_min, slope_max, slope_mean = determine_dimming_slope(eve_line_event,
                                                                           earliest_allowed_time=slope_start_time,
                                                                           latest_allowed_time=slope_end_time,
                                                                           plot_path_filename='{0}Event {1} {2} Slope.png'.format(slope_path, flare_index, column))

                # Make sure times haven't become NaT instead of NaN
                slope_start_time = valid_time(slope_start_time)
                slope_end_time = valid_time(slope_end_time)

                jedi_row[column + ' Slope Min [%/s]'] = slope_min
                jedi_row[column + ' Slope Max [%/s]'] = slope_max
                jedi_row[column + ' Slope Mean [%/s]'] = slope_mean
                # jedi_row[column + ' Slope Uncertainty [%]'] = slope_uncertainty  # TODO: make determine_dimming_depth return the propagated uncertainty
                jedi_row[column + ' Slope Start Time'] = slope_start_time
                jedi_row[column + ' Slope End Time'] = slope_end_time

                # Determine dimming duration (if any)
                duration_path = jedi_config.output_path + 'Duration/'

                plt.close('all')
                duration_seconds, duration_start_time, duration_end_time = determine_dimming_duration(eve_line_event,
                                                                                                      earliest_allowed_time=slope_start_time,
                                                                                                      plot_path_filename='{0}Event {1} {2} Duration.png'.format(duration_path, flare_index, column))

                # Make sure times haven't become NaT instead of NaN
                duration_start_time = valid_time(duration_start_time)
                duration_end_time = valid_time(duration_end_time)

                jedi_row[column + ' Duration [s]'] = duration_seconds
                jedi_row[column + ' Duration Start Time'] = duration_start_time
                jedi_row[column + ' Duration End Time'] = duration_end_time

            if jedi_config.verbose:
                jedi_config.logger.info("Event {0} {1} parameterizations complete.".format(flare_index, column))


def valid_time(time_to_check):
    """Forces numpy NaTs to be numpy NaNs, otherwise does nothing"""
    if isinstance(time_to_check, np.datetime64):
        if np.isnat(time_to_check):
            return np.nan
    return time_to_check


def produce_summary_plot(eve_lines_event, flare_index):
    """Make a plot of the fitted light curve, annotated with every dimming parameter that could be determined

    Inputs:
        eve_lines_event [pandas DataFrame]: The (39) EVE extracted emission lines (columns) trimmed in time (rows).
        flare_index [int]:                  The identifier for which event in JEDI to process.

    Optional Inputs:
        None.

    Outputs:
        Creates a .png file on disk for the plot

    Optional Outputs:
        None.

    Example:
        produce_summary_plot(eve_lines_event, flare_index)
    """
    if jedi_config.verbose:
        jedi_config.logger.info("Fitting light curves for event {0}.".format(flare_index))

    # Produce a summary plot for each light curve
    for column in eve_lines_event:
        if eve_lines_event[column].isnull().all().all():
            continue

        eve_line_event = pd.DataFrame(eve_lines_event[column])
        eve_line_event.columns = ['irradiance']

        # Extract the parameters to simplify multiple calls below
        depth_first = jedi_row[column + ' Depth First [%]'].values[0]
        depth_first_time = jedi_row[column + ' Depth First Time'].values[0]
        depth_max = jedi_row[column + ' Depth Max [%]'].values[0]
        depth_max_time = jedi_row[column + ' Depth Max Time'].values[0]
        slope_min = jedi_row[column + ' Slope Min [%/s]'].values[0]
        slope_max = jedi_row[column + ' Slope Max [%/s]'].values[0]
        slope_mean = jedi_row[column + ' Slope Mean [%/s]'].values[0]
        slope_start_time = jedi_row[column + ' Slope Start Time'].values[0]
        slope_end_time = jedi_row[column + ' Slope End Time'].values[0]
        duration_seconds = jedi_row[column + ' Duration [s]'].values[0]
        duration_start_time = jedi_row[column + ' Duration Start Time'].values[0]
        duration_end_time = jedi_row[column + ' Duration End Time'].values[0]

        if type(duration_end_time) is np.datetime64:
            plot_window_end_time = duration_end_time + np.timedelta64(1, 'h')
        elif type(depth_first_time) is np.datetime64:
            plot_window_end_time = depth_first_time + np.timedelta64(1, 'h')
        else:
            plot_window_end_time = eve_line_event.index.values[-1]

        plt.close('all')
        ax = eve_line_event['irradiance'].plot(color='black')
        plt.xlim(jedi_row['GOES Flare Start Time'].values[0], plot_window_end_time)
        plt.axhline(linestyle='dashed', color='grey')
        start_date = jedi_row['GOES Flare Start Time'].values[0]
        start_date_string = pd.to_datetime(str(start_date))
        plt.xlabel(start_date_string.strftime('%Y-%m-%d %H:%M:%S'))
        plt.ylabel('Irradiance [%]')
        fmtr = dates.DateFormatter("%H:%M:%S")
        ax.xaxis.set_major_formatter(fmtr)
        ax.xaxis.set_major_locator(dates.HourLocator())
        plt.title('Event {0} {1} nm Parameters'.format(flare_index, column))

        if not np.isnan(depth_first):
            plt.annotate('', xy=(depth_first_time, -depth_first), xycoords='data',
                         xytext=(depth_first_time, 0), textcoords='data',
                         arrowprops=dict(facecolor='limegreen', edgecolor='limegreen', linewidth=2))
            mid_depth = -depth_first / 2.0
            plt.annotate('{0:.2f} %'.format(depth_first), xy=(depth_first_time, mid_depth), xycoords='data',
                         ha='right', va='center', rotation=90, size=18, color='limegreen')

            if depth_max != depth_first:
                plt.annotate('', xy=(depth_max_time, -depth_max), xycoords='data',
                             xytext=(depth_max_time, 0), textcoords='data',
                             arrowprops=dict(facecolor='limegreen', edgecolor='limegreen', linewidth=2))
                mid_depth = -depth_max / 2.0
                plt.annotate('{0:.2f} %'.format(depth_max), xy=(depth_max_time, mid_depth), xycoords='data',
                             ha='right', va='center', rotation=90, size=18, color='limegreen')

        if not np.isnan(slope_mean):
            p = plt.plot(eve_line_event[slope_start_time:slope_end_time]['irradiance'], c='tomato')

            inverse_str = '$^{-1}$'
            plt.annotate('slope_min={0} % s{1}'.format(latex_float(slope_min), inverse_str),
                         xy=(0.98, 0.88), xycoords='axes fraction', ha='right',
                         size=12, color=p[0].get_color())
            plt.annotate('slope_max={0} % s{1}'.format(latex_float(slope_max), inverse_str),
                         xy=(0.98, 0.84), xycoords='axes fraction', ha='right',
                         size=12, color=p[0].get_color())
            plt.annotate('slope_mean={0} % s{1}'.format(latex_float(slope_mean), inverse_str),
                         xy=(0.98, 0.80), xycoords='axes fraction', ha='right',
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
        summary_filename = '{0}Event {1} {2} Parameter Summary.png'.format(summary_path, flare_index, column)
        plt.savefig(summary_filename)
        plt.close('all')

        if jedi_config.verbose:
            jedi_config.logger.info("Summary plot saved to %s" % summary_filename)


def merge_jedi_catalog_files(file_path='/Users/jmason86/Dropbox/Research/Postdoc_NASA/Analysis/Coronal Dimming Analysis/JEDI Catalog/'):
    """Function for merging the .csv output files of generate_jedi_catalog()

    Inputs:
        None.

    Optional Inputs:
        file_path [str]: Set to a path for saving the JEDI catalog table.
                         Default is '/Users/jmason86/Dropbox/Research/Postdoc_NASA/Analysis/Coronal Dimming Analysis/JEDI Catalog/'.

    Outputs:
        No direct return, but writes a csv to disk with the dimming parameterization results.

    Optional Outputs:
        None.

    Example:
        merge_jedi_catalog_files()
    """
    # Create one sorted, clean dataframe from all of the csv files
    list_dfs = []
    for file in os.listdir(file_path):
        #if file.endswith(".h5") and "merged" not in file:
        if file.endswith(".h5") and "merged" in file:
            #flare_index = int(file.split()[-1].split('.')[0])
            #if flare_index >= 2500:  # This is only here because the full JEDI merge is too big to do all at once with only 16 GB of RAM
            #jedi_rows = pd.read_hdf(os.path.join(file_path, file), 'jedi_row')
            jedi_rows = pd.read_hdf(os.path.join(file_path, file), 'jedi')
            list_dfs.append(jedi_rows)
    jedi_catalog_df = pd.concat(list_dfs, ignore_index=True)
    jedi_catalog_df.dropna(axis=0, how='all', inplace=True)
    jedi_catalog_df.drop_duplicates(inplace=True)
    jedi_catalog_df.sort_values(by=['Event #'], inplace=True)
    jedi_catalog_df.reset_index(drop=True, inplace=True)
    jedi_config.init()
    jedi_row_standard = jedi_config.init_jedi_row()
    cols = jedi_row_standard.columns.tolist()
    jedi_catalog_df = jedi_catalog_df[cols]
    jedi_catalog_df = jedi_catalog_df.apply(pd.to_numeric, errors='ignore')
    if jedi_config.verbose:
        print("Read files, sorted, dropped empty and duplicate rows, and reset index.")

    # Write the catalog to disk
    hdf_filename = file_path + 'jedi_merged_{0}.h5'.format(Time.now().iso)
    jedi_catalog_df.to_hdf(hdf_filename, key='jedi', mode='w')
    if jedi_config.verbose:
        print("Wrote merged file to {0}".format(hdf_filename))

    return 1


if __name__ == '__main__':
    generate_jedi_catalog(range(32, 5052))
    #merge_jedi_catalog_files()

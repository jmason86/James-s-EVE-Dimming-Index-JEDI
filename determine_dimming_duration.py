# Standard modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom modules
import jedi_config

__author__ = 'James Paul Mason'
__contact__ = 'jmason86@gmail.com'


def determine_dimming_duration(light_curve_df,
                               earliest_allowed_time=None, smooth_points=0,
                               plot_path_filename=None):
    """Find the duration of dimming in a light curve, if any.
    Assumes light curve is normalized such that pre-flare = 0%.

    Inputs:
        light_curve_df [pd DataFrame]:    A pandas DataFrame with a DatetimeIndex and a column for irradiance.

    Optional Inputs:
        earliest_allowed_time [metatime]: The function won't return a duration if the only 0 crossings are earlier than this.
                                          Default is None, meaning the beginning of the light_curve_df.
        smooth_points [integer]:          Used to apply a rolling mean with the number of points (indices) specified.
                                          Default is 0, meaning no smoothing will be performed.
        plot_path_filename [str]:         Set to a path and filename in order to save the summary plot to disk.
                                          Default is None, meaning the plot will not be saved to disk.

    Outputs:
        duration_seconds [integer]:         The duration of dimming in seconds.
        duration_start_time [pd.Timestamp]: The time the duration starts (downward 0 crossing).
        duration_end_time [pd.Timestamp]:   The time the duration ends (upward 0 crossing).

    Optional Outputs:
        None

    Example:
        duration_seconds, duration_start_time, duration_end_time = determine_dimming_duration(light_curve_df,
                                                                                              plot_path_filename='./bla.png')
    """
    # If no earliest_allowed_time set, then set it to beginning of light_curve_df
    if not earliest_allowed_time:
        earliest_allowed_time = pd.Timestamp(light_curve_df.index.values[0])

    if jedi_config.verbose:
        jedi_config.logger.info("Running on event with light curve start time of {0}.".format(light_curve_df.index[0]))

    # Set up a successful processing flag
    found_duration = True

    # Optionally smooth the light curve with a rolling mean
    if smooth_points:
        light_curve_df['smooth'] = light_curve_df.rolling(smooth_points, center=True).mean()
    else:
        light_curve_df['smooth'] = light_curve_df['irradiance']

    first_non_nan = light_curve_df['smooth'].first_valid_index()
    nan_indices = np.isnan(light_curve_df['smooth'])
    light_curve_df['smooth'][nan_indices] = light_curve_df['smooth'][first_non_nan]

    # Find the indices where the light curve is closest to 0
    zero_crossing_indices = np.where(np.diff(np.signbit(light_curve_df['smooth'])))[0]
    zero_crossing_times = light_curve_df.index[zero_crossing_indices]

    # Discard any indices prior to the user-provided earliest_allowed_time, else cannot compute
    zero_crossing_indices = zero_crossing_indices[zero_crossing_times > earliest_allowed_time]
    if zero_crossing_indices.size == 0:
        if jedi_config.verbose:
            jedi_config.logger.warning('No zero crossings detected after earliest allowed time of %s' % earliest_allowed_time)
        found_duration = False

    # Figure out which way the light curve is sloping
    if found_duration:
        light_curve_df['diff'] = light_curve_df['smooth'].diff()

    # Find the first negative slope zero crossing time
    if found_duration:
        neg_zero_crossing_indices = np.where(light_curve_df['diff'][zero_crossing_indices + 1] < 0)[0]
        if len(neg_zero_crossing_indices) > 0:
            first_neg_zero_crossing_index = neg_zero_crossing_indices[0]
            first_neg_zero_crossing_time = light_curve_df.index[zero_crossing_indices[first_neg_zero_crossing_index]]
        else:
            if jedi_config.verbose:
                jedi_config.logger.warning('No negative slope 0-crossing found. Duration cannot be defined.')
            found_duration = False

    # Find the first subsequent positive slope zero crossing
    if found_duration:
        pos_zero_crossing_indices = np.where(light_curve_df['diff'][zero_crossing_indices + 1] > 0)[0]
        subsequent_pos_zero_crossing_indices = pos_zero_crossing_indices[np.where(light_curve_df.iloc[zero_crossing_indices[pos_zero_crossing_indices]].index > first_neg_zero_crossing_time)[0]]
        if len(subsequent_pos_zero_crossing_indices) > 0:
            first_pos_zero_crossing_index = subsequent_pos_zero_crossing_indices[0]
            first_pos_zero_crossing_time = light_curve_df.index[zero_crossing_indices[first_pos_zero_crossing_index]]
        else:
            if jedi_config.verbose:
                jedi_config.logger.warning('No positive slope 0-crossing found. Duration cannot be defined.')
            found_duration = False

    # If the first negative slope zero crossing isn't earlier than the positive one, return null
    if (found_duration) and (first_neg_zero_crossing_time > first_pos_zero_crossing_time):
        if jedi_config.verbose:
            jedi_config.logger.warning('Dimming light curve may be misaligned in window. Negative slope 0-crossing detected after positive one.')
        found_duration = False

    # Return the time difference in seconds between the selected zero crossings
    if found_duration:
        duration_seconds = int((first_pos_zero_crossing_time - first_neg_zero_crossing_time).total_seconds())

    if plot_path_filename:
        plt.style.use('jpm-transparent-light')
        plt.close('all')
        from matplotlib import dates

        if found_duration:
            light_curve_df = light_curve_df.drop('diff', 1)

        ax = light_curve_df['smooth'].plot()
        start_date = light_curve_df.index.values[0]
        start_date_string = pd.to_datetime(str(start_date))
        plt.xlabel(start_date_string.strftime('%Y-%m-%d %H:%M:%S'))
        plt.ylabel('Irradiance [%]')
        fmtr = dates.DateFormatter("%H:%M:%S")
        ax.xaxis.set_major_formatter(fmtr)
        ax.xaxis.set_major_locator(dates.HourLocator())
        plt.title('Dimming Duration')

        if found_duration:
            plt.scatter([first_neg_zero_crossing_time, first_pos_zero_crossing_time],
                        [light_curve_df['smooth'].loc[first_neg_zero_crossing_time],
                         light_curve_df['smooth'].loc[first_pos_zero_crossing_time]],
                        c='black', s=300, zorder=3)
            plt.annotate('', xy=(first_neg_zero_crossing_time, 0), xycoords='data',
                         xytext=(first_pos_zero_crossing_time, 0), textcoords='data',
                         arrowprops=dict(facecolor='black', linewidth=5, arrowstyle='<->'))
            mid_time = first_neg_zero_crossing_time + (first_pos_zero_crossing_time - first_neg_zero_crossing_time) / 2
            plt.annotate(str(duration_seconds) + ' s', xy=(mid_time, 0), xycoords='data', ha='center', va='bottom', size=18)

        plt.savefig(plot_path_filename)
        if jedi_config.verbose:
            jedi_config.logger.info("Summary plot saved to %s" % plot_path_filename)

    if not found_duration:
        duration_seconds = np.nan
        first_neg_zero_crossing_time = np.nan
        first_pos_zero_crossing_time = np.nan

    return duration_seconds, first_neg_zero_crossing_time, first_pos_zero_crossing_time

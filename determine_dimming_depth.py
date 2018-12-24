# Standard modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelmin

# Custom modules
import jedi_config

__author__ = 'James Paul Mason'
__contact__ = 'jmason86@gmail.com'


def determine_dimming_depth(light_curve_df,
                            earliest_allowed_time=None, latest_allowed_time=None, smooth_points=0,
                            plot_path_filename=None):
    """Find the depth of dimming in a light curve, if any.
    Assumes light curve is normalized such that pre-flare = 0%.

    Inputs:
        light_curve_df [pd DataFrame]:    A pandas DataFrame with a DatetimeIndex and a column for irradiance.

    Optional Inputs:
        earliest_allowed_time [metatime]: The function won't return a depth determined any earlier than this.
                                          Default is None, meaning the beginning of the light_curve_df.
        latest_allowed_time [metatime]:   The function won't return a depth determined any later than this.
                                          Default is None, meaning the end of the light_curve_df.
        smooth_points [integer]:          Used to apply a rolling mean with the number of points (indices) specified.
                                          Default is 0, meaning no smoothing will be performed.
        plot_path_filename [str]:         Set to a path and filename in order to save the summary plot to disk.
                                          Default is None, meaning the plot will not be saved to disk.

    Outputs:
        depth_first [float]:             The first detected depth of dimming in percent terms. np.nan if indeterminable.
        depth_first_time [pd.Timestamp]: The time of the first depth. np.nan if indeterminable.
        depth_max [float]:               The max detected depth of dimming in percent terms. np.nan if indeterminable.
        depth_max_time [pd.Timestamp]:   The time of the max depth. np.nan if indeterminable.

    Optional Outputs:
        None

    Example:
        depth_first, depth_first_time, depth_max, depth_max_time = determine_dimming_depth(light_curve_df)
    """
    if jedi_config.verbose:
        jedi_config.logger.info("Running on event with light curve start time of {0}.".format(light_curve_df.index[0]))

    # If no earliest_allowed_time set, then set it to beginning of light_curve_df
    if not earliest_allowed_time:
        earliest_allowed_time = light_curve_df.index[0]
        if jedi_config.verbose:
            jedi_config.logger.info("No earliest allowed time provided. Setting to beginning of light curve: {0}".format(earliest_allowed_time))

    # If no latest_allowed_time set, then set it to end of light_curve_df
    if not latest_allowed_time:
        latest_allowed_time = light_curve_df.index[-1]
        if jedi_config.verbose:
            jedi_config.logger.info("No latest allowed time provided. Setting to end of light curve: {0}".format(latest_allowed_time))

    # Truncate light curve to earliest-latest time allowed
    light_curve_df = light_curve_df[earliest_allowed_time:latest_allowed_time]

    # Optionally smooth the light curve with a rolling mean
    if smooth_points:
        light_curve_df['irradiance'] = light_curve_df.rolling(smooth_points, center=True).mean()
        if jedi_config.verbose:
            jedi_config.logger.info('Applied {0} point smooth.'.format(smooth_points))

    first_non_nan = light_curve_df['irradiance'].first_valid_index()
    nan_indices = np.isnan(light_curve_df['irradiance'])
    light_curve_df['irradiance'][nan_indices] = light_curve_df['irradiance'][first_non_nan]

    # Find the local minima
    minima_indices = argrelmin(light_curve_df['irradiance'].values)[0]
    minima_times = light_curve_df.index[minima_indices]
    minima_irradiances = light_curve_df['irradiance'].values[minima_indices]
    if jedi_config.verbose:
        if minima_indices.size > 0:
            jedi_config.logger.info('Found {0} local minima.'.format(minima_indices.size))
        else:
            jedi_config.logger.warning('No local minima found.')

    # Apply conditions to obtain first and max dimming depths
    less_than_zero_indices = minima_indices[minima_irradiances < 0]  # Assumes 0 is the baseline
    if less_than_zero_indices.size > 0:
        depth_first_index = less_than_zero_indices[0]
        depth_first_time = light_curve_df.index[depth_first_index]
        depth_first = np.abs(light_curve_df['irradiance'].values[depth_first_index])

        depth_max_index = np.argmax(np.abs(minima_irradiances))
        depth_max_time = light_curve_df.index[minima_indices[depth_max_index]]
        depth_max = max(np.abs(minima_irradiances))

        if jedi_config.verbose:
            jedi_config.logger.info('First depth determined to be {0:.2f} at {1}'.format(depth_first, depth_first_time))
            jedi_config.logger.info('Max depth determined to be {0:.2f} at {1}'.format(depth_max, depth_max_time))
    else:
        depth_first = np.nan
        depth_first_time = np.nan
        depth_max = np.nan
        depth_max_time = np.nan
        if jedi_config.verbose and minima_indices.size > 0:
            jedi_config.logger.warning('None of the minima are below 0.')

    # Produce a summary plot
    if plot_path_filename:
        plt.style.use('jpm-transparent-light')
        plt.close('all')
        from matplotlib import dates

        ax = light_curve_df['irradiance'].plot()
        plt.scatter(minima_times, minima_irradiances,
                    c='black', s=300, zorder=3, label='minima')
        if depth_first:
            plt.scatter(depth_first_time, -depth_first,
                        c='goldenrod', s=300, zorder=3, label='first depth')
            plt.scatter(depth_max_time, -depth_max,
                        c='tomato', s=300, zorder=3, label='max depth')
            plt.title('First Depth at {0} \n'
                      'Max Depth at {1}'.format(depth_first_time, depth_max_time))
        else:
            plt.title('No Depth Found')
        plt.axhline(linestyle='dashed', color='black')
        start_date = light_curve_df.index.values[0]
        start_date_string = pd.to_datetime(str(start_date))
        plt.xlabel(start_date_string.strftime('%Y-%m-%d %H:%M:%S'))
        plt.ylabel('Irradiance [%]')
        fmtr = dates.DateFormatter("%H:%M:%S")
        ax.xaxis.set_major_formatter(fmtr)
        ax.xaxis.set_major_locator(dates.HourLocator())
        ax.legend(loc='best')

        if depth_first:
            plt.annotate('', xy=(depth_first_time, -depth_first), xycoords='data',
                         xytext=(depth_first_time, 0), textcoords='data',
                         arrowprops=dict(facecolor='goldenrod', edgecolor='goldenrod', linewidth=2))
            mid_depth = -depth_first / 2.0
            plt.annotate('{0:.2f}'.format(depth_first), xy=(depth_first_time, mid_depth), xycoords='data',
                         ha='left', va='center', rotation=90, size=18, color='goldenrod')

            plt.annotate('', xy=(depth_max_time, -depth_max), xycoords='data',
                         xytext=(depth_max_time, 0), textcoords='data',
                         arrowprops=dict(facecolor='tomato', edgecolor='tomato', linewidth=2))
            mid_depth = -depth_max / 2.0
            plt.annotate('{0:.2f}'.format(depth_max), xy=(depth_max_time, mid_depth), xycoords='data',
                         ha='right', va='center', rotation=90, size=18, color='tomato')

        plt.savefig(plot_path_filename)
        if jedi_config.verbose:
            jedi_config.logger.info("Summary plot saved to %s" % plot_path_filename)

    return depth_first, depth_first_time, depth_max, depth_max_time

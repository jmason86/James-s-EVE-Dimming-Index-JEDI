# Standard modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom modules
import jedi_config
from jpm_number_printing import latex_float

__author__ = 'James Paul Mason'
__contact__ = 'jmason86@gmail.com'


def determine_dimming_slope(light_curve_df,
                            earliest_allowed_time=None, latest_allowed_time=None, smooth_points=0,
                            plot_path_filename=None):
    """Find the slope of dimming in a light curve, if any.

    Inputs:
        light_curve_df [pd DataFrame]:    A pandas DataFrame with a DatetimeIndex and a column for irradiance.

    Optional Inputs:
        earliest_allowed_time [metatime]: The function won't return a slope determined any earlier than this.
                                          It is recommended that this be the peak time of the flare.
                                          Default is None, meaning the beginning of the light_curve_df.
        latest_allowed_time [metatime]:   The function won't return a slope determined any later than this.
                                          It is recommended that this be the identified time of dimming depth.
                                          Default is None, meaning the end of the light_curve_df.
        smooth_points [integer]:          Used to apply a rolling mean with the number of points (indices) specified.
                                          Default is 0, meaning no smoothing will be performed.
        plot_path_filename [str]:         Set to a path and filename in order to save the summary plot to disk.
                                          Default is None, meaning the plot will not be saved to disk.
        verbose [bool]:                   Set to log the processing messages to disk and console. Default is False.

    Outputs:
        slope_min [float]: The minimum slope of dimming in percent/second terms.
        slope_max [float]: The maximum slope of dimming in percent/second terms.
        slope_mean [float]: The mean slope of dimming in percent/second terms.

    Optional Outputs:
        None

    Example:
        slope_min, slope_max, slope_mean = determine_dimming_slope(light_curve_df,
                                                                   plot_path_filename='./determine_dimming_slope_summary.png')
    """
    if jedi_config.verbose:
        jedi_config.logger.info("Running on event with light curve start time of {0}.".format(light_curve_df.index[0]))

    # If no earliest_allowed_time set, then set it to beginning of light_curve_df
    if not earliest_allowed_time:
        earliest_allowed_time = light_curve_df.index[0]
        jedi_config.logger.info("No earliest allowed time provided. Setting to beginning of light curve: {0}".format(earliest_allowed_time))

    # If no latest_allowed_time set, then set it to end of light_curve_df
    if not latest_allowed_time:
        latest_allowed_time = light_curve_df.index[-1]
        jedi_config.logger.info("No latest allowed time provided. Setting to end of light curve: {0}".format(latest_allowed_time))

    # Optionally smooth the light curve with a rolling mean
    if smooth_points:
        light_curve_df['irradiance'] = light_curve_df.rolling(smooth_points, center=True).mean()
        if jedi_config.verbose:
            jedi_config.logger.info('Applied {0} point smooth.'.format(smooth_points))

    first_non_nan = light_curve_df['irradiance'].first_valid_index()
    nan_indices = np.isnan(light_curve_df['irradiance'])
    light_curve_df['irradiance'][nan_indices] = light_curve_df['irradiance'][first_non_nan]

    # Find the max in the allowed window
    max_time = light_curve_df[earliest_allowed_time:latest_allowed_time]['irradiance'].idxmax()
    max_irradiance = light_curve_df['irradiance'].loc[max_time]
    if jedi_config.verbose:
        jedi_config.logger.info('Maximum in allowed window found with value of {0:.2f} at time {1}'.format(max_irradiance, max_time))

    # Compute the derivative in the time window of interest (inverting sign so that we describe "downward slope")
    derivative = -light_curve_df[max_time:latest_allowed_time]['irradiance'].diff() / light_curve_df[max_time:latest_allowed_time].index.to_series().diff().dt.total_seconds()
    if jedi_config.verbose:
        jedi_config.logger.info("Computed derivative of light curve within time window of interest.")

    # Get the min, max, and mean slope
    slope_min = derivative.min()
    slope_max = derivative.max()
    slope_mean = derivative.mean()
    slope_min_str = latex_float(slope_min)
    slope_max_str = latex_float(slope_max)
    slope_mean_str = latex_float(slope_mean)
    if jedi_config.verbose:
        jedi_config.logger.info("Computed min ({0}), max ({1}), and mean ({2}) %/s slope.".format(slope_min_str, slope_max_str, slope_mean_str))

    # Do a few sanity checks for the log
    if jedi_config.verbose:
        if slope_min < 0:
            jedi_config.logger.warning("Minimum slope of {0} is unexpectedly < 0.".format(slope_min))
        if slope_max < 0:
            jedi_config.logger.warning("Maximum slope of {0} is unexpectedly < 0.".format(slope_max))
        if slope_mean < 0:
            jedi_config.logger.warning("Mean slope of {0} is unexpectedly < 0.".format(slope_mean))

    # Produce a summary plot
    if plot_path_filename:
        plt.style.use('jpm-transparent-light')
        from matplotlib import dates

        p = plt.plot(light_curve_df['irradiance'])
        p = plt.plot(light_curve_df[max_time:latest_allowed_time]['irradiance'], label='slope region')
        ax = plt.gca()
        plt.axvline(x=earliest_allowed_time, linestyle='dashed', color='grey')
        plt.axvline(x=latest_allowed_time, linestyle='dashed', color='black')
        plt.axvline(x=max_time, linestyle='dashed', color='black')
        plt.title('Identified Slope')

        start_date = light_curve_df.index.values[0]
        start_date_string = pd.to_datetime(str(start_date))
        plt.xlabel(start_date_string.strftime('%Y-%m-%d %H:%M:%S'))
        fmtr = dates.DateFormatter("%H:%M:%S")
        ax.xaxis.set_major_formatter(fmtr)
        ax.xaxis.set_major_locator(dates.HourLocator())
        ax.xaxis.grid(b=True, which='minor')
        plt.ylabel('Irradiance [%]')

        inverse_str = '$^{-1}$'
        plt.annotate('slope_min={0} % sec{1}'.format(slope_min_str, inverse_str),
                     xy=(0.98, 0.12), xycoords='axes fraction', ha='right',
                     size=12, color=p[0].get_color())
        plt.annotate('slope_max={0} % sec{1}'.format(slope_max_str, inverse_str),
                     xy=(0.98, 0.08), xycoords='axes fraction', ha='right',
                     size=12, color=p[0].get_color())
        plt.annotate('slope_mean={0} % sec{1}'.format(slope_mean_str, inverse_str),
                     xy=(0.98, 0.04), xycoords='axes fraction', ha='right',
                     size=12, color=p[0].get_color())

        ax.legend(loc='best')

        plt.savefig(plot_path_filename)
        if jedi_config.verbose:
            jedi_config.logger.info("Summary plot saved to %s" % plot_path_filename)

    # Return the slopes
    return slope_min, slope_max, slope_mean

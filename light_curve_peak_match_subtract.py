# Standard modules
import os
import numpy as np
import matplotlib.pyplot as plt

# Custom modules
from jpm_logger import JpmLogger
import peakutils.peak
from closest import closest

__author__ = 'James Paul Mason'
__contact__ = 'jmason86@gmail.com'


def light_curve_peak_match_subtract(light_curve_to_subtract_from_df, light_curve_to_subtract_with_df, estimated_time_of_peak,
                                    max_seconds_shift=1800,
                                    plot_path_filename=None, verbose=False, logger=None):
    """Align the peak of a second light curve to the first, scale its magnitude to match, and subtract it off.

    Inputs:
        light_curve_to_subtract_from_df [pd DataFrame]: A pandas DataFrame with a DatetimeIndex and a column for irradiance.
        light_curve_to_subtract_with_df [pd DataFrame]: A pandas DataFrame with a DatetimeIndex and a column for irradiance.
        estimated_time_of_peak [metatime]: The estimated time that the peak should occur. This could come from, e.g., GOES/XRS.

    Optional Inputs:
        max_seconds_shift [int]:  The maximum allowed time shift in seconds to get the peaks to match.
        plot_path_filename [str]: Set to a path and filename in order to save the summary plot to disk.
                                  Default is None, meaning the plot will not be saved to disk.
        verbose [bool]:           Set to log the processing messages to disk and console. Default is False.
        logger [JpmLogger]:       A configured logger from jpm_logger.py. If set to None, will generate a
                                  new one. Default is None.

    Outputs:
        light_curve_corrected_df [pd DataFrame]: A pandas DataFrame with the same format as light_curve_to_subtract_from_df but
                                                 with the resultant peak match and subtraction performed. Returns np.nan if
                                                 the peaks couldn't be found.
        seconds_shift [float]:                   The number of seconds that light_curve_to_subtract_with_df was shifted to get
                                                 its peak to match light_curve_to_subtract_from_df. Returns np.nan if
                                                 the peaks couldn't be found.
        scale_factor [float]:                    The multiplicative factor applied to light_curve_to_subtract_with_df to get
                                                 its peak to match light_curve_to_subtract_from_df. Returns np.nan if
                                                 the peaks couldn't be found.

    Optional Outputs:
        None

    Example:
        light_curve_corrected_df, seconds_shift, scale_factor = light_curve_peak_match_subtract(light_curve_to_subtract_from_df,
                                                                                                light_curve_to_subtract_with_df,
                                                                                                estimated_time_of_peak,
                                                                                                plot_path_filename='./',
                                                                                                verbose=True)
    """

    # Prepare the logger for verbose
    if verbose:
        if not logger:
            logger = JpmLogger(filename='light_curve_peak_match_subtract_log', path='/Users/jmason86/Desktop/')
        logger.info("Running on event with light curve start time of {0}.".format(light_curve_to_subtract_from_df.index[0]))

    # Check that the two input light curves have the same length and return NaN if not
    # This is to handle the (numerous) cases where MEGS-B cadence is < MEGS-A and vice versa
    if len(light_curve_to_subtract_from_df) != len(light_curve_to_subtract_with_df):
        if verbose:
            logger.warning('Input light curves have different length, i.e. cadence. Must skip.')
        return np.nan, np.nan, np.nan

    # Drop NaNs since peakutils can't handle them
    light_curve_to_subtract_from_df = light_curve_to_subtract_from_df.dropna()
    light_curve_to_subtract_with_df = light_curve_to_subtract_with_df.dropna()

    # Detrend and find the peaks that are ≥ 95% of the max irradiance within
    if verbose:
        logger.info("Detrending light curves.")
    if (light_curve_to_subtract_from_df['irradiance'].values < 0).all():
        light_curve_to_subtract_from_df.iloc[0] = 1  # Else can crash peakutils.baseline
    base_from = peakutils.baseline(light_curve_to_subtract_from_df)
    detrend_from = light_curve_to_subtract_from_df - base_from
    indices_from = peakutils.indexes(detrend_from.values.squeeze(), thres=0.95)

    if (light_curve_to_subtract_with_df['irradiance'].values < 0).all():
        light_curve_to_subtract_with_df.iloc[0] = 1  # Else can crash peakutils.baseline
    base_with = peakutils.baseline(light_curve_to_subtract_with_df)
    detrend_with = light_curve_to_subtract_with_df - base_with
    indices_with = peakutils.indexes(detrend_with.values.squeeze(), thres=0.95)

    if len(indices_from) == 0:
        if verbose:
            logger.warning('Could not find peak in light curve to subtract from.')
        return np.nan, np.nan, np.nan
    if len(indices_with) == 0:
        if verbose:
            logger.warning('Could not find peak in light curve to subtract with.')
        return np.nan, np.nan, np.nan

    # Identify the peak closest to the input estimated peak time (e.g., from GOES/XRS)
    if verbose:
        logger.info("Identifying peaks closest to initial guess in light curves.")
    peak_index_from = indices_from[closest(light_curve_to_subtract_from_df.index[indices_from], estimated_time_of_peak)]
    peak_index_with = indices_with[closest(light_curve_to_subtract_with_df.index[indices_with], estimated_time_of_peak)]
    peak_time_from = light_curve_to_subtract_from_df.index[peak_index_from]
    index_shift = peak_index_from - peak_index_with

    # Compute how many seconds the time shift corresponds to
    seconds_shift = (light_curve_to_subtract_from_df.index[peak_index_from] -
                     light_curve_to_subtract_with_df.index[peak_index_with]).total_seconds()

    # Fail if seconds_shift > max_seconds_shift
    isTimeShiftValid = True
    if abs(seconds_shift) > max_seconds_shift:
        if verbose:
            logger.warning("Cannot do peak match. Time shift of {0} seconds is greater than max allowed shift of {1} seconds.".format(seconds_shift, max_seconds_shift))
        isTimeShiftValid = False

    # Shift the subtract_with light curve in time to align its peak to the subtract_from light curve
    if isTimeShiftValid:
        if verbose:
            logger.info("Shifting and scaling the light curve to subtract with.")
        shifted_with = light_curve_to_subtract_with_df.shift(index_shift)

        # Scale the subtract_with light curve peak irradiance to match the subtract_from light curve peak irradiance
        scale_factor = (detrend_from.values[peak_index_from] / shifted_with.values[peak_index_with + index_shift])[0]
        shifted_scaled_with = shifted_with * scale_factor
        light_curve_corrected_df = light_curve_to_subtract_from_df - shifted_scaled_with

        if verbose:
            if light_curve_corrected_df.isnull().values.sum() > 1:
                logger.warning("%s points were shifted to become NaN." % light_curve_corrected_df.isnull().values.sum())
            logger.info("Light curve peak matching and subtraction complete.")

    if plot_path_filename:
        from jpm_number_printing import latex_float
        seconds_shift_string = '+' if seconds_shift >= 0 else ''
        seconds_shift_string += str(int(seconds_shift))
        if isTimeShiftValid:
            scale_factor_string = latex_float(scale_factor)

        plt.style.use('jpm-transparent-light')
        from matplotlib import dates

        plt.clf()
        fig, ax = plt.subplots()
        plt.plot(light_curve_to_subtract_from_df.index.values, light_curve_to_subtract_from_df.values, c='limegreen')
        plt.tick_params(axis='x', which='minor', labelbottom='off')
        plt.xlabel(estimated_time_of_peak)
        plt.ylabel('Irradiance [%]')
        fmtr = dates.DateFormatter("%H:%M:%S")
        ax.xaxis.set_major_formatter(fmtr)
        ax.xaxis.set_major_locator(dates.HourLocator())

        if isTimeShiftValid:
            plt.title('I: $\\times$' + scale_factor_string + ', t: ' + seconds_shift_string + ' s', color='tomato')
            shifted_scaled_with.plot(c='tomato', label='subtract with', ax=ax)
            light_curve_corrected_df.plot(c='darkgrey', label='result', ax=ax)
        else:
            plt.title('t: ' + seconds_shift_string + ' s > max allowed {0} s'.format(max_seconds_shift), color='tomato')
            plt.plot(light_curve_to_subtract_with_df.index.values, light_curve_to_subtract_with_df.values, c='tomato')
        plt.scatter(light_curve_to_subtract_from_df.index[peak_index_from], light_curve_to_subtract_from_df.values[peak_index_from], c='black')

        if isTimeShiftValid:
            plt.scatter(shifted_scaled_with.index[peak_index_with + index_shift], shifted_scaled_with.values[peak_index_with + index_shift], c='black')
            ax.legend(['subtract from', 'subtract with', 'result'], loc='best')
        else:
            plt.scatter(light_curve_to_subtract_with_df.index[peak_index_with], light_curve_to_subtract_with_df.values[peak_index_with], c='black')
            ax.legend(['subtract from', 'subtract with'], loc='best')

        path = os.path.dirname(plot_path_filename)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(plot_path_filename)

        if verbose:
            logger.info("Summary plot saved to %s" % plot_path_filename)

    if isTimeShiftValid:
        return light_curve_corrected_df, seconds_shift, scale_factor
    else:
        return np.nan, seconds_shift, np.nan

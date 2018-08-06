# Standard modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import astropy.units as u

# Custom modules
from jpm_logger import JpmLogger
from jpm_number_printing import latex_float

# Configuration
import jedi_config

__author__ = 'James Paul Mason'
__contact__ = 'jmason86@gmail.com'


def determine_preflare_irradiance(light_curve_df, estimated_time_of_peak_start,
                                  max_median_diff_threshold=1.5, std_threshold=1.0,
                                  plot_path_filename=None, verbose=False, logger=None):
    """Determine pre-flare irradiance level in a solar light curve.
    Or, more generally, find the pre-peak level in a time series.

    Inputs:
        light_curve_df [pd DataFrame]:           A pandas DataFrame with a DatetimeIndex and a column for irradiance.
        estimated_time_of_peak_start [metatime]: The estimated time that the dramatic increase starts.
                                                 This could come from, e.g., GOES/XRS.

    Optional Inputs:
        max_median_diff_threshold [float]: The maximum allowed difference in medians between the 3 pre-flare windows
                                           in percent terms. This value gets multiplied by the mean of the stds from
                                           each sub-window and is then compared to the max_median_diff. The default is 1.5.
        std_threshold [float]:             The maximum allowed standard deviation in the pre-flare windows in percent
                                           terms. The default is 0.5.
        plot_path_filename [str]:          Set to a path and filename in order to save the summary plot to disk.
                                           Default is None, meaning the plot will not be saved to disk.
        verbose [bool]:                    Set to log the processing messages to disk and console. Default is False.
        logger [JpmLogger]:                A configured logger from jpm_logger.py. If set to None, will generate a
                                           new one. Default is None.

    Outputs:
        preflare_irradiance [float]: The identified pre-flare irradiance level in the same units as light_curve_df.irradiance.

    Optional Outputs:
        None

    Example:
        preflare_irradiance = determine_preflare_irradiance(light_curve_df, pd.Timestamp('2012-04-15 17:52:20.0'),
                                                            plot_path_filename='./bla.png',
                                                            verbose=True)
    """
    # Prepare the logger for verbose
    if verbose:
        if not logger:
            logger = JpmLogger(filename='determine_preflare_irradiance_log', path='/Users/jmason86/Desktop/')
        logger.info("Running on event with peak start time of {0}.".format(estimated_time_of_peak_start))

    # Verify that not all values are nan
    if light_curve_df.isna().all().all():
        if verbose:
            logger.warning("All irradiance values are NaN. Returning.")
        return np.nan

    # Convert irradiance to percent if not already present
    if 'irradiance_percent' not in light_curve_df.columns:
        median_irradiance = light_curve_df['irradiance'].median()
        light_curve_df['irradiance_percent'] = (light_curve_df['irradiance'].values - median_irradiance) / median_irradiance * 100.
        if verbose:
            logger.info("Converted irradiance to percent, baselining median in entire pre-flare window.")

    # Divide the pre-flare period into 3 equal-length windows
    windows = np.array_split(light_curve_df[:estimated_time_of_peak_start], 3)
    if verbose:
        logger.info("Divided pre-flare period into 3 equal-length windows.")

    # Compute median and σ in each window
    medians = [windowed_df['irradiance_percent'].median() for windowed_df in windows]
    medians_abs = [windowed_df['irradiance'].median() for windowed_df in windows]
    stds = np.array([windowed_df['irradiance_percent'].std() for windowed_df in windows])
    if verbose:
        logger.info("Computed medians and standard deviations in each window.")

    # Compute max difference between the medians
    max_median_diff = np.max(np.abs(np.append(np.diff(medians), medians[2] - medians[0])))

    # Compare medians and σs in each window to thresholds
    failed_median_threshold = False
    failed_std_threshold = False
    if np.all(np.isnan(stds)):
        if verbose:
            logger.warning('Cannot compute pre-flare irradiance. All standard deviations are nan.')
        failed_std_threshold = True
    else:
        if max_median_diff > max_median_diff_threshold * np.mean(stds):
            if verbose:
                logger.warning(
                    'Cannot compute pre-flare irradiance. Maximum difference in window medians ({0}) exceeded threshold ({1}).'.format(max_median_diff, max_median_diff_threshold * np.mean(stds)))
            failed_median_threshold = True
        if (stds < std_threshold).sum() < 2:
            if verbose:
                logger.warning('Cannot compute pre-flare irradiance. Standard deviation in more than 1 window is larger than threshold ({0}).'.format(std_threshold))
            failed_std_threshold = True

    # Compute pre-flare irradiance (mean of the medians in absolute units)
    if failed_median_threshold or failed_std_threshold:
        preflare_irradiance = np.nan
    else:
        preflare_irradiance = np.mean([windowed_df['irradiance'].median() for windowed_df in windows])
        if verbose:
            logger.info("Computed pre-flare irradiance: {0}".format(preflare_irradiance))

    # Produce summary plot
    if plot_path_filename:
        plt.style.use('jpm-transparent-light')
        from matplotlib import dates
        from matplotlib.patches import Rectangle

        light_curve_df = light_curve_df.drop('irradiance_percent', 1)
        ax = light_curve_df[:estimated_time_of_peak_start].plot(legend=False, c='grey')
        #  plt.plot(light_curve_df[:estimated_time_of_peak_start].irradiance, c='grey') #  using matplotlib instead of pandas
        #  ax = plt.gca()
        start_date = light_curve_df.index.values[0]
        start_date_string = pd.to_datetime(str(start_date))
        plt.title('Pre-flare Windows')
        plt.xlabel(start_date_string.strftime('%Y-%m-%d %H:%M:%S'))
        plt.ylabel('Irradiance [W m$^{-2}$]')
        fmtr = dates.DateFormatter("%H:%M:%S")
        ax.xaxis.set_major_formatter(fmtr)
        ax.xaxis.set_major_locator(dates.HourLocator())
        ax2 = ax.twinx()
        light_curve_df[:estimated_time_of_peak_start].plot(ax=ax2, legend=False, c='grey')
        #  ax2.plot(light_curve_df[:estimated_time_of_peak_start].irradiance, color='grey')
        vals = ax2.get_yticks()
        ax2.set_yticklabels(['{:3.2f}%'.format((x - median_irradiance) / median_irradiance * 100)
                             for x in vals])

        # First window
        start = dates.date2num(light_curve_df.index[0])
        end = dates.date2num(windows[0].index[-1])
        width = end - start
        rect = Rectangle((start, 0), width, 1, color='deepskyblue', alpha=0.2)
        ax.add_patch(rect)
        plt.plot([windows[0].index[0], windows[0].index[-1]], [medians_abs[0], medians_abs[0]],
                 linestyle='dashed', c='dimgrey')
        ax.text(start + width / 2.0, np.min(light_curve_df[:estimated_time_of_peak_start].irradiance),
                'median$_1$ = ' + latex_float(medians[0]) + '% \n' +
                '$\sigma_1$ = ' + latex_float(stds[0]) + '%',
                fontsize=11, ha='center', va='bottom')

        # Second window
        start = dates.date2num(windows[1].index[0])
        end = dates.date2num(windows[1].index[-1])
        width = end - start
        rect = Rectangle((start, 0), width, 1, color='slateblue', alpha=0.2)
        ax.add_patch(rect)
        plt.plot([windows[1].index[0], windows[1].index[-1]], [medians_abs[1], medians_abs[1]],
                 linestyle='dashed', c='dimgrey')
        ax.text(start + width / 2.0, np.min(light_curve_df[:estimated_time_of_peak_start].irradiance),
                'median$_2$ = ' + latex_float(medians[1]) + '% \n' +
                '$\sigma_2$ = ' + latex_float(stds[1]) + '%',
                fontsize=11, ha='center', va='bottom')

        if not np.isnan(preflare_irradiance):
            ax.axes.axhline(y=preflare_irradiance, linewidth=2, color='tomato', linestyle='dashed')
            ax.text(start + width / 2.0, np.max(light_curve_df[:estimated_time_of_peak_start].irradiance),
                    'pre-flare I = ' + latex_float(preflare_irradiance) + ' W m$^{-2}$',
                    fontsize=11, ha='center', va='top', color='tomato')
        else:
            ax.text(start + width / 2.0, np.max(light_curve_df[:estimated_time_of_peak_start].irradiance),
                    'pre-flare I = N/A \n' +
                    'median condition ok: ' + str(not failed_median_threshold) + '\n' +
                    '$\sigma$ condition ok: ' + str(not failed_std_threshold),
                    fontsize=11, ha='center', va='top', color='tomato')

        # Third window
        start = dates.date2num(windows[2].index[0])
        end = dates.date2num(windows[2].index[-1])
        width = end - start
        rect = Rectangle((start, 0), width, 1, color='violet', alpha=0.2)
        ax.add_patch(rect)
        plt.plot([windows[2].index[0], windows[2].index[-1]], [medians_abs[2], medians_abs[2]],
                 linestyle='dashed', c='dimgrey')
        ax.text(start + width / 2.0, np.min(light_curve_df[:estimated_time_of_peak_start].irradiance),
                'median$_3$ = ' + latex_float(medians[2]) + '% \n' +
                '$\sigma_3$ = ' + latex_float(stds[2]) + '%',
                fontsize=11, ha='center', va='bottom')
        ax.text(end, np.max(light_curve_df[:estimated_time_of_peak_start].irradiance),
                'median diff = ' + latex_float(max_median_diff) + '% \n' +
                r'thresh $\times \mu_{\sigma n}$ = ' + latex_float(max_median_diff_threshold * np.mean(stds)) + '%',
                fontsize=11, ha='right', va='top')

        # Increase border so y-axes don't get cut off in savefig, even though they don't in plt.show()
        plt.gcf().subplots_adjust(left=0.22)

        plt.savefig(plot_path_filename)
        if verbose:
            logger.info("Summary plot for event with start time {0} saved to {1}".format(estimated_time_of_peak_start, plot_path_filename))

    return preflare_irradiance


def get_preflare_irradiance_all_emission_lines(flare_index,
                             verbose=False, logger=None):
    """Loop through all (39) of the EVE extracted emission lines and get the pre-flare irradiance for each

        Inputs:
            flare_index [int]: The identifier for which event in JEDI to process

        Optional Inputs:
            verbose [bool]:     Set to log the processing messages to disk and console. Default is False.
            logger [JpmLogger]: A configured logger from jpm_logger.py. If set to None, will generate a new one.
                                Default is None.

        Outputs:
            preflare_irradiance [float]: The identified pre-flare irradiance level in the same units as light_curve_df.irradiance.
            preflare_window_start [str]: The time that the pre-flare irradiance calculation starts.
            preflare_window_end [str]:   The time that the pre-flare irradiance calculation ends.

        Optional Outputs:
            None

        Example:
            preflare_irradiance, preflare_window_start, preflare_window_end = get_preflare_irradiance_all_emission_lines(flare_index,
                                                                                                       verbose=verbose, logger=logger)
    """
    # Prepare the logger for verbose
    if verbose:
        if not logger:
            logger = JpmLogger(filename='determine_preflare_irradiance_log', path='/Users/jmason86/Desktop/')
        logger.info("Running on event {0}.".format(flare_index))

    # Clip EVE data from threshold_time_prior_flare_minutes prior to flare up to peak flare time
    preflare_window_start = (jedi_config.goes_flare_events['peak_time'][flare_index] - (jedi_config.threshold_time_prior_flare_minutes * u.minute)).iso
    preflare_window_end = (jedi_config.goes_flare_events['peak_time'][flare_index]).iso
    eve_lines_preflare_time = jedi_config.eve_lines[preflare_window_start:preflare_window_end]

    # Loop through the emission lines and get pre-flare irradiance for each
    preflare_irradiance = []
    for column in eve_lines_preflare_time:
        eve_line_preflare_time = pd.DataFrame(eve_lines_preflare_time[column])
        eve_line_preflare_time.columns = ['irradiance']

        preflare_temp = determine_preflare_irradiance(eve_line_preflare_time,
                                                      pd.Timestamp(jedi_config.goes_flare_events['start_time'][flare_index].iso),
                                                      plot_path_filename=os.path.join(jedi_config.output_path, 'Preflare_Determination', 'Event_%d_%s.png' % (flare_index, column)),
                                                      verbose=jedi_config.verbose,
                                                      logger=jedi_config.logger)

        preflare_irradiance.append(preflare_temp)

    return preflare_irradiance, preflare_window_start, preflare_window_end


def multiprocess_preflare_irradiance(preflare_indices,
                                     nworkers=2, verbose=False, logger=None):
    """Multi-threaded processing of pre-flare irradiance across time-independent flares

        Inputs:
            preflare_indices [np int array]: The subset of flare_indices that correspond to time-independent flares.

        Optional Inputs:
            nworkers [int]:     The number of parallel threads to use. Default is 2.
            verbose [bool]:     Set to log the processing messages to disk and console. Default is False.
            logger [JpmLogger]: A configured logger from jpm_logger.py. If set to None, will generate a new one.
                                Default is None.

        Outputs:
            preflare_irradiance [float]: The identified pre-flare irradiance level in the same units as light_curve_df.irradiance.
            preflare_window_start [str]: The time that the pre-flare irradiance calculation starts.
            preflare_window_end [str]:   The time that the pre-flare irradiance calculation ends.

        Optional Outputs:
            None

        Example:
            preflare_irradiance, preflare_window_start, preflare_window_end = multiprocess_preflare_irradiance(preflare_indices, 4)
    """
    # Prepare the logger for verbose
    if verbose:
        if not logger:
            logger = JpmLogger(filename='determine_preflare_irradiance_log', path='/Users/jmason86/Desktop/')
        logger.info("Running on {0} events with {1} threads.".format(len(preflare_indices), nworkers))

    if nworkers == 1:
        preflare_irradiances, preflare_windows_start, preflare_windows_end = zip(*map(get_preflare_irradiance_all_emission_lines, preflare_indices))
        logger.info('Preparing export of dataframe')
    else:
        pool = mp.Pool(processes=nworkers)
        preflare_irradiances, preflare_windows_start, preflare_windows_end = zip(*pool.map(get_preflare_irradiance_all_emission_lines, preflare_indices))
        pool.close()
        logger.info('Pool closed. Preparing export of dataframe')

    preflare_irradiances = np.array(preflare_irradiances)
    preflare_windows_start = preflare_windows_start  # TODO: Should this be np.array()?
    preflare_windows_end = preflare_windows_end  # TODO: Should this be np.array()?

    return preflare_irradiances, preflare_windows_start, preflare_windows_end

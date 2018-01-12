# Standard modules
import numpy as np
from sunpy.instr.goes import get_goes_event_list
from sunpy.time import TimeRange

# Custom modules
from jpm_time_conversions import metatimes_to_human
from jpm_logger import JpmLogger

__author__ = 'James Paul Mason'
__contact__ = 'jmason86@gmail.com'


def get_goes_flare_events(start_time, end_time, minimum_flare_size='C1',
                          verbose=False):
    """Get a list of flare events from NOAA's GOES/XRS. Just a wrapper around sunpy.instr.goes get_goes_event_list.

    Inputs:
        start_time [metatime or string]: The beginning of the time window of interest. See jpm_time_conversions.py
                                         (https://github.com/jmason86/python_convenience_functions/blob/master/jpm_time_conversions.py)
                                         for allowed metatime formats if not using an iso or human like time string.
        end_time [metatime]:             Same as start time but for the end of the time window.

    Optional Inputs:
        minimum_flare_size [string]: The minimum flare size to search for. Default is 'C1'.
        verbose [bool]:              Set to log the processing messages to disk and console. Default is False.

    Outputs:
        goes_events [list]: The list of GOES flare events corresponding to the input search criteria.

    Optional Outputs:
        None

    Example:
        goes_events = get_goes_flare_events(pd.Timestamp('2010-05-01 00:00:00'),
                                            pd.Timestamp('2018-01-12 00:00:00'),
                                            verbose=True)
    """

    # Prepare the logger for verbose
    if verbose:
        # TODO: Update the path
        logger = JpmLogger(filename='get_goes_flare_events_log', path='/Users/jmason86/Desktop/')
        logger.info("Getting > {0} flares from {1} to {2}.".format(minimum_flare_size, start_time, end_time))

    if not isinstance(start_time, str):
        start_time = metatimes_to_human(np.array([start_time]))[0]
    if not isinstance(end_time, str):
        end_time = metatimes_to_human(np.array([end_time]))[0]
    time_range = TimeRange(start_time, end_time)
    goes_events = get_goes_event_list(time_range, goes_class_filter='c1')

    if verbose:
        logger.info("Found {0} events.".format(len(goes_events)))

    # Return the slopes
    return goes_events

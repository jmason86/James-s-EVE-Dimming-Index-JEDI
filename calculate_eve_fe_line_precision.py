# Standard modules
import numpy as np
import pandas as pd
from getpass import getuser

# Custom modules
from closest import closest
from jpm_time_conversions import sod_to_hhmmss

__author__ = 'James Paul Mason'
__contact__ = 'jmason86@gmail.com'


def calculate_eve_fe_line_precision(number_of_samples_to_average=6,
                                    save_path='/Users/' + getuser() + '/Dropbox/Research/Postdoc_NASA/Analysis/Coronal Dimming Analysis/EVE Precision/',
                                    reload_eve_data=False,
                                    verbose=False):
    """Compute precisions for important lines in SDO/EVE.

    Inputs:
        None

    Optional Inputs:
        number_of_samples_to_average [float]: The number of 10 second integrations to average. Default is 6 (60 seconds).
        save_path [str]:                      The path to save (or load) the EVE data to (from) disk.
                                              Default is '/Users/' + getuser() + '/Dropbox/Research/Postdoc_NASA/Analysis/Coronal Dimming Analysis/EVE Precision/'
        reload_eve_data [bool]:               Set this to force the code to grab data from the EVE_DATA (environment variable) server.
                                              The data will be saved to disk in save_path. Default is False.
        verbose [bool]:                       Set to print out the precisions. Default is False.

    Outputs:
        precisions [pd.Series]: A pandas series where the index is the wavelength [Å] of the emission line and the data are the precisions.

    Optional Outputs:
        None

    Example:
        precisions = calculate_eve_fe_line_precisions(verbose=True)
    """

    # Get data for a quiet period - beginning of the below day is very quiet in 171 Å
    if reload_eve_data:
        if verbose:
            print('Fetching remote EVE data')
        # TODO: Implement this functionality
    else:
        from scipy.io.idl import readsav
        eveLines = readsav(save_path + 'EVE Line Data.sav')

    end_index = closest(eveLines['sod'], 3600)
    timestamp_iso = '2013-01-28 ' + sod_to_hhmmss(eveLines['sod'])[:end_index]
    eve_lines = eveLines['evelines'][:end_index, :]

    # Format data into pandas DataFrame
    selected_lines_dictionary = {'94': pd.Series(eve_lines[:, 0], index=timestamp_iso),
                                 '132': pd.Series(eve_lines[:, 2], index=timestamp_iso),
                                 '171': pd.Series(eve_lines[:, 3], index=timestamp_iso),
                                 '177': pd.Series(eve_lines[:, 4], index=timestamp_iso),
                                 '180': pd.Series(eve_lines[:, 5], index=timestamp_iso),
                                 '195': pd.Series(eve_lines[:, 6], index=timestamp_iso),
                                 '202': pd.Series(eve_lines[:, 7], index=timestamp_iso),
                                 '211': pd.Series(eve_lines[:, 8], index=timestamp_iso),
                                 '284': pd.Series(eve_lines[:, 10], index=timestamp_iso),
                                 '335': pd.Series(eve_lines[:, 12], index=timestamp_iso)}
    selected_lines = pd.DataFrame(selected_lines_dictionary)
    selected_lines.index.name = 'Timestamp'

    # Compute normalized precision time series
    group_to_average = selected_lines.groupby(np.arange(len(selected_lines)) // number_of_samples_to_average)
    precision_time_series = group_to_average.std() / (group_to_average.mean() * np.sqrt(number_of_samples_to_average))

    # Take average of normalized precision time series over the hour long period
    precision = precision_time_series.mean()

    if verbose:
        print(precision)

    return precision

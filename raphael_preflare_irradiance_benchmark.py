from importlib import reload
import os
import numpy as np
from astropy.time import Time
import time
import eve_jedi

import flare_settings


def parallel_process(ncores):

    time_start = time.time()

    if ncores == 1:
        preflare_irradiance, preflare_window_start, preflare_window_end = zip(*list(map(eve_jedi.loop_preflare_irradiance, preflare_indices)))
    else:
        preflare_irradiance, preflare_window_start, preflare_window_end = eve_jedi.multiprocess_preflare_irradiance(preflare_indices, ncores)

    elapsed_time = time.time() - time_start
    print('%d core(s): elapsed time = %d s' % (ncores, elapsed_time))
    return preflare_irradiance, preflare_window_start, preflare_window_end, elapsed_time


# Main

flare_settings.eve_data_path = '/Users/rattie/Data/James/eve_lines_2010121-2014146 MEGS-A Mission Bare Bones.sav'
flare_settings.goes_data_path = '/Users/rattie/Data/James/GoesEventsC1MinMegsAEra.sav'
flare_settings.output_path = '/Users/rattie/Data/James/output/'
flare_settings.csv_filename = flare_settings.output_path + 'jedi_{0}.csv'.format(Time.now().iso)
flare_settings.verbose = False

# Instantiante the flare settings with default values. Eve and Goes data are loaded in that object.
# flare_settings = eve_jedi.FlareSettings(eve_data_path, goes_data_path, output_path, verbose=verbose)
flare_settings.init()

# precalculate the series of minutes_since_last_flare which otherwise is calculated at each loop index
peak_time = flare_settings.goes_flare_events['peak_time']
all_minutes_since_last_flare = (peak_time[1:] - peak_time[0:-1]).sec / 60.0
tmask = all_minutes_since_last_flare > flare_settings.threshold_time_prior_flare_minutes
preflare_idx = np.where(tmask)[0]


preflare_indices = preflare_idx[0:8]

# Single cpu
r1 = parallel_process(1)
r2 = parallel_process(2)
r4 = parallel_process(4)
r6 = parallel_process(6)

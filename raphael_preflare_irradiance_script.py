from importlib import reload
import os
import numpy as np
import pandas as pd
from astropy.time import Time
import time
import jedi_config
import eve_jedi
from collections import OrderedDict


def prep_preflare_irradiance():
    # precalculate the series of minutes_since_last_flare which otherwise is calculated at each loop index
    peak_time = jedi_config.goes_flare_events['peak_time']
    all_minutes_since_last_flare = (peak_time[1:] - peak_time[0:-1]).sec / 60.0
    tmask = all_minutes_since_last_flare > jedi_config.threshold_time_prior_flare_minutes
    preflare_idx = np.where(tmask)[0] + 1  # Add 1 to map back to actual event index and not to the differentiated vector

    return preflare_idx


jedi_config.eve_data_path = '/Users/rattie/Data/James/eve_lines_2010121-2014146 MEGS-A Mission Bare Bones.sav'
jedi_config.goes_data_path = '/Users/rattie/Data/James/GoesEventsC1MinMegsAEra.sav'
jedi_config.output_path = '/Users/rattie/Data/James/output/'
jedi_config.csv_filename = jedi_config.output_path + 'jedi_{0}.csv'.format(Time.now().iso)
jedi_config.verbose = False
jedi_config.logger_filename = 'determine_preflare_irradiance_log'
# Nb of workers used in processing parallelized over the pre-flare indices.
nworkers = 4
# Load eve and goes data using above settings
jedi_config.init()
# Total number of events
nevents = jedi_config.goes_flare_events['peak_time'].size
# precalculate the series of minutes_since_last_flare which otherwise is calculated at each loop index
preflare_indices = prep_preflare_irradiance()
#preflare_indices = np.sort([4188, 4192, 4200, 4201, 4202, 4203, 4204, 4209, 4210, 4212, 5224, 5227, 4179, 5238, 5240, 4180, 5242, 5245, 5246, 4182, 5247, 4186, 5250, 5259])

# flare_index = 3032
# preflare_irradiance, preflare_window_start, preflare_window_end = eve_jedi.loop_preflare_irradiance(flare_index)

tstart = time.time()
preflare_irradiance, preflare_window_start, preflare_window_end = eve_jedi.multiprocess_preflare_irradiance(preflare_indices, nworkers)
telapsed = time.time() - tstart

print('Exporting dataframe')
# Build a dataframe for export
wavelengths = jedi_config.eve_lines.columns
wavelengths_str = jedi_config.eve_lines.columns.tolist()

preflare_window_df = pd.DataFrame(OrderedDict([('window start', preflare_window_start), ('window end', preflare_window_end)]))
preflare_irradiance_df = pd.DataFrame(preflare_irradiance, columns=wavelengths)

preflare_df = preflare_window_df.join(preflare_irradiance_df)
preflare_df.to_hdf(jedi_config.preflare_hdf_filename, 'preflare_df')

print('Preflare irradiance processing time: %sd' % telapsed)

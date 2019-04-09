import jedi_config
import os
import numpy as np
import pandas as pd

jedi_config.init()  # Configures and loads everything - takes about 60 seconds


def test_logger():
    try:
        jedi_config.logger.info('Executing tests only.')
    except:
        return False


def test_folders_exist():
    assert os.path.exists(jedi_config.output_path)
    assert os.path.exists(jedi_config.output_path + 'Processed Pre-Parameterization Data')
    assert os.path.exists(jedi_config.output_path + 'Processed Lines Data')
    assert os.path.exists(jedi_config.output_path + 'Peak Subtractions')
    assert os.path.exists(jedi_config.output_path + 'Fitting')
    assert os.path.exists(jedi_config.output_path + 'Depth')
    assert os.path.exists(jedi_config.output_path + 'Slope')
    assert os.path.exists(jedi_config.output_path + 'Duration')
    assert os.path.exists(jedi_config.output_path + 'Summary Plots')


def test_global_filenames_defined():
    assert jedi_config.jedi_hdf_filename is not None  # Can't test precise name since it includes a time-of-creation timestamp
    assert jedi_config.preflare_csv_filename == os.path.join(jedi_config.output_path, 'Preflare Determination/Preflare Irradiances.csv')


def test_load_eve_data():
    assert isinstance(jedi_config.eve_lines, pd.DataFrame)
    assert len(jedi_config.eve_lines.columns) == 39  # The number of SDO/EVE extracted emission lines
    assert len(jedi_config.eve_lines.index) > 1
    assert isinstance(jedi_config.eve_lines.index, pd.DatetimeIndex)


def test_load_goes_flare_event_data():
    import scipy.io.idl as idl
    assert isinstance(jedi_config.goes_flare_events, idl.AttrDict)
    assert goes_flare_event_dict_contains_expected_keys()
    assert goes_flare_event_dict_values_are_expected_type()


def goes_flare_event_dict_contains_expected_keys():
    keys = list(jedi_config.goes_flare_events.keys())
    assert 'class' in keys
    assert 'event_peak_time_human' in keys
    assert 'event_start_time_human' in keys
    assert 'peak_time' in keys
    assert 'start_time' in keys
    return True


def goes_flare_event_dict_values_are_expected_type():
    assert isinstance(jedi_config.goes_flare_events['class'][0], str)
    assert isinstance(jedi_config.goes_flare_events['event_peak_time_human'][0], str)
    assert isinstance(jedi_config.goes_flare_events['event_start_time_human'][0], str)
    from astropy.time.core import Time
    assert isinstance(jedi_config.goes_flare_events['peak_time'], Time)
    assert isinstance(jedi_config.goes_flare_events['start_time'], Time)
    return True


def test_all_minutes_since_last_flare_array():
    assert isinstance(jedi_config.all_minutes_since_last_flare, np.ndarray)
    assert len(jedi_config.all_minutes_since_last_flare) > 1


def test_preflare_indices_array():
    assert isinstance(jedi_config.preflare_indices, np.ndarray)
    assert len(jedi_config.preflare_indices) > 1


def test_jedi_row():
    jedi_row = jedi_config.init_jedi_row()
    jedi_config.write_new_jedi_file_to_disk(jedi_row)

    assert isinstance(jedi_row, pd.DataFrame)
    assert len(jedi_row) == 1
    assert os.path.isfile(jedi_config.jedi_hdf_filename)
    ion_names_for_jedi_row_created()


def ion_names_for_jedi_row_created():
    assert isinstance(jedi_config.ion_tuples, list)
    assert len(jedi_config.ion_tuples) == 1482  # The number of SDO/EVE extracted emission lines permutated

    assert isinstance(jedi_config.ion_permutations, pd.Index)
    assert len(jedi_config.ion_permutations) == 1482  # The number of SDO/EVE extracted emission lines permutated

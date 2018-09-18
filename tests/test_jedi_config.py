import jedi_config
import os

# Initialize the config
print('Initializing the jedi_config. This will take about a minute.')
jedi_config.init()  # Configures and loads everything
print('jedi_config initialization complete.')


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
    assert jedi_config.jedi_csv_filename is not None  # Can't test precise name since it includes a time-of-creation timestamp
    assert jedi_config.preflare_csv_filename == os.path.join(jedi_config.output_path, 'Preflare Determination/Preflare Irradiances.csv')


def test_load_eve_data():
    import pandas as pd
    assert isinstance(jedi_config.eve_lines, pd.DataFrame)
    assert len(jedi_config.eve_lines.columns) == 39  # The number of extracted emission lines
    assert len(jedi_config.eve_lines.index) > 1  # More than 1 time in the series
    assert isinstance(jedi_config.eve_lines.index, pd.DatetimeIndex)

def test_load_goes_flare_event_data():
    import scipy.io.idl as idl
    assert isinstance(jedi_config.goes_flare_events, idl.AttrDict)
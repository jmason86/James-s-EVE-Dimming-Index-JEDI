import os
import re
import pandas as pd
from astropy.time import Time
from tqdm import tqdm

JEDI_DIR = os.path.expanduser('~/Dropbox/Research/Postdoc_NASA/Analysis/Coronal Dimming Analysis/JEDI Catalog/')
JEDI_V2_FILENAME = 'jedi_v2.h5'
JEDI_V2_1_FILENAME = 'jedi_v2.1.0.h5'

def load_jedi_catalog(version_filename=JEDI_V2_FILENAME):
    """Load the specified JEDI catalog HDF5 file."""
    path = os.path.join(JEDI_DIR, version_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"JEDI catalog file not found: {path}")
    return pd.read_hdf(path)

def get_event_number_from_filename(filename):
    """
    Extract the event number from a filename like:
    'jedi_2025-08-21 13/50/42.319 Event 5044.h5'
    Returns the event number as an int, or None if not found.
    """
    match = re.search(r'Event (\d+)\.h5$', filename)
    if match:
        return int(match.group(1))
    return None

def find_single_event_files(directory):
    """
    Find all single event files in the directory.
    Returns a dict mapping event number to full file path.
    """
    event_files = {}
    all_files = os.listdir(directory)
    for fname in tqdm(all_files, desc="Scanning directory for event files"):
        if fname.startswith('jedi_') and 'Event' in fname and fname.endswith('.h5'):
            event_num = get_event_number_from_filename(fname)
            if event_num is not None:
                event_files[event_num] = os.path.join(directory, fname)
    return event_files

def load_single_event_row(filepath):
    """
    Load the single event row from the given HDF5 file.
    Assumes the key is 'jedi' or 'jedi_row'.
    """
    try:
        # Try both possible keys
        try:
            df = pd.read_hdf(filepath, key='jedi') 
        except (KeyError, ValueError):
            df = pd.read_hdf(filepath, key='jedi_row')
        # If it's a DataFrame with one row, return that row as a DataFrame
        if isinstance(df, pd.DataFrame):
            if len(df) == 1:
                return df.iloc[[0]]
            else:
                # If more than one row, try to find the correct one
                return df.iloc[[0]]
        else:
            # If it's a Series, convert to DataFrame
            return pd.DataFrame([df])
    except Exception as e:
        print(f"Failed to load event from {filepath}: {e}")
        return None

def replace_events_in_catalog(jedi_df, event_files):
    """
    For each event in event_files, replace the corresponding row in jedi_df.
    Returns the updated DataFrame.
    """
    # Map event numbers to their index in the DataFrame
    if 'Event #' not in jedi_df.columns:
        raise ValueError("Expected 'Event #' column in JEDI catalog DataFrame.")
    event_to_index = {int(ev): idx for idx, ev in enumerate(jedi_df['Event #'])}
    
    # Collect all new rows first
    new_rows_data = []
    valid_indices = []
    replaced = 0
    
    print("Loading all event files...")
    for event_num, filepath in tqdm(event_files.items(), desc="Loading event files", total=len(event_files)):
        if event_num in event_to_index:
            new_row = load_single_event_row(filepath)
            if new_row is not None:
                # Ensure columns match
                new_row = new_row[jedi_df.columns.intersection(new_row.columns)]
                new_rows_data.append(new_row.iloc[0])
                valid_indices.append(event_to_index[event_num])
                replaced += 1
        else:
            print(f"Event {event_num} not found in catalog, skipping.")
    
    if not new_rows_data:
        print("No valid event files found to replace.")
        return jedi_df
    
    print(f"Applying {replaced} replacements in bulk...")
    
    # Create a DataFrame from all new rows
    new_df = pd.DataFrame(new_rows_data)
    
    # Suppress FutureWarnings for dtype compatibility
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        # Handle datetime columns explicitly
        for col in new_df.columns:
            if col in jedi_df.columns and pd.api.types.is_datetime64_any_dtype(jedi_df[col]):
                try:
                    # Convert the entire column to datetime
                    new_df[col] = pd.to_datetime(new_df[col])
                except (ValueError, TypeError):
                    # If conversion fails, keep original values
                    pass
        
        # Bulk replace all rows at once
        jedi_df.loc[valid_indices, new_df.columns] = new_df.values
    
    print(f"Replaced {replaced} events in catalog.")
    return jedi_df

def save_jedi_catalog(jedi_df, filename=JEDI_V2_1_FILENAME):
    """Save the updated JEDI catalog to a new HDF5 file."""
    path = os.path.join(JEDI_DIR, filename)
    jedi_df.to_hdf(path, key='jedi', mode='w')
    print(f"Saved updated catalog to {path}")

def main():
    jedi_df = load_jedi_catalog()
    event_files = find_single_event_files(JEDI_DIR)
    if not event_files:
        print("No single event files found.")
        return
    jedi_df_updated = replace_events_in_catalog(jedi_df, event_files)
    save_jedi_catalog(jedi_df_updated)

if __name__ == '__main__':
    main()

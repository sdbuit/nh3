# dataset.py
import os
import re
import math
import logging
from typing import Tuple, Dict, List, Optional

import polars as pl
from tqdm.contrib.concurrent import thread_map

from config import config, Coordinate

logger = logging.getLogger(__name__)


class DatasetLoader:
    def __init__(self, base_path: str):
        self.base_path = base_path
    
    def load_all(self) -> List[Tuple[pl.DataFrame, Dict]]:
        return load_all_datasets(self.base_path)
    
    def load_all_datasets_parallel(self) -> List[Tuple[pl.DataFrame, Dict]]:
        return load_all_datasets(self.base_path)

def update_dataset(vehicle_df: pl.DataFrame, updated_coords: List[Coordinate],
        alt_col: str) -> pl.DataFrame:
    altitudes = [coord.altitude for coord in updated_coords]
    
    return vehicle_df.with_columns([pl.Series(name=alt_col, values=altitudes)])


# TODO FIX CATEMP11_$00(∞C),CATEMP21_$00(∞C)
#   - 2016\Chevrolet\Colorado\03_ECM.csv: column with name 'obd_cat_temp' has 
#     more than one occurrence.

# TODO vehicle_v2\2019\Toyota\Tacoma\03_ECM.csv
#   - column with name 'obd_cat_temp' has more than one occurrence

def process_nox_cols(df: pl.DataFrame, shift_count: int = 1) -> pl.DataFrame:
    """
    Process NOx data columns:  
        - Assumes apply_aliases() applied to the header col names.
        - Applies hard-coded NH3 calibration to new alias nh3_ppm.    
    
    TODO CHECK CALIB PROCEDURE
        - VERIFY calib data
        - Test Model
        
    TODO REFACTOR
        - Implement function dedicated for duplicate variables.  They are
          in the the OBD and NOx node sensor data sections.
        - Thus causing a processing failure on a few datasets.
        - Calibration procedures should be isolated/separated for better
          modularity.
    """
    if 'nox_node_01' in df.columns and 'nox_node_02' in df.columns:
        df = df.with_columns([
            pl.col('nox_node_01').shift(shift_count).alias('nox_node_01_shifted')
        ])
        diff_expr = pl.col('nox_node_02') - pl.col('nox_node_01_shifted')
        df = df.with_columns([
            pl.when(diff_expr < 0)
              .then(0.0)
              .otherwise(diff_expr)
              .alias('nox_ppm')
        ])
        
        def calibrate_nh3(x: float) -> float:
            if x is None or math.isnan(x):
                return None
            return 1.0553 * x + 0.3223
        
        df = df.with_columns([
            pl.col('nox_node_01_shifted')
              .map(calibrate_nh3)
              .alias('nh3_ppm')
        ])
        logger.info('nox_node_01 and nox_node_02')
    
    elif 'nox_node_01' in df.columns:
        logger.warning('Only one NOX column found. nox_node_02 is missing; setting nox_ppm to null.')
        df = df.with_columns([pl.lit(None).alias('nox_ppm')])
    
    else:
        logger.warning('No NOX column found (expected nox_node_01).')
    
    return df

def detect_header_row(file_path: str, max_skip: int = 100) -> Optional[int]:
    recognized_aliases = {alias.lower().strip() for aliases in config.COLUMN_ALIASES.values() for alias in aliases}
    candidate = None
    for skip in range(max_skip + 1):
        try:
            df_sample = pl.read_csv(file_path, has_header=False, n_rows=1, skip_rows=skip)
            row = df_sample.row(0)
            header = [str(x).strip() for x in row]
            if sum(1 for col in header if col) < 4:
                continue
            score = sum(1 for col in header if col.lower() in recognized_aliases)
            if score >= 4:
                candidate = skip
                break
        except Exception:
            continue
    if candidate is None:
        print(f'[Warning] No valid header found in {file_path} (checked {max_skip} lines).')
        return 0
    
    return candidate

def apply_aliases(df: pl.DataFrame) -> pl.DataFrame:
    new_names = []
    for col in df.columns:
        normalized = col.lower().strip()
        renamed = None
        for canonical, aliases in config.COLUMN_ALIASES.items():
            for alias in aliases:
                if normalized == alias.lower().strip():
                    renamed = canonical
                    break
            if renamed:
                break
        new_names.append(renamed if renamed is not None else col)
    df.columns = new_names
    return df

def normalize_headers(df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns pl.DataFrame assuming apply_alias fn performed on arg pl.DataFrame.
    The alias (col names) in the header are normalized:
        - '_duplicated_<#>' is removed
        - NOX(ppm) and O2R(%) data (or col aliases) are renamed.
            e.g. nox_node_1, nox_node_2, etc.
        - Other duplicate columns are suffixed with a counter
            e.g. col, col_2, ...
    """
    raw_cols = df.columns
    cleaned_cols = [re.sub(r'_duplicated_\d+', '', col).strip() for col in raw_cols]
    special_bases = {'NOX(ppm)': 'nox_node', 'nox(ppm)': 'nox_node',
                     'nox_node': 'nox_node', 'o2r(%)': 'o2r', 'o2r': 'o2r',}
    special_counts = {base: 0 for base in special_bases.values()}
    generic_counts = {}
    normalized = []
    for col in cleaned_cols:
        lower_col = col.lower()
        special_key = None
        for key in special_bases:
            if lower_col == key or lower_col.startswith(special_bases[key]):
                special_key = special_bases[key]
                break
        if special_key:
            special_counts[special_key] += 1
            new_name = f'{special_key}_{special_counts[special_key]}'
        else:
            generic_counts[col] = generic_counts.get(col, 0) + 1
            new_name = col if generic_counts[col] == 1 else f'{col}_{generic_counts[col]}'
        normalized.append(new_name)
    df.columns = normalized
    
    return df

def load_dataset(dataset_path: str) -> Tuple[pl.DataFrame, Dict]:
    filename = os.path.basename(dataset_path)
    header_row = detect_header_row(dataset_path)
    trial_number = None
    match = re.match(r'^(\d+)_', filename)
    if match:
        trial_number = match.group(1).zfill(2)  # Pad with leading zero 
    if header_row is None:
        print(f'[Warning] Using the first row as header for {dataset_path}')
        header_row = 0
    try:
        df = pl.read_csv(dataset_path, has_header=True, skip_rows=header_row,
                         infer_schema_length=10000, ignore_errors=True, 
                         quote_char="'", null_values=['', 'NULL'])
        df = apply_aliases(df)
        df = normalize_headers(df)
        parts = os.path.normpath(dataset_path).split(os.sep)
        try:
            idx = parts.index('vehicle_v2')
            year = parts[idx + 1] if len(parts) > idx + 1 else None
            make = parts[idx + 2] if len(parts) > idx + 2 else None
            model = parts[idx + 3] if len(parts) > idx + 3 else None
        except ValueError:
            year = make = model = None
        if 'ECM' in filename:
            dataset_type = 'ECM'
        else:
            dataset_type = 'unknown'
        meta = {'file': dataset_path, 'skip_rows': header_row, 'year': year,
                'make': make, 'model': model, 'dataset_type': dataset_type, 
                'trial_number': trial_number}  # Add drive cycle trial to meta
        return df, meta
    
    except Exception as e:
        print(f'[Error] Failed to read {dataset_path}: {e}')
        return pl.DataFrame(), {}

def load_all_datasets(base_path: str) -> List[Tuple[pl.DataFrame, Dict]]:
    dataset_paths = [os.path.join(root, file)
                     for root, _, files in os.walk(base_path)
                     for file in files if file.endswith('.csv')]
    
    return thread_map(load_dataset, dataset_paths, chunksize=1)

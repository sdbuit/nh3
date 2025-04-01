"""
Loads, cleans, preprocesses ECM data, adds geospatial info.

Returns Polars DataFrame.
"""

import os
import re
import json
import logging
import math
import time
from typing import Tuple, Dict, List, Optional
from pathlib import Path

import polars as pl
import numpy as np
from scipy.spatial import cKDTree

# Configuration
try:
    from .config import (
        COLUMN_ALIASES,
        HEADER_DETECTION_KEYWORDS,
        # ROUTE_FILE_PATH,
        DRIVE_CYCLE_ROUTE_PATH,
        EARTH_RADIUS,
    )
except ImportError:
    logging.error('Could not import from .config. Ensure script is run as part of the package or adjust imports.')
    COLUMN_ALIASES = {}
    HEADER_DETECTION_KEYWORDS = ['Time', 'Speed', 'Lat', 'Lon']
    DRIVE_CYCLE_ROUTE_PATH = 'data/geo/drive_cycle_route.json'
    EARTH_RADIUS = 6371000


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


# Helper Functions
def load_route_elevation_data(route_file_path: str) -> Tuple[Optional[np.ndarray], Optional[cKDTree]]:
    """Loads route elevation data from JSON file and builds KDTree."""
    logger.info(f'Loading route elevation data from: {route_file_path}')
    try:
        route_path = Path(route_file_path)
        if not route_path.is_file(): logger.error(f'Route file not found: {route_path}'); return None, None
        with open(route_path, 'r') as f: route_data = json.load(f)
        if not isinstance(route_data, list) or not route_data: logger.error('Route elevation data empty/invalid list format.'); return None, None
        if not isinstance(route_data[0], dict) or not all(k in route_data[0] for k in ['lat', 'lon', 'elev']): logger.error('Route data elements missing required keys (lat, lon, elev) or not dictionaries.'); return None, None
        coords_list, elevations_list = [], []
        for i, item in enumerate(route_data):
             try:
                 coords_list.append([float(item['lat']), float(item['lon'])])
                 elevations_list.append(float(item['elev']))
             except (ValueError, TypeError, KeyError) as e: logger.warning(f'Skipping invalid route data point at index {i}: {item}. Error: {e}'); continue
        if not coords_list: logger.error('No valid coordinates found in route data after filtering.'); return None, None
        coords = np.array(coords_list, dtype=np.float64); elevations = np.array(elevations_list, dtype=np.float64)
        logger.info(f'Building KD-tree from {len(coords)} valid elevation points...'); kdtree = cKDTree(coords); logger.info('KD-tree built successfully.')
        return elevations, kdtree
    except FileNotFoundError: logger.error(f'Route file not found during open: {route_file_path}'); return None, None
    except json.JSONDecodeError: logger.error(f'Failed JSON parse: {route_file_path}'); return None, None
    except Exception as e: logger.error(f'Error loading/processing route data: {e}', exc_info=True); return None, None

# Header Processing Functions
def detect_header_row(filepath: str, keywords: List[str], max_lines_to_check: int = 15) -> int:
    """Detects header row using a smaller, more robust set of keywords."""
    logger.debug(f'Detecting header row in: {filepath} using keywords: {keywords}')
    if not keywords: return 0
    try:
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1']; detected_encoding = None; lines = []
        for enc in encodings_to_try:
            try:
                with open(filepath, 'r', encoding=enc) as f: lines = [f.readline() for _ in range(max_lines_to_check)]
                detected_encoding = enc; logger.debug(f'Read start of file with encoding: {enc}'); break
            except UnicodeDecodeError: logger.debug(f'Encoding {enc} failed for {filepath}, trying next...'); continue
            except Exception as e: logger.error(f'Error reading start of {filepath} with encoding {enc}: {e}'); return 0
        if not detected_encoding: logger.error(f'Could not read start of {filepath} with any tried encoding.'); return 0
        for i, line in enumerate(lines):
            if not line: continue
            line_lower = line.lower(); found_kws = sum(1 for kw in keywords if kw.lower() in line_lower)
            is_header = (found_kws >= min(3, len(keywords) // 2 + 1) or ('time' in line_lower and ('speed' in line_lower or 'rpm' in line_lower)) or ('lat' in line_lower and 'lon' in line_lower))
            if is_header: logger.info(f'Detected header at row {i} in {filepath} (encoding: {detected_encoding}, found {found_kws} keywords)'); return i
    except Exception as e: logger.error(f'Unexpected error during header detection for {filepath}: {e}', exc_info=True); return 0
    logger.warning(f'Header row not reliably detected in {filepath}, defaulting to row 0.'); return 0

def clean_header_name(raw_name: str) -> str:
    """Cleans raw header names: aggressive removal + lowercase."""
    if not isinstance(raw_name, str): raw_name = str(raw_name)
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '', raw_name); cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned.lower()

def apply_aliases_and_normalize(raw_columns: List[str], alias_map: Dict[str, List[str]]) -> Dict[str, str]:
    """Applies aliases based on user structure, handles cleaning & duplicates."""
    rename_map: Dict[str, str] = {}; final_name_counts: Dict[str, int] = {}
    cleaned_alias_lookup: Dict[str, str] = {}
    logger.debug('Pre-cleaning alias variants...')
    for standard_name, raw_variants in alias_map.items():
        for raw_variant in raw_variants:
            cleaned_variant = clean_header_name(raw_variant)
            if not cleaned_variant: continue
            if cleaned_variant in cleaned_alias_lookup: logger.warning(f"Ambiguous alias: '{cleaned_variant}' maps to '{cleaned_alias_lookup[cleaned_variant]}' and '{standard_name}'. Using first.")
            else: cleaned_alias_lookup[cleaned_variant] = standard_name
    logger.debug(f'Created cleaned alias lookup with {len(cleaned_alias_lookup)} entries.')
    processed_final_names = set()
    logger.debug(f'Applying aliases to raw columns: {raw_columns[:10]}...')
    for raw_col in raw_columns:
        original_raw_col = raw_col
        if not raw_col:
            logger.warning('Encountered empty raw column name, assigning placeholder.')
            placeholder_count = final_name_counts.get('_empty_col', 0) + 1; final_name_counts['_empty_col'] = placeholder_count
            final_name = f'_empty_col_{placeholder_count}'; unique_final_name = final_name
            rename_map[original_raw_col] = unique_final_name; processed_final_names.add(unique_final_name); continue
        cleaned_col = clean_header_name(raw_col)
        if not cleaned_col:
             logger.warning(f"Cleaning raw column '{raw_col}' resulted in empty string, assigning placeholder.")
             placeholder_count = final_name_counts.get('_cleaned_empty', 0) + 1; final_name_counts['_cleaned_empty'] = placeholder_count
             final_name = f'_cleaned_empty_{placeholder_count}'; unique_final_name = final_name
             rename_map[original_raw_col] = unique_final_name; processed_final_names.add(unique_final_name); continue
        final_name = cleaned_alias_lookup.get(cleaned_col)
        if final_name is None: final_name = cleaned_col
        current_count = final_name_counts.get(final_name, 0); unique_final_name = final_name
        while unique_final_name in processed_final_names: current_count += 1; unique_final_name = f'{final_name}_{current_count}'
        if current_count > 0: logger.warning(f"Duplicate final name '{final_name}' generated (e.g., from raw '{original_raw_col}'). Renaming to '{unique_final_name}'.")
        final_name_counts[final_name] = current_count; processed_final_names.add(unique_final_name)
        rename_map[original_raw_col] = unique_final_name
    logger.debug(f'Generated rename map for {len(rename_map)} columns.'); return rename_map

# Geospatial Functions
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad; dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)); distance = EARTH_RADIUS * c
    return distance

def calculate_distance_and_grade(df: pl.DataFrame, lat_col: str, lon_col: str, alt_col: str) -> pl.DataFrame:
    file_basename = 'Dist/Grade Calc'; logger.info(f'[{file_basename}] Calculating distance and grade...')
    required_cols = {lat_col, lon_col, alt_col}
    if not required_cols.issubset(df.columns):
        logger.error(f'[{file_basename}] Missing required columns for grade: {required_cols - set(df.columns)}')
        return df.with_columns([pl.lit(None, dtype=pl.Float64).alias('distance_m'), pl.lit(None, dtype=pl.Float64).alias('grade_percent')])
    logger.debug(f'[{file_basename}] Casting geo columns to Float64...')
    df = df.with_columns([pl.col(lat_col).cast(pl.Float64, strict=False), pl.col(lon_col).cast(pl.Float64, strict=False), pl.col(alt_col).cast(pl.Float64, strict=False)])
    logger.debug(f'[{file_basename}] Calculating shifted values...')
    df = df.with_columns([pl.col(lat_col).shift(1).alias('_prev_lat'), pl.col(lon_col).shift(1).alias('_prev_lon'), pl.col(alt_col).shift(1).alias('_prev_alt')])
    logger.debug(f'[{file_basename}] Calculating Haversine distances...')
    distances = pl.struct([pl.col('_prev_lat'), pl.col('_prev_lon'), pl.col(lat_col), pl.col(lon_col)]) \
                  .map_elements(lambda row: haversine_distance(row['_prev_lat'], row['_prev_lon'], row[lat_col], row[lon_col]) if all(row.get(k) is not None for k in ['_prev_lat', '_prev_lon', lat_col, lon_col]) else None, return_dtype=pl.Float64, skip_nulls=False)
    df = df.with_columns(distances.alias('distance_m')); logger.debug(f'[{file_basename}] Finished distance calculation.')
    logger.debug(f'[{file_basename}] Calculating grade percentage...')
    df = df.with_columns(((pl.col(alt_col) - pl.col('_prev_alt')) / pl.when(pl.col('distance_m').is_not_null() & (pl.col('distance_m') != 0)).then(pl.col('distance_m')).otherwise(None) * 100.0).alias('grade_percent'))
    logger.debug(f'[{file_basename}] Finished grade calculation.'); return df.drop(['_prev_lat', '_prev_lon', '_prev_alt'])

def map_elevation_to_vehicle_data(df: pl.DataFrame, kdtree: cKDTree, route_elevations: np.ndarray, lat_col: str, lon_col: str) -> pl.DataFrame:
    file_basename = 'Elevation Map'; alt_col_name = 'altitude_m_mapped'; logger.info(f'[{file_basename}] Mapping elevation data for {df.height} points...')
    if kdtree is None or route_elevations is None: logger.error(f'[{file_basename}] KD-tree/route elevations missing.'); return df.with_columns(pl.lit(None, dtype=pl.Float64).alias(alt_col_name))
    if lat_col not in df.columns or lon_col not in df.columns: logger.error(f'[{file_basename}] Missing lat/lon columns: {lat_col}, {lon_col}'); return df.with_columns(pl.lit(None, dtype=pl.Float64).alias(alt_col_name))
    df_with_idx = df.with_row_count('original_index')
    df_coords_for_query = df_with_idx.select([pl.col('original_index'), pl.col(lat_col).cast(pl.Float64, strict=False), pl.col(lon_col).cast(pl.Float64, strict=False)]).drop_nulls([lat_col, lon_col])
    if df_coords_for_query.height == 0: logger.warning(f'[{file_basename}] No valid coordinates found for KDTree query after filtering.'); return df.with_columns(pl.lit(None, dtype=pl.Float64).alias(alt_col_name))
    vehicle_coords = df_coords_for_query.select([lat_col, lon_col]).to_numpy(); logger.info(f'[{file_basename}] Querying KD-tree with {len(vehicle_coords)} valid points...')
    try:
        distances, indices = kdtree.query(vehicle_coords, k=1)
        mapped_elev_df = pl.DataFrame({'original_index': df_coords_for_query['original_index'], alt_col_name: route_elevations[indices]})
        df_result = df_with_idx.join(mapped_elev_df, on='original_index', how='left'); logger.info(f'[{file_basename}] Elevation mapping complete. Joined back {mapped_elev_df.height} results.')
        return df_result.drop('original_index')
    except Exception as e: logger.error(f'[{file_basename}] Error during KDTree query or elevation mapping: {e}', exc_info=True); return df.with_columns(pl.lit(None, dtype=pl.Float64).alias(alt_col_name))

def preprocess_ecm_file(
    filepath: str,
    aliases: Dict[str, List[str]],
    route_kdtree: Optional[cKDTree],
    route_elevations: Optional[np.ndarray]
    ) -> Optional[pl.DataFrame]:
    """Reads, cleans, preprocesses, adds geospatial info. Returns Polars DataFrame or None."""
    file_basename = os.path.basename(filepath)
    logger.info(f'===> [{file_basename}] Starting Preprocessing <===')

    # 1. Read Data
    header_row_index = detect_header_row(filepath, HEADER_DETECTION_KEYWORDS)
    try:
        try:
            df_pl = pl.read_csv(filepath, has_header=True, skip_rows=header_row_index, infer_schema_length=10000, ignore_errors=True, null_values=['', 'NULL', 'NA', '#N/A', 'None'], encoding='utf-8')
            logger.info(f'[{file_basename}] Read successful with UTF-8. Shape: {df_pl.shape}.')
        except (pl.ComputeError, UnicodeDecodeError) as e:
             if 'invalid utf-8 sequence' in str(e).lower() or isinstance(e, UnicodeDecodeError):
                  logger.warning(f'[{file_basename}] UTF-8 read failed ({type(e).__name__}), trying latin-1...')
                  df_pl = pl.read_csv(filepath, has_header=True, skip_rows=header_row_index, infer_schema_length=10000, ignore_errors=True, null_values=['', 'NULL', 'NA', '#N/A', 'None'], encoding='latin-1')
                  logger.info(f'[{file_basename}] Read successful with latin-1. Shape: {df_pl.shape}.')
             else: raise e
    except Exception as e: logger.error(f'[{file_basename}] Failed CSV read: {e}', exc_info=True); return None

    # 2. Clean and Alias Headers
    if not df_pl.columns: logger.error(f'[{file_basename}] No columns found after reading.'); return None
    try:
        logger.debug(f'[{file_basename}] Raw columns: {df_pl.columns}')
        rename_map = apply_aliases_and_normalize(df_pl.columns, aliases)
        df_pl = df_pl.rename(rename_map); logger.info(f'[{file_basename}] Headers cleaned/aliased. Final columns: {df_pl.columns}')
    except Exception as e: logger.error(f'[{file_basename}] Failed column rename: {e}', exc_info=True); return None

    # 3. Basic Data Type Conversion
    numeric_cols = ['time_s', 'speed_mph', 'speed_kmh', 'latitude', 'longitude', 'engine_rpm', 'altitude_m', 'nox_ppm', 'nh3_ppm', 'o2r_pct', 'cat_temp_c', 'intake_air_temp_c', 'maf_gs', 'engine_load_pct']
    logger.debug(f'[{file_basename}] Casting standard numeric columns...')
    for col in numeric_cols:
        if col in df_pl.columns:
            try: df_pl = df_pl.with_columns(pl.col(col).cast(pl.Float64, strict=False))
            except Exception as e: logger.warning(f"[{file_basename}] Could not cast column '{col}' to Float64: {e}")

    # 4. Geospatial Processing
    lat_col, lon_col = 'latitude', 'longitude'; alt_col_mapped = 'altitude_m_mapped'
    df_pl = map_elevation_to_vehicle_data(df_pl, route_kdtree, route_elevations, lat_col, lon_col)
    df_pl = calculate_distance_and_grade(df_pl, lat_col, lon_col, alt_col_mapped)
    logger.info(f'[{file_basename}] Geospatial processing complete.')

    # 5. Select and Reorder Final Columns
    final_columns_ordered = ['time_s', 'speed_mph', 'speed_kmh', 'latitude', 'longitude', 'altitude_m_mapped', 'distance_m', 'grade_percent', 'engine_rpm', 'engine_load_pct', 'nox_ppm', 'nh3_ppm', 'lambda', 'o2r_pct', 'afr', 'cat_temp_c', 'intake_air_temp_c', 'maf_gs']
    existing_final_cols = [col for col in final_columns_ordered if col in df_pl.columns]; missing_cols = set(final_columns_ordered) - set(existing_final_cols)
    if missing_cols: logger.warning(f'[{file_basename}] Final processed data missing expected columns: {sorted(list(missing_cols))}')
    df_pl = df_pl.select(existing_final_cols)
    logger.info(f'===> [{file_basename}] Finished Preprocessing. Final Shape: {df_pl.shape} <===')
    return df_pl


if __name__ == '__main__':
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    print(f'_PROJECT_ROOT: {_PROJECT_ROOT}')
    TEST_ECM_FILE_PATH = str(_PROJECT_ROOT / 'data/vehicle/2017/Dodge/RAM2500/01_ECM.csv')
    test_aliases = COLUMN_ALIASES
    test_route_elevs, test_route_tree = load_route_elevation_data(str(DRIVE_CYCLE_ROUTE_PATH))
    if Path(TEST_ECM_FILE_PATH).exists() and test_route_tree is not None:
        print(f'Attempting to process: {TEST_ECM_FILE_PATH}')
        _start_time = time.time()
        test_df = preprocess_ecm_file(TEST_ECM_FILE_PATH, test_aliases, test_route_tree, test_route_elevs)
        _end_time = time.time(); print(f'Single file processing time: {_end_time - _start_time:.2f} seconds')
        if test_df is not None:
            print(f'\n--- Test Result for {os.path.basename(TEST_ECM_FILE_PATH)} ---')
            print(f'Processed DataFrame Shape: {test_df.shape}'); print(f'Schema: {test_df.schema}'); print('Head:'); print(test_df.head())
            print('--- End Test Result ---')
        else: print(f'Test processing failed for {TEST_ECM_FILE_PATH}')
    else:
        if not Path(TEST_ECM_FILE_PATH).exists(): print(f'Test ECM file not found: {TEST_ECM_FILE_PATH}')
        if test_route_tree is None: print(f'Route data failed to load or KDTree build failed from: {DRIVE_CYCLE_ROUTE_PATH}')
        print('Cannot run test.')

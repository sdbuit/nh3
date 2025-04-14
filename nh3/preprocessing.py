import os
import re
import json
import logging
import time
from typing import Tuple, Dict, List, Optional, Set
from pathlib import Path

import polars as pl
import numpy as np
from scipy.spatial import cKDTree


try:
    from nh3 import config  # Absolute import from 'nh3'
    COLUMN_ALIASES = config.COLUMN_ALIASES
    HEADER_DETECTION_KEYWORDS = config.HEADER_DETECTION_KEYWORDS
    DRIVE_CYCLE_ROUTE_PATH = str(config.DRIVE_CYCLE_ROUTE_PATH)
    EARTH_RADIUS = config.EARTH_RADIUS

except ImportError:
    logging.warning('Could not import \'config\' from \'nh3\'. Using default values. '
                    'Ensure script is run from the workspace directory or nh3 package is installed.')
    COLUMN_ALIASES: Dict[str, List[str]] = {}
    HEADER_DETECTION_KEYWORDS: List[str] = ['Time', 'Speed', 'Lat', 'Lon']
    DRIVE_CYCLE_ROUTE_PATH: str = 'data/geo/drive_cycle_route.json'
    EARTH_RADIUS: float = 6371000.0


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def clean_header_name(raw_name: str) -> str:
    """Cleans a raw column header name."""
    if not isinstance(raw_name, str):
        raw_name = str(raw_name)
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '', raw_name)
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned.lower()

def _validate_input_columns(df: pl.DataFrame, required_cols: Set[str], context_msg: str) -> bool:
    """Checks if required columns exist in the DataFrame."""
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.error(f'[{context_msg}] Missing required columns: {missing_cols}')
        return False
    return True

def _cast_geo_columns(df: pl.DataFrame, cols: List[str], context_msg: str) -> pl.DataFrame:
    """Casts specified columns to Float64, logging progress."""
    logger.debug(f'[{context_msg}] Casting columns to Float64: {cols}')
    try:
        return df.with_columns(
            [pl.col(c).cast(pl.Float64, strict=False) for c in cols if c in df.columns]
        )
    except Exception as e:
        logger.error(
            f'[{context_msg}] Error casting columns {cols} to Float64: {e}',
            exc_info=True,
        )
        # Return original df if casting fails catastrophically
        # Individual column errors are handled by strict=False
        return df

def _add_shifted_columns(df: pl.DataFrame, cols: List[str], context_msg: str) -> pl.DataFrame:
    """Adds shifted (previous row) values for specified columns."""
    logger.debug(f'[{context_msg}] Calculating shifted values for: {cols}')
    shift_expressions = [
        pl.col(c).shift(1).alias(f'_prev_{c}') for c in cols if c in df.columns
    ]
    if not shift_expressions:
        logger.warning(f'[{context_msg}] No columns found to shift from list: {cols}')
        return df
    return df.with_columns(shift_expressions)

def _calculate_haversine_vectorized(
    lat1_rad: pl.Expr, lon1_rad: pl.Expr, lat2_rad: pl.Expr, lon2_rad: pl.Expr
) -> pl.Expr:
    """
    Calculates Haversine distance using vectorized Polars expressions.
    Expects inputs in radians. Corrected to use expression methods.
    """
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        (dlat / 2).sin().pow(2)
        + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2).sin().pow(2)
    )

    # Clip 'a' to the valid domain [0, 1] for sqrt/arcsin due to potential
    # floating point inaccuracies
    a_clipped = pl.when(a < 0.0).then(0.0).when(a > 1.0).then(1.0).otherwise(a)

    # Calculate 'c' using arcsin: c = 2 * arcsin(sqrt(a))
    # Use .sqrt() and .arcsin() methods
    c = 2 * a_clipped.sqrt().arcsin()

    distance = EARTH_RADIUS * c
    return distance

def _calculate_grade_vectorized(delta_alt: pl.Expr, distance: pl.Expr) -> pl.Expr:
    """Calculates grade percentage using vectorized Polars expressions."""
    # Ensure distance is not null and not zero for safe division
    safe_distance = pl.when(distance.is_not_null() & (distance != 0.0)).then(distance).otherwise(None)
    grade_percent = (delta_alt / safe_distance) * 100.0
    return grade_percent

def load_route_elevation_data(route_file_path: str,) -> Tuple[Optional[np.ndarray], Optional[cKDTree]]:
    """Loads route elevation data from JSON file and builds KDTree."""
    context_msg = 'Route Elevation Load'
    logger.info(f'[{context_msg}] Loading data from: {route_file_path}')
    try:
        route_path = Path(route_file_path)
        if not route_path.is_file():
            logger.error(f'[{context_msg}] Route file not found: {route_path}')
            return None, None

        with open(route_path, 'r') as f:
            route_data = json.load(f)

        if not isinstance(route_data, list) or not route_data:
            logger.error(f'[{context_msg}] Data empty or invalid list format.')
            return None, None
        # if not isinstance(route_data, dict) or not all(
        #     k in route_data for k in ['lat', 'lon', 'elev']
        # ):
        #     logger.error(
        #         f'[{context_msg}] Data elements missing required keys (lat, lon, elev).'
        #     )
        #     return None, None
        # Extract and validate coordinates/elevations
        # coords_list, elevations_list =,
        coords_list, elevations_list = [], []
        for i, item in enumerate(route_data):
            try:
                lat = float(item['lat'])
                lon = float(item['lon'])
                elev = float(item['elev'])
                # Check for typical lat/lon ranges
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    coords_list.append([lat, lon])
                    elevations_list.append(elev)
                else:
                    logger.warning(f'[{context_msg}] Skipping point {i} with invalid lat/lon range: {item}')
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(
                    f'[{context_msg}] Skipping invalid data point at index {i}: {item}. Error: {e}'
                )
                continue

        if not coords_list:
            logger.error(f'[{context_msg}] No valid coordinates found after filtering.')
            return None, None

        coords = np.array(coords_list, dtype=np.float64)
        elevations = np.array(elevations_list, dtype=np.float64)

        logger.info(f'[{context_msg}] Building KD-tree from {len(coords)} valid points...')
        kdtree = cKDTree(coords)
        logger.info(f'[{context_msg}] KD-tree built successfully.')
        return elevations, kdtree

    except FileNotFoundError:
        logger.error(f'[{context_msg}] Route file not found during open: {route_file_path}')
        return None, None
    except json.JSONDecodeError:
        logger.error(f'[{context_msg}] Failed JSON parse: {route_file_path}')
        return None, None
    except Exception as e:
        logger.error(f'[{context_msg}] Error loading/processing route data: {e}', exc_info=True)
        return None, None

def detect_header_row(filepath: str, keywords: List[str], max_lines_to_check: int = 15) -> int:
    """Detects header row using keywords, trying multiple encodings."""
    context_msg = 'Header Detect'
    logger.debug(f'[{context_msg}] Detecting header in: {filepath} using keywords: {keywords}')
    if not keywords:
        logger.warning(f'[{context_msg}] No keywords provided, defaulting to row 0.')
        return 0

    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1']
    detected_encoding = None
    lines = []
    for enc in encodings_to_try:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                lines = [f.readline() for _ in range(max_lines_to_check)]
            detected_encoding = enc
            logger.debug(f'[{context_msg}] Read start of file with encoding: {enc}')
            break
        except UnicodeDecodeError:
            logger.debug(f'[{context_msg}] Encoding {enc} failed, trying next...')
            continue
        except Exception as e:
            logger.error(f'[{context_msg}] Error reading start of {filepath} with {enc}: {e}')
            return 0

    if not detected_encoding:
        logger.error(f'[{context_msg}] Could not read start of {filepath} with any tried encoding.')
        return 0

    # Heuristic for header detection
    for i, line in enumerate(lines):
        if not line:  # Skip empty lines
            continue
        line_lower = line.lower()
        # Count how many keywords (case-insensitive) appear in the line
        found_kws = sum(1 for kw in keywords if kw.lower() in line_lower)

        # Define conditions for considering a line as a header
        # Condition 1: Found a significant number of keywords
        # (e.g., >= 3, or more than half the keywords if fewer than 5 total)
        cond1 = found_kws >= min(3, len(keywords) // 2 + 1)
        # Condition 2: Found common pairs like 'time' and ('speed' or 'rpm')
        cond2 = 'time' in line_lower and ('speed' in line_lower or 'rpm' in line_lower)
        # Condition 3: Found common geo pair 'lat' and 'lon'
        cond3 = 'lat' in line_lower and 'lon' in line_lower

        if cond1 or cond2 or cond3:
            logger.info(f'[{context_msg}] Detected header at row {i} in {filepath} '
                        f'(encoding: {detected_encoding}, found {found_kws} keywords)')
            return i
    logger.warning(f'[{context_msg}] Header not reliably detected in {filepath}, defaulting to row 0.')
    return 0

def apply_aliases_and_normalize(
    raw_columns: List[str], alias_map: Dict[str, List[str]]
) -> Dict[str, str]:
    """Applies aliases, cleans names, and handles potential duplicates."""
    context_msg = 'Alias/Normalize'
    rename_map: Dict[str, str] = {}
    final_name_counts: Dict[str, int] = {}
    cleaned_alias_lookup: Dict[str, str] = {}

    # Pre-process the alias map for faster lookup
    logger.debug(f'[{context_msg}] Pre-cleaning alias variants...')
    for standard_name, raw_variants in alias_map.items():
        for raw_variant in raw_variants:
            cleaned_variant = clean_header_name(raw_variant)
            if not cleaned_variant:  # Skip empty results from cleaning
                continue
            if (
                cleaned_variant in cleaned_alias_lookup
                and cleaned_alias_lookup[cleaned_variant] != standard_name
            ):
                logger.warning(
                    f'[{context_msg}] Ambiguous alias: \'{cleaned_variant}\' maps to '
                    f'\'{cleaned_alias_lookup[cleaned_variant]}\' and \'{standard_name}\'. '
                    f'Using first mapping: \'{cleaned_alias_lookup[cleaned_variant]}\'.'
                )
            elif cleaned_variant not in cleaned_alias_lookup:
                cleaned_alias_lookup[cleaned_variant] = standard_name
    logger.debug(f'[{context_msg}] Created cleaned alias lookup with {len(cleaned_alias_lookup)} entries.')

    processed_final_names: Set[str] = set()
    logger.debug(f'[{context_msg}] Applying aliases to {len(raw_columns)} raw columns...')

    for raw_col in raw_columns:
        original_raw_col = raw_col  # Keep original for the rename map key

        if not raw_col or (isinstance(raw_col, str) and raw_col.strip() == ''):
            logger.warning(f'[{context_msg}] Encountered empty raw column name, assigning placeholder.')
            # Generate a unique placeholder name like _empty_col_1, _empty_col_2
            placeholder_count = final_name_counts.get('_empty_col', 0) + 1
            final_name_counts['_empty_col'] = placeholder_count
            final_name = f'_empty_col_{placeholder_count}'
            unique_final_name = final_name  # Assume placeholder is unique initially
        else:
            cleaned_col = clean_header_name(raw_col)
            if not cleaned_col:
                logger.warning(
                    f'[{context_msg}] Cleaning raw column \'{raw_col}\' resulted in empty string, assigning placeholder.'
                )
                # Generate a unique placeholder like _cleaned_empty_1
                placeholder_count = final_name_counts.get('_cleaned_empty', 0) + 1
                final_name_counts['_cleaned_empty'] = placeholder_count
                final_name = f'_cleaned_empty_{placeholder_count}'
                unique_final_name = final_name
            else:
                # Find standard name from alias map, otherwise use cleaned name
                final_name = cleaned_alias_lookup.get(cleaned_col, cleaned_col)

                # Handle potential duplicates in the final names
                current_count = final_name_counts.get(final_name, 0)
                unique_final_name = final_name
                # Append suffix (_1, _2, etc.) if name already exists
                while unique_final_name in processed_final_names:
                    current_count += 1
                    unique_final_name = f'{final_name}_{current_count}'

                if current_count > 0:
                    logger.warning(
                        f'[{context_msg}] Duplicate final name \'{final_name}\' generated '
                        f'(e.g., from raw \'{original_raw_col}\'). Renaming to \'{unique_final_name}\'.'
                    )
                # Update the count for the base name (before suffix)
                final_name_counts[final_name] = current_count

        # Add the unique name to the set and the rename map
        processed_final_names.add(unique_final_name)
        # Ensure the key in rename_map is the original raw column name
        rename_map[original_raw_col] = unique_final_name

    logger.debug(f'[{context_msg}] Generated rename map for {len(rename_map)} columns.')
    return rename_map

def calculate_distance_and_grade(
    df: pl.DataFrame, lat_col: str, lon_col: str, alt_col: str
) -> pl.DataFrame:
    """
    Calculates Haversine distance and grade percentage between consecutive points
    using vectorized Polars operations.

    Args:
        df: Input Polars DataFrame.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        alt_col: Name of the altitude column (used for grade).

    Returns:
        Polars DataFrame with added 'distance_m' and 'grade_percent' columns.
    """
    context_msg = 'Dist/Grade Calc'
    logger.info(f'[{context_msg}] Calculating distance and grade...')

    required_cols = {lat_col, lon_col, alt_col}
    if not _validate_input_columns(df, required_cols, context_msg):
        return df.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias('distance_m'),
                pl.lit(None, dtype=pl.Float64).alias('grade_percent'),
            ]
        )

    # Preparation
    df = _cast_geo_columns(df, [lat_col, lon_col, alt_col], context_msg)
    df = _add_shifted_columns(df, [lat_col, lon_col, alt_col], context_msg)

    # Define shifted column names first (needed for debug print)
    prev_lat_col = f'_prev_{lat_col}'
    prev_lon_col = f'_prev_{lon_col}'
    prev_alt_col = f'_prev_{alt_col}'

    # === START DEBUG BLOCK ===
    print('\n--- Debugging Grade Calculation ---')
    # Select relevant columns for inspection
    # Ensure columns exist before selecting to avoid errors if _add_shifted_columns failed
    cols_to_debug = [c for c in [lat_col, lon_col, alt_col, prev_lat_col, prev_lon_col, prev_alt_col] if c in df.columns]
    if cols_to_debug:
        try:
            debug_df = df.select(cols_to_debug).head(10)  # Look at the first 10 rows
            print('Input and Shifted Columns (Head):')
            print(debug_df)
        except Exception as e:
            print(f'Error printing debug columns: {e}')
    else:
        print('Could not find columns for debug printing.')
    # === END DEBUG BLOCK ===

    # Vectorized Calculation
    try:
        logger.debug(f'[{context_msg}] Calculating Haversine distances (vectorized)...')
        df = df.with_columns(
            # Convert degrees to radians
            pl.col(lat_col).radians().alias('_lat2_rad'),
            pl.col(lon_col).radians().alias('_lon2_rad'),
            pl.col(prev_lat_col).radians().alias('_lat1_rad'),
            pl.col(prev_lon_col).radians().alias('_lon1_rad'),
        ).with_columns(
            # Calculate Haversine distance
            _calculate_haversine_vectorized(
                pl.col('_lat1_rad'),
                pl.col('_lon1_rad'),
                pl.col('_lat2_rad'),
                pl.col('_lon2_rad'),
            ).alias('distance_m')
        )
        logger.debug(f'[{context_msg}] Finished distance calculation.')

        # === START DEBUG BLOCK ===
        print('\nCalculated Distances (Head):')
        print(df.select(['distance_m']).head(10))
        print('\nDistance Stats:')
        # print(df.select(pl.col('distance_m').describe())) # Get min, max, mean etc.
        print(df.select('distance_m').describe())
        # === END DEBUG BLOCK ===

        logger.debug(f'[{context_msg}] Calculating grade percentage (vectorized)...')
        df = df.with_columns((pl.col(alt_col) - pl.col(prev_alt_col)).alias('_delta_alt'))

        # === START DEBUG BLOCK ===
        print('\nCalculated Altitude Change (Head):')
        print(df.select(['_delta_alt']).head(10))
        print('\nAltitude Change Stats:')
        # print(df.select(pl.col('_delta_alt').describe()))
        print(df.select('_delta_alt').describe())
        # === END DEBUG BLOCK ===

        df = df.with_columns(
            # Calculate grade percentage safely
            _calculate_grade_vectorized(pl.col('_delta_alt'), pl.col('distance_m')).alias('grade_percent'))
        logger.debug(f'[{context_msg}] Finished grade calculation.')

    except Exception as e:
        logger.error(f'[{context_msg}] Error during vectorized distance/grade calculation: {e}', exc_info=True,)
        # Add null columns in case of calculation error
        return df.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias('distance_m'),
                pl.lit(None, dtype=pl.Float64).alias('grade_percent'),
            ]
        ).drop(  # Ensure temp columns are dropped even on error
            [c for c in df.columns if c.startswith('_prev_') or c.startswith('_lat') or c.startswith('_lon') or c == '_delta_alt']
        )

    # Cleanup
    temp_cols_to_drop = [
        prev_lat_col, prev_lon_col, prev_alt_col,
        '_lat1_rad', '_lon1_rad', '_lat2_rad', '_lon2_rad',
        '_delta_alt',
    ]
    # Drop only columns that actually exist to avoid errors
    existing_temp_cols = [col for col in temp_cols_to_drop if col in df.columns]
    logger.debug(f'[{context_msg}] Dropping temporary columns: {existing_temp_cols}')

    # === START DEBUG BLOCK ===
    print('\nFinal Grade Output (Head):')
    print(df.select(['grade_percent']).head(10))
    print('\nGrade Stats:')
    # print(df.select(pl.col('grade_percent').describe()))
    print(df.select('grade_percent').describe())
    print('--- End Debugging Info ---')
    # === END DEBUG BLOCK ===

    return df.drop(existing_temp_cols)

def map_elevation_to_vehicle_data(
    df: pl.DataFrame,
    kdtree: Optional[cKDTree],
    route_elevations: Optional[np.ndarray],  # Made Optional explicit
    lat_col: str,
    lon_col: str,
) -> pl.DataFrame:
    """Maps elevation data to vehicle data using KDTree nearest neighbor search."""
    context_msg = 'Elevation Map'
    alt_col_name = 'altitude_m_mapped'

    logger.info(f'[{context_msg}] Mapping elevation for {df.height} points...')

    # Input Validation
    if kdtree is None or route_elevations is None:
        logger.error(f'[{context_msg}] KD-tree or route elevations data is missing.')
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias(alt_col_name))

    required_cols = {lat_col, lon_col}
    if not _validate_input_columns(df, required_cols, context_msg):
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias(alt_col_name))

    # Prepare Data for Query
    # Add original index to join back later, cast, and drop null coordinates
    df_with_idx = df.with_row_count('original_index')
    df_coords_for_query = (
        df_with_idx.select(
            [
                pl.col('original_index'),
                pl.col(lat_col).cast(pl.Float64, strict=False),
                pl.col(lon_col).cast(pl.Float64, strict=False),
            ]
        )
        .drop_nulls([lat_col, lon_col])
    )

    if df_coords_for_query.height == 0:
        logger.warning(f'[{context_msg}] No valid (non-null) coordinates found for KDTree query.')
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias(alt_col_name))

    vehicle_coords = df_coords_for_query.select([lat_col, lon_col]).to_numpy()

    logger.info(f'[{context_msg}] Querying KD-tree with {len(vehicle_coords)} valid points...')
    try:
        distances, indices = kdtree.query(vehicle_coords, k=1)
        mapped_elev_df = pl.DataFrame(
            {
                'original_index': df_coords_for_query['original_index'],
                alt_col_name: route_elevations[indices],
                # Optionally include distance to nearest point:
                # 'nearest_route_point_dist_m': distances
            }
        )
        df_result = df_with_idx.join(mapped_elev_df, on='original_index', how='left')
        logger.info(f'[{context_msg}] Elevation mapping complete. Joined back {mapped_elev_df.height} results.')
        return df_result.drop('original_index')  # Remove the temporary index column

    except Exception as e:
        logger.error(f'[{context_msg}] Error during KDTree query or elevation mapping: {e}', exc_info=True)
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias(alt_col_name))

def preprocess_ecm_file(
    filepath: str,
    aliases: Dict[str, List[str]],
    route_kdtree: Optional[cKDTree],
    route_elevations: Optional[np.ndarray]
) -> Optional[pl.DataFrame]:
    """Reads, cleans, preprocesses ECM data, adding geospatial info."""
    file_basename = os.path.basename(filepath)
    context_msg = f'Preprocess: {file_basename}'
    logger.info(f'===> [{context_msg}] Starting Preprocessing <===')

    header_row_index = detect_header_row(filepath, HEADER_DETECTION_KEYWORDS)
    df_pl = None
    try:
        df_pl = pl.read_csv(
            filepath,
            has_header=True,
            skip_rows=header_row_index,
            infer_schema_length=10000,
            ignore_errors=True,
            null_values=['', 'NULL', '#N/A'],
            encoding='utf-8',
        )
        logger.info(f'[{context_msg}] Read successful with UTF-8. Shape: {df_pl.shape}.')
    except (pl.ComputeError, UnicodeDecodeError) as e_utf8:
        if 'invalid utf-8 sequence' in str(e_utf8).lower() or isinstance(e_utf8, UnicodeDecodeError):
            logger.warning(f'[{context_msg}] UTF-8 read failed ({type(e_utf8).__name__}), trying latin-1...')
            try:
                df_pl = pl.read_csv(
                    filepath,
                    has_header=True,
                    skip_rows=header_row_index,
                    infer_schema_length=10000,
                    ignore_errors=True,
                    null_values=[''],
                    encoding='latin-1',
                )
                logger.info(f'[{context_msg}] Read successful with latin-1. Shape: {df_pl.shape}.')
            except Exception as e_latin1:
                logger.error(f'[{context_msg}] Failed CSV read with latin-1 as well: {e_latin1}', exc_info=True)
                return None
        else:
            logger.error(f'[{context_msg}] Failed CSV read (non-encoding error): {e_utf8}', exc_info=True)
            return None
    except Exception as e_other:
        logger.error(f'[{context_msg}] Unexpected error during CSV read: {e_other}', exc_info=True)
        return None

    if df_pl is None or df_pl.height == 0:
        logger.error(f'[{context_msg}] DataFrame is empty or None after reading.')
        return None
    if not df_pl.columns:
        logger.error(f'[{context_msg}] No columns found after reading.')
        return None

    # Header Cleaning and Aliasing
    # try:
    #     logger.debug(f'[{context_msg}] Raw columns: {df_pl.columns}')
    #     rename_map = apply_aliases_and_normalize(df_pl.columns, aliases)
    #     df_pl = df_pl.rename(rename_map)
    #     logger.info(f'[{context_msg}] Headers cleaned/aliased. Final columns preview: {df_pl.columns[:10]}')
    # except Exception as e:
    #     logger.error(f'[{context_msg}] Failed column rename/aliasing: {e}', exc_info=True)
    #     # Return None if essential renaming fails
    #     return None

    try:
        logger.debug(f'[{context_msg}] Raw columns: {df_pl.columns}')
        rename_map = apply_aliases_and_normalize(df_pl.columns, aliases)

        # START DEBUG
        print('\n=== Rename Map Generated ===')
        # Limit printing if map is huge
        print({k: v for i, (k, v) in enumerate(rename_map.items()) if i < 20})  # Print first 20 items
        # END DEBUG

        df_pl = df_pl.rename(rename_map)

        # START DEBUG
        print('\n=== Columns After Renaming ===')
        print(df_pl.columns)
        # END DEBUG

        logger.info(f'[{context_msg}] Headers cleaned/aliased. Final columns preview: {df_pl.columns[:10]}')
    except Exception as e:
        logger.error(f'[{context_msg}] Failed column rename/aliasing: {e}', exc_info=True)
        return None

    numeric_cols = [
        'time_s', 'speed_mph', 'speed_kmh', 'latitude', 'longitude',
        'engine_rpm', 'altitude_m', 'nox_ppm', 'nh3_ppm', 'o2r_pct',
        'cat_temp_c', 'intake_air_temp_c', 'maf_gs', 'engine_load_pct',
    ]
    logger.debug(f'[{context_msg}] Casting standard numeric columns...')
    cast_expressions = list()
    for col in numeric_cols:
        if col in df_pl.columns:
            # Check if not already float to avoid unnecessary casts
            if not isinstance(df_pl.schema[col], pl.Float64):
                cast_expressions.append(pl.col(col).cast(pl.Float64, strict=False))
            # else: logger.debug(f'[{context_msg}] Column \'{col}\' is already Float64.') # Optional: more verbose logging
        # else: logger.debug(f'[{context_msg}] Numeric column \'{col}\' not found, skipping cast.') # Optional: more verbose logging

    if cast_expressions:
        try:
            df_pl = df_pl.with_columns(cast_expressions)
            logger.info(f'[{context_msg}] Attempted casting for {len(cast_expressions)} numeric columns.')
        except Exception as e:
            logger.warning(f'[{context_msg}] Error during bulk casting of numeric columns: {e}', exc_info=True)

    # Geospatial Processing
    lat_col, lon_col = 'latitude', 'longitude'
    alt_col_mapped = 'altitude_m_mapped'

    df_pl = map_elevation_to_vehicle_data(df_pl, route_kdtree, route_elevations, lat_col, lon_col)

    if alt_col_mapped in df_pl.columns:
        df_pl = calculate_distance_and_grade(df_pl, lat_col, lon_col, alt_col_mapped)
        logger.info(f'[{context_msg}] Distance and grade calculated using \'{alt_col_mapped}\'.')
    else:
        logger.warning(f'[{context_msg}] Mapped altitude column \'{alt_col_mapped}\' not found. ')
        df_pl = df_pl.with_columns([
            pl.lit(None, dtype=pl.Float64).alias('distance_m'),
            pl.lit(None, dtype=pl.Float64).alias('grade_percent'),
        ])

    # Final Column Selection and Ordering
    final_columns_ordered = [
        'time_s', 'speed_mph', 'speed_kmh', 'latitude', 'longitude',
        'altitude_m_mapped',  # Use the mapped altitude
        'distance_m', 'grade_percent', 'engine_rpm', 'engine_load_pct',
        'nox_ppm', 'nh3_ppm', 'lambda', 'o2r_pct', 'afr', 'cat_temp_c',
        'intake_air_temp_c', 'maf_gs',
        # TODO Add other columns as needed.
    ]

    existing_final_cols = [col for col in final_columns_ordered if col in df_pl.columns]
    missing_cols = set(final_columns_ordered) - set(existing_final_cols)

    if missing_cols:
        logger.warning(f'[{context_msg}] Final processed data missing expected columns: {sorted(list(missing_cols))}')
    df_pl = df_pl.select(existing_final_cols)
    logger.info(f'===> [{context_msg}] Finished Preprocessing. Final Shape: {df_pl.shape} <===')
    return df_pl


if __name__ == '__main__':
    try:
        _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    except NameError:
        # Fallback if __file__ is not defined (e.g., interactive environment)
        _PROJECT_ROOT = Path('.').resolve()
    print(f'Project Root (estimated): {_PROJECT_ROOT}')
    TEST_ECM_FILE_PATH = _PROJECT_ROOT / 'data' / 'vehicle' / '2017' / 'Dodge' / 'RAM2500' / '02_ECM.csv'
    ROUTE_DATA_PATH = _PROJECT_ROOT / DRIVE_CYCLE_ROUTE_PATH

    print(f'Attempting to load route data from: {ROUTE_DATA_PATH}')
    test_route_elevs, test_route_tree = load_route_elevation_data(str(ROUTE_DATA_PATH))

    test_aliases = COLUMN_ALIASES

    if TEST_ECM_FILE_PATH.exists() and test_route_tree is not None:
        print(f'Attempting to process test file: {TEST_ECM_FILE_PATH}')
        _start_time = time.time()
        test_df = preprocess_ecm_file(str(TEST_ECM_FILE_PATH), test_aliases, test_route_tree, test_route_elevs)
        _end_time = time.time()
        print(f'Single file processing time: {_end_time - _start_time:.2f} seconds')

        if test_df is not None:
            print(f'\n=== Test Result for {TEST_ECM_FILE_PATH.name} ===')
            print(f'Processed DataFrame Shape: {test_df.shape}')
            with pl.Config(tbl_rows=10, tbl_cols=20):
                print(f'Schema: {test_df.schema}')
                print('Head:')
                print(test_df.head())
            print('=== End Test Result ===')
        else:
            print(f'Test processing failed for {TEST_ECM_FILE_PATH.name}')
    else:
        if not TEST_ECM_FILE_PATH.exists():
            print(f'Test ECM file not found: {TEST_ECM_FILE_PATH}')
        if test_route_tree is None:
            print(f'Route data failed to load or KDTree build failed from: {ROUTE_DATA_PATH}')
        print('Cannot run test processing.')

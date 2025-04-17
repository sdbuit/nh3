import os
from pathlib import Path
import logging
import sys
import re
import time # Added for potential timing logs
import concurrent.futures # Keep for parallel processing
from typing import Any, Optional

import polars as pl
import numpy as np
# Removed matplotlib imports as this script focuses on processing, not plotting
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# try:
#     import nh3
# except ImportError:
#     current_dir = Path(__file__).resolve().parent
#     nh3_parent_dir = current_dir.parent # Assumes 'scripts' is one level down from nh3's parent
#     if nh3_parent_dir not in sys.path:
#         logger.info(f"Adding {nh3_parent_dir} to sys.path")
#         sys.path.insert(0, str(nh3_parent_dir))

try:
    from nh3.preprocessing import preprocess_ecm_file, load_route_elevation_data
    from nh3.vsp_compute import VSPCalculator
    from nh3.config import (
        DATA_DIR,
        OUTPUT_DIR,
        DRIVE_CYCLE_ROUTE_PATH,
        COLUMN_ALIASES,
        REQUIRED_COLUMNS_POST_PROCESSING, # Ensure this includes 'grade_percent'
        MAX_WORKERS,
        DEFAULT_VEHICLE_PARAMS)

    from nh3.extensions import fast_grade_calculation, has_cpp_extensions
    from nh3.optimized_computation import calculate_grade_with_smoothing as python_grade_fallback

    logger.info(f"C++ Grade Calculation Extension Loaded: {has_cpp_extensions}")

except ImportError as e:
     logging.critical(f'Failed to import necessary modules: {e}', exc_info=True)
     sys.exit(1)


# --- Constants (Best Practice: Define constants clearly) ---
# Define column names used frequently
LAT_COL = 'latitude'
LON_COL = 'longitude'
ALT_MAPPED_COL = 'altitude_m_mapped'
GRADE_COL = 'grade_percent'
VSP_COL = 'vsp_kw_tonne' # Standardized VSP column name

# Ensure GRADE_COL is in the required columns for validation and saving
if GRADE_COL not in REQUIRED_COLUMNS_POST_PROCESSING:
    logger.warning(f"'{GRADE_COL}' not found in REQUIRED_COLUMNS_POST_PROCESSING from config. Adding it.")
    # Convert Set to list, add, convert back to Set if needed, or just use as list
    REQUIRED_COLUMNS_POST_PROCESSING = set(list(REQUIRED_COLUMNS_POST_PROCESSING) + [GRADE_COL])


class DatasetFinder:
    """Finds ECM datasets based on Year/Make/Model dir structure"""
    def __init__(self, base_data_path: Path):
        # Use Path object methods for joining paths
        self.vehicle_path = base_data_path / 'vehicle_v2'
        if not self.vehicle_path.is_dir():
            logger.warning(f'Base vehicle data directory not found: {self.vehicle_path}')
            self.vehicle_path = None

    def find_datasets(self) -> list[dict[str, Any]]:
        datasets_info = []
        if not self.vehicle_path:
            logger.error('Vehicle path invalid.')
            return datasets_info

        logger.info(f'Searching for datasets under: {self.vehicle_path}')
        # Use rglob for recursive search, simplifying the pattern
        for ecm_file_path in self.vehicle_path.rglob('*/*/*/*_ECM.csv'):
            try:
                relative_path = ecm_file_path.relative_to(self.vehicle_path)
                parts = relative_path.parts
                # Check if the structure is as expected (year/make/model/filename)
                if len(parts) == 4:
                    year, make, model, filename = parts
                    # Basic validation for year
                    if not (year.isdigit() and len(year) == 4):
                        logger.warning(f'Skipping non-4-digit year directory structure: {ecm_file_path}')
                        continue

                    match = re.match(r'(\d+)_ECM\.csv', filename, re.IGNORECASE)
                    trial_number = match.group(1) if match else 'unknown'

                    datasets_info.append({
                        'filepath': str(ecm_file_path.resolve()),
                        'year': year,
                        'make': make,
                        'model': model,
                        'trial_number': trial_number,
                        'relative_path': str(relative_path)
                    })
                else:
                    logger.warning(f'Skipping unexpected path structure: {ecm_file_path}')
            except Exception as e:
                logger.error(f'Error processing path {ecm_file_path}: {e}', exc_info=True)

        logger.info(f'Found {len(datasets_info)} valid ECM datasets.')
        return datasets_info


class AnalysisPipeline:
    """Loading, processing (incl VSP), validation, and saving."""
    def __init__(self):
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        self.drive_cycle_route_path = DRIVE_CYCLE_ROUTE_PATH
        self.aliases = COLUMN_ALIASES
        # Use the potentially updated set including GRADE_COL
        self.required_columns = REQUIRED_COLUMNS_POST_PROCESSING
        self.max_workers = MAX_WORKERS
        self.default_vsp_params = DEFAULT_VEHICLE_PARAMS
        self.finder = DatasetFinder(self.data_dir)
        self.route_tree = None
        self.route_elevs = None
        self.vsp_calculator = None

    def _load_shared_resources(self) -> bool:
        """Loads KDTree/Elevations and initializes VSP calculator."""
        logger.info('Loading shared resources...')
        try:
            self.route_elevs, self.route_tree = load_route_elevation_data(str(self.drive_cycle_route_path))
            if self.route_tree is None or self.route_elevs is None:
                # Be more specific about the failure
                raise RuntimeError('Failed to load route data or build KDTree.')

            self.vsp_calculator = VSPCalculator(**self.default_vsp_params)
            logger.info('Shared resources loaded successfully.')
            return True
        except Exception as e:
            logger.critical(f'Fatal error loading shared resources: {e}', exc_info=True)
            return False

    # *** MODIFIED PROCESSING STEP ***
    def _process_single_file(self, dataset_info: dict[str, Any]) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
        """
        Wrapper to preprocess, calculate grade (using extension), calculate VSP,
        and return results as a dictionary.
        """
        filepath = dataset_info['filepath']
        file_basename = os.path.basename(filepath)
        processed_df = None
        output_data_dict = None

        try:
            logger.debug(f'[{file_basename}] Starting preprocessing...')
            # Preprocess (maps elevation, calculates initial grade - which we will overwrite)
            processed_df = preprocess_ecm_file(filepath, self.aliases, self.route_tree, self.route_elevs)

            if processed_df is None or processed_df.is_empty():
                logger.warning(f'[{file_basename}] Preprocessing failed or returned empty DataFrame.')
                return None, dataset_info

            logger.debug(f'[{file_basename}] Preprocessing successful (Shape: {processed_df.shape}).')

            # --- Grade Calculation using Extension/Fallback ---
            grade_calculation_success = False
            if all(col in processed_df.columns for col in [LAT_COL, LON_COL, ALT_MAPPED_COL]):
                try:
                    logger.debug(f"[{file_basename}] Extracting Lat/Lon/Alt arrays for grade calculation.")
                    lat_np = processed_df[LAT_COL].to_numpy()
                    lon_np = processed_df[LON_COL].to_numpy()
                    alt_np = processed_df[ALT_MAPPED_COL].to_numpy()

                    # Default parameters for grade calculation (can be moved to config)
                    distance_threshold = 10.0
                    smoothing_window = 5

                    if has_cpp_extensions:
                        logger.info(f'[{file_basename}] Using C++ extension for grade calculation.')
                        grade_func = fast_grade_calculation
                    else:
                        logger.info(f'[{file_basename}] Using Python fallback (Numba) for grade calculation.')
                        grade_func = python_grade_fallback # Use the imported Numba func

                    start_grade_time = time.perf_counter()
                    calculated_grades = grade_func(
                        lat_np, lon_np, alt_np,
                        distance_threshold=distance_threshold, # Pass as kwargs if needed
                        smoothing_window=smoothing_window
                    )
                    end_grade_time = time.perf_counter()
                    logger.debug(f"[{file_basename}] Grade calculation took {end_grade_time - start_grade_time:.4f}s")

                    # Overwrite the grade_percent column
                    if isinstance(calculated_grades, np.ndarray) and len(calculated_grades) == processed_df.height:
                        processed_df = processed_df.with_columns(
                            pl.Series(name=GRADE_COL, values=calculated_grades)
                        )
                        logger.debug(f'[{file_basename}] Replaced "{GRADE_COL}" with new calculation.')
                        grade_calculation_success = True
                    else:
                        logger.error(f'[{file_basename}] Grade calculation result type/length mismatch. Expected {processed_df.height}, got {type(calculated_grades)} len {len(calculated_grades) if hasattr(calculated_grades, "__len__") else "N/A"}')

                except Exception as e_grade:
                    logger.error(f'[{file_basename}] Error during grade calculation or replacement: {e_grade}', exc_info=True)
            else:
                logger.warning(f"[{file_basename}] Missing columns for grade calculation ({LAT_COL}, {LON_COL}, {ALT_MAPPED_COL}). Skipping recalculation.")

            if not grade_calculation_success:
                 logger.warning(f"[{file_basename}] Proceeding with grade calculated during preprocessing (if any).")


            # --- VSP Calculation ---
            vsp_results = None
            if self.vsp_calculator:
                logger.debug(f'[{file_basename}] Calling VSPCalculator.calculate_vsp...')
                try:
                    # Pass the DataFrame (which now has the potentially updated grade)
                    vsp_results = self.vsp_calculator.calculate_vsp(processed_df)
                    if isinstance(vsp_results, np.ndarray):
                        logger.debug(f'[{file_basename}] VSP calculation successful.')
                    else:
                        logger.error(f'[{file_basename}] VSP calculation did NOT return NumPy array!')
                        vsp_results = None # Ensure it's None if failed
                except Exception as e_vsp_calc:
                    logger.error(f'[{file_basename}] Error during VSP calculation call: {e_vsp_calc}', exc_info=True)
                    vsp_results = None # Ensure it's None if failed
            else:
                logger.error(f'[{file_basename}] VSP Calculator not initialized.')

            # --- Prepare Output Dictionary ---
            final_df_for_extraction = processed_df # Start with the potentially modified df
            try:
                # Add VSP results to the DataFrame temporarily for extraction
                if isinstance(vsp_results, np.ndarray) and len(vsp_results) == final_df_for_extraction.height:
                    final_df_for_extraction = final_df_for_extraction.with_columns(
                        pl.Series(name=VSP_COL, values=vsp_results)
                    )
                else:
                    logger.warning(f'[{file_basename}] VSP result invalid/length mismatch or calc failed. Adding NULL column for "{VSP_COL}".')
                    # Ensure VSP column exists, even if null, if it's required
                    if VSP_COL in self.required_columns:
                         final_df_for_extraction = final_df_for_extraction.with_columns(
                             pl.lit(None, dtype=pl.Float64).alias(VSP_COL)
                         )
            except Exception as e_add_vsp:
                 logger.error(f'[{file_basename}] Error adding VSP column for extraction: {e_add_vsp}', exc_info=True)
                 if VSP_COL in self.required_columns and VSP_COL not in final_df_for_extraction.columns:
                      final_df_for_extraction = final_df_for_extraction.with_columns(pl.lit(None, dtype=pl.Float64).alias(VSP_COL))

            # Extract required columns to dictionary
            output_data_dict = {}
            logger.debug(f'[{file_basename}] Extracting final columns to dict. Required: {self.required_columns}')
            for col_name in self.required_columns:
                 if col_name in final_df_for_extraction.columns:
                      try:
                           output_data_dict[col_name] = final_df_for_extraction[col_name].to_numpy()
                      except Exception as e_extract:
                           logger.error(f'[{file_basename}] Failed to extract column "{col_name}" to NumPy: {e_extract}', exc_info=True)
                           # Decide how to handle: skip column or add NaNs? Skipping for now.
                 else:
                      logger.warning(f'[{file_basename}] Required column "{col_name}" not found in final DataFrame for extraction.')

            # Add essential metadata keys for saving/identification later
            for meta_key in ['year', 'make', 'model', 'trial_number', 'filepath']:
                if meta_key in dataset_info:
                     output_data_dict[meta_key] = dataset_info[meta_key]

        except Exception as e:
            logger.error(f"Unhandled error during processing for {filepath}: {e}", exc_info=True)
            return None, dataset_info # Return None for data dict on error

        logger.debug(f'[{file_basename}] Finishing _process_single_file. Returning dict with keys: {list(output_data_dict.keys()) if output_data_dict else "None"}')
        return output_data_dict, dataset_info # Return dict and original metadata

    def _run_parallel_processing(self, datasets_to_process: list[dict[str, Any]]) -> list[tuple[Optional[dict[str, Any]], dict[str, Any]]]:
        """Processes datasets in parallel, returning (Data Dict, metadata) tuples."""
        results_with_meta = []
        num_datasets = len(datasets_to_process)
        logger.info(f'Starting parallel processing for {num_datasets} datasets with {self.max_workers} workers.')

        # Check shared resources *before* starting pool
        if self.route_tree is None or self.route_elevs is None or self.vsp_calculator is None:
             logger.error('Cannot start processing: Shared resources not loaded.')
             # Return list indicating failure for all
             return [(None, meta) for meta in datasets_to_process]

        # Use ProcessPoolExecutor for CPU-bound tasks (like Numba/C++ calls)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks: map dataset_info -> future
            future_to_meta = {executor.submit(self._process_single_file, d): d for d in datasets_to_process}
            logger.info('Waiting for processing tasks to complete...')

            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_meta), total=num_datasets, desc="Processing Files", unit="file", ncols=100):
                metadata = future_to_meta[future] # Get original metadata back
                try:
                    # Result is now a tuple (data_dict, original_metadata_again)
                    data_dict, _ = future.result() # Unpack the result tuple
                    results_with_meta.append((data_dict, metadata)) # Store dict and original metadata
                except Exception as exc:
                    filepath = metadata.get('filepath', 'unknown file')
                    logger.error(f'File {os.path.basename(filepath)} generated an exception during task execution: {exc}', exc_info=True)
                    results_with_meta.append((None, metadata)) # Append None for data on error

        processed_count = sum(1 for data_dict, meta in results_with_meta if data_dict is not None)
        logger.info(f'Parallel processing finished. Successfully processed {processed_count} out of {num_datasets} files.')
        return results_with_meta # Return list of (data_dict, metadata)

    def _validate_result(self, data_dict: Optional[dict[str, Any]], metadata: dict[str, Any]) -> bool:
        """Validates the received dictionary of arrays/metadata."""
        filename = os.path.basename(metadata.get('filepath', 'unknown'))
        if data_dict is None:
            logger.warning(f'Validation failed for {filename}: Processing returned None data.')
            return False

        # Check if all required columns are present as keys with numpy arrays
        array_keys = {k for k, v in data_dict.items() if isinstance(v, np.ndarray)}
        missing_cols = self.required_columns - array_keys
        if missing_cols:
            logger.warning(f'Validation failed for {filename}: Missing required columns {sorted(list(missing_cols))} in returned data dict keys: {sorted(list(array_keys))}.')
            return False

        # Check if arrays are non-empty (assuming all required arrays should have same length)
        first_array_key = next((k for k in self.required_columns if k in data_dict), None)
        if first_array_key is None or len(data_dict[first_array_key]) == 0:
             logger.warning(f'Validation failed for {filename}: Result arrays seem empty.')
             return False

        logger.debug(f'Validation passed for {filename}.')
        return True

    def _generate_filename(self, meta: dict[str, Any], extension: str = ".parquet") -> str:
        """Generates a sanitized output filename."""
        # Sanitize parts to be filesystem-friendly
        year = re.sub(r'\W+', '', meta.get('year', 'unk_yr'))
        make = re.sub(r'\W+', '', meta.get('make', 'unk_make'))
        model = re.sub(r'\W+', '', meta.get('model', 'unk_model'))
        trial = re.sub(r'\W+', '', meta.get('trial_number', 'unk_trial'))
        base_name = f'processed_{year}_{make}_{model}_T{trial}'
        return base_name + extension

    def _save_result(self, data_dict: dict[str, Any], metadata: dict[str, Any]):
        """Reconstructs DataFrame from dict and saves it as Parquet."""
        # Use metadata passed alongside for filename generation
        try:
            output_format = 'parquet' # Standardize on Parquet
            extension = '.parquet'
            output_filename = self._generate_filename(metadata, extension=extension)
            output_path = self.output_dir / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists

            logger.debug(f'Reconstructing DataFrame for saving: {output_filename}')
            # Filter only numpy arrays for DataFrame creation
            arrays_for_df = {k: v for k, v in data_dict.items() if isinstance(v, np.ndarray)}

            if not arrays_for_df:
                logger.error(f'No array data found in dictionary for {output_filename}. Cannot save.')
                return

            df_to_save = pl.DataFrame(arrays_for_df)

            # Ensure required columns are present and order them first
            ordered_cols = [col for col in self.required_columns if col in df_to_save.columns]
            # Include any other columns that might have been extracted
            extra_cols = [col for col in df_to_save.columns if col not in self.required_columns]
            df_to_save = df_to_save.select(ordered_cols + extra_cols) # Reorder

            logger.debug(f'Attempting to save {output_format} to: {output_path}')
            df_to_save.write_parquet(output_path, compression='zstd') # Use zstd compression
            logger.info(f'Saved {output_format}: {output_path} ({df_to_save.height} records)')

        except Exception as e:
            logger.error(f'Failed to save result for {metadata.get("filepath", "unknown file")}: {e}', exc_info=True)

    def run(self):
        """Executes the full analysis pipeline."""
        pipeline_start_time = time.time()
        logger.info('Starting Analysis Pipeline Run')
        logger.info('=============================================================')

        if not self._load_shared_resources():
            logger.critical('Failed to load shared resources. Aborting pipeline.')
            return

        datasets_to_process = self.finder.find_datasets()
        if not datasets_to_process:
            logger.warning('No datasets found to process. Exiting.')
            return

        # Run Parallel Processing
        processed_results = self._run_parallel_processing(datasets_to_process) # Returns list of (dict, meta)

        success_count, failure_count = 0, 0
        logger.info('Starting validation and saving of processed results...')
        for data_dict, metadata in tqdm(processed_results, desc='Validating & Saving', unit='result', ncols=100):
            if self._validate_result(data_dict, metadata):
                self._save_result(data_dict, metadata)
                success_count += 1
            else:
                failure_count += 1
                # Log which file failed validation
                logger.warning(f"Validation failed for file: {metadata.get('filepath', 'unknown')}. Skipping save.")

        pipeline_end_time = time.time()
        logger.info('=============================================================')
        logger.info('Analysis Pipeline Summary')
        logger.info('=============================================================')
        logger.info(f'Total datasets found: {len(datasets_to_process)}')
        logger.info(f'Successfully processed & saved: {success_count}')
        logger.info(f'Failed processing or validation: {failure_count}')
        logger.info(f'Total pipeline execution time: {pipeline_end_time - pipeline_start_time:.2f} seconds')
        logger.info('=============================================================')


if __name__ == "__main__":
    pipeline = AnalysisPipeline()
    pipeline.run()
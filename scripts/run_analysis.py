import time
import os
import re
import logging
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Optional, Tuple, Any

import polars as pl
import numpy as np
from tqdm import tqdm

try:
    from nh3.preprocessing import preprocess_ecm_file, load_route_elevation_data
    from nh3.vsp_compute import VSPCalculator
    from nh3.config import (
        DATA_DIR,
        OUTPUT_DIR,
        DRIVE_CYCLE_ROUTE_PATH,
        COLUMN_ALIASES,
        REQUIRED_COLUMNS_POST_PROCESSING,
        MAX_WORKERS,
        DEFAULT_VEHICLE_PARAMS
    )

except ImportError as e:
    logging.critical(f'Failed to import necessary modules: {e}', exc_info=True)
    import sys
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


logger = logging.getLogger(__name__)


class DatasetFinder:
    """Finds ECM datasets based on Year/Make/Model dir structure"""
    def __init__(self, base_data_path: Path):
        self.vehicle_path = Path(base_data_path) / 'vehicle_v2'
        if not self.vehicle_path.is_dir():
            logger.warning(f'Base vehicle data directory not found: {self.vehicle_path}')
            self.vehicle_path = None

    def find_datasets(self) -> List[Dict[str, Any]]:
        datasets_info = []
        if not self.vehicle_path:
            logger.error('Vehicle path invalid.')
            return datasets_info
        logger.info(f'Searching for datasets under: {self.vehicle_path}')
        file_pattern = '*/*/*/*_ECM.csv'
        potential_files = list(self.vehicle_path.rglob(file_pattern))
        logger.info(f'Found {len(potential_files)} potential ECM files using rglob pattern \'{file_pattern}\'.')
        for ecm_file_path in potential_files:
            try:
                relative_path = ecm_file_path.relative_to(self.vehicle_path)
                parts = relative_path.parts
                if len(parts) == 4:
                    year, make, model, filename = parts
                    if not (year.isdigit() and len(year) == 4):
                        logger.warning(f'Skipping non-4-digit year: {ecm_file_path}')
                        continue
                    match = re.match(r'(\d+)_ECM\.csv',
                                     filename,
                                     re.IGNORECASE)
                    trial_number = match.group(1) if match else 'unknown'
                    datasets_info.append({
                        'filepath': str(ecm_file_path.resolve()),
                        'year': year,
                        'make': make,
                        'model': model,
                        'trial_number': trial_number,
                        'relative_path': str(relative_path)}
                    )
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
        self.required_columns = REQUIRED_COLUMNS_POST_PROCESSING
        self.max_workers = MAX_WORKERS
        self.default_vsp_params = DEFAULT_VEHICLE_PARAMS
        self.finder = DatasetFinder(self.data_dir)
        self.route_tree = None
        self.route_elevs = None
        self.vsp_calculator = None

    def _load_shared_resources(self) -> bool:
        logger.info('Loading shared resources...')
        try:
            self.route_elevs, self.route_tree = load_route_elevation_data(str(self.drive_cycle_route_path))
            if self.route_tree is None or self.route_elevs is None:
                raise RuntimeError('Failed KDTree build.')
            self.vsp_calculator = VSPCalculator(**self.default_vsp_params)
            logger.info('Shared resources loaded successfully.')
            return True
        except Exception as e:
            logger.critical(f'Fatal error loading shared resources: {e}', exc_info=True)
            return False

    def _process_single_file(self, dataset_info: Dict[str, Any]) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
        """Wrapper to preprocess and calculate VSP for a single file."""
        filepath = dataset_info['filepath']
        file_basename = os.path.basename(filepath)
        processed_df = None
        vsp_col_name = 'vsp_kw_tonne'
        output_data_dict = None
        try:
            logger.debug(f'[{file_basename}] Starting _process_single_file...')
            processed_df = preprocess_ecm_file(filepath, self.aliases, self.route_tree, self.route_elevs)
            if processed_df is not None and not processed_df.is_empty():
                logger.debug(f'[{file_basename}] Preprocessing successful (Shape: {processed_df.shape}). Proceeding to VSP.')
                vsp_results = None
                if self.vsp_calculator:
                    logger.debug(f'[{file_basename}] Calling VSPCalculator.calculate_vsp...')
                    try:
                        vsp_results = self.vsp_calculator.calculate_vsp(processed_df)
                        if isinstance(vsp_results, np.ndarray):
                            logger.debug(f'[{file_basename}] VSP calculation returned NumPy array shape: {vsp_results.shape}')
                        else:
                            logger.error(f'[{file_basename}] VSP calculation did NOT return NumPy array!')
                            vsp_results = None
                    except Exception as e_vsp_calc:
                        logger.error(f'[{file_basename}] Error during VSP calculation call: {e_vsp_calc}', exc_info=True)
                        vsp_results = None
                else:
                    logger.error(f'[{file_basename}] VSP Calculator not initialized.')
                    vsp_results = None
                temp_df = processed_df
                try:
                    if isinstance(vsp_results, np.ndarray) and len(vsp_results) == temp_df.height:
                        temp_df = temp_df.with_columns(pl.Series(name=vsp_col_name, values=vsp_results))
                        logger.info(f'[{file_basename}] Temporarily added calculated VSP column.')
                    else:
                        logger.warning(f'[{file_basename}] VSP result invalid/length mismatch. Adding NULL column for \'{vsp_col_name}\'.')
                        temp_df = temp_df.with_columns(pl.lit(None, dtype=pl.Float64).alias(vsp_col_name))
                except Exception as e_add:
                    logger.error(f'[{file_basename}] Error adding VSP column temporarily: {e_add}', exc_info=True)
                    temp_df = temp_df.with_columns(pl.lit(None, dtype=pl.Float64).alias(vsp_col_name))
                output_data_dict = {}
                logger.debug(f'[{file_basename}] Extracting final columns to dict. Required: {self.required_columns}')
                for col_name in self.required_columns:
                    if col_name in temp_df.columns:
                        try:
                            output_data_dict[col_name] = temp_df[col_name].to_numpy()
                        except Exception as e_extract:
                            logger.error(f'[{file_basename}] Failed to extract column \'{col_name}\' to NumPy: {e_extract}', exc_info=True)
                    else:
                        logger.warning(f'[{file_basename}] Required column \'{col_name}\' not found in temp DataFrame. Skipping extraction for this column.')
                for meta_key in ['year', 'make', 'model', 'trial_number', 'filepath']:
                    if meta_key in dataset_info:
                        output_data_dict[meta_key] = dataset_info[meta_key]
            else:
                logger.warning(f'[{file_basename}] Preprocessing failed or returned empty DataFrame. Skipping VSP.')
                output_data_dict = None
            logger.debug(f'[{file_basename}] Finishing _process_single_file. Returning dict with keys: {list(output_data_dict.keys()) if output_data_dict else 'None'}')
            return output_data_dict, dataset_info
        except Exception as e:
            logger.error(f'Error during combined processing for {filepath}: {e}', exc_info=True)
            return None, dataset_info

    def _run_parallel_processing(self, datasets_to_process: List[Dict[str, Any]]
                                   ) -> List[Tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
        """Processes datasets in parallel, returning (Data Dict, metadata) tuples."""
        results_with_meta = []
        num_datasets = len(datasets_to_process)
        logger.info(f'Starting parallel processing for {num_datasets} datasets with {self.max_workers} workers.')
        if self.route_tree is None or self.route_elevs is None or self.aliases is None or self.vsp_calculator is None:
            logger.error('Cannot start processing: Shared resources not loaded.')
            return [(None, meta) for meta in datasets_to_process]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_meta = {executor.submit(self._process_single_file, d): d for d in datasets_to_process}
            logger.info('Waiting for processing tasks to complete...')
            for future in tqdm(concurrent.futures.as_completed(future_to_meta), total=num_datasets, desc='Processing Files', unit='file', ncols=100):
                metadata = future_to_meta[future]
                try:
                    data_dict, _ = future.result()
                    # \'{col_name}\'
                    logger.debug(f'[{os.path.basename(metadata["filepath"])}] Received result from future. Dict keys: {list(data_dict.keys()) if data_dict else None}')
                    results_with_meta.append((data_dict, metadata))
                except Exception as exc:
                    filepath = metadata.get('filepath', 'unknown file')
                    logger.error(f'File {os.path.basename(filepath)} generated an exception during task execution: {exc}', exc_info=True)
                    results_with_meta.append((None, metadata))
        processed_count = sum(1 for data_dict, meta in results_with_meta if data_dict is not None)
        logger.info(f'Parallel processing finished. Successfully processed {processed_count} out of {num_datasets} files.')
        return results_with_meta

    def _validate_result(self, data_dict: Optional[Dict[str, Any]], metadata: Dict[str, Any]) -> bool:
        """Validates the received dictionary of arrays/metadata."""
        filename = os.path.basename(metadata['filepath'])
        if data_dict is None:
            logger.warning(f'Validation failed for {filename}: Processing returned None data.')
            return False
        array_keys = {k for k, v in data_dict.items() if isinstance(v, np.ndarray)}
        missing_cols = self.required_columns - array_keys
        if missing_cols:
            logger.warning(f'Validation failed for {filename}: Missing required columns {sorted(list(missing_cols))} in returned data dict keys: {sorted(list(array_keys))}.')
            return False
        first_array_key = next((k for k in self.required_columns if k in data_dict), None)
        if first_array_key is None or len(data_dict[first_array_key]) == 0:
            logger.warning(f'Validation failed for {filename}: Result arrays seem empty.')
            return False
        logger.debug(f'Validation passed for {filename}.')
        return True

    def _generate_filename(self, meta: Dict[str, Any],
                           extension: str = '.parquet') -> str:
        year = re.sub(r'\W+', '', meta.get('year', 'unk_yr'))
        make = re.sub(r'\W+', '', meta.get('make', 'unk_make'))
        model = re.sub(r'\W+', '', meta.get('model', 'unk_model'))
        trial = re.sub(r'\W+', '', meta.get('trial_number', 'unk_trial'))
        base_name = f'processed_{year}_{make}_{model}_T{trial}'
        return base_name + extension

    def _save_result(self, data_dict: Dict[str, Any], metadata: Dict[str, Any]):
        """Reconstructs DataFrame from dict and saves it."""
        try:
            output_format = 'parquet'
            extension = '.parquet'
            output_filename = self._generate_filename(metadata, extension=extension)
            output_path = self.output_dir / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f'Reconstructing DataFrame for saving: {output_filename}')
            arrays_for_df = {k: v for k, v in data_dict.items() if isinstance(v, np.ndarray)}
            if not arrays_for_df:
                logger.error(f'No array data found in dictionary for {output_filename}. Cannot save.')
                return
            df_to_save = pl.DataFrame(arrays_for_df)
            ordered_cols = [col for col in self.required_columns if col in df_to_save.columns]
            extra_cols = [col for col in df_to_save.columns if col not in self.required_columns]
            df_to_save = df_to_save.select(ordered_cols + extra_cols)
            logger.debug(f'Attempting to save {output_format} to: {output_path}')

            # TODO CSV WRITE

            if output_format == 'parquet':
                df_to_save.write_parquet(output_path, compression='zstd')

            logger.info(f'Saved {output_format}: {output_path} ({df_to_save.height} records)')
        except Exception as e:
            logger.error(f'Failed to save result for {metadata.get("filepath", "unknown file")}: {e}', exc_info=True)

    def run(self):
        """Executes the full analysis pipeline using parallel processing."""
        pipeline_start_time = time.time()
        logger.info('Starting Analysis Pipeline Run')
        if not self._load_shared_resources():
            logger.critical('Failed to load shared resources. Aborting pipeline.')
            return
        datasets_to_process = self.finder.find_datasets()
        if not datasets_to_process:
            logger.warning('No datasets found to process. Exiting.')
            return

        processed_results = self._run_parallel_processing(datasets_to_process)
        success_count, failure_count = 0, 0
        logger.info('Starting validation and saving of processed results...')
        for data_dict, metadata in tqdm(processed_results, desc='Validating & Saving', unit='result', ncols=100):
            if self._validate_result(data_dict, metadata):
                self._save_result(data_dict, metadata)
                success_count += 1
            else:
                failure_count += 1
                logger.warning(f'Validation failed for file: {metadata.get("filepath", "unknown")}. Skipping save.')
        pipeline_end_time = time.time()
        logger.info('Analysis Pipeline Summary')
        logger.info(f'Total datasets found: {len(datasets_to_process)}')
        logger.info(f'Successfully processed & saved: {success_count}')
        logger.info(f'Failed processing or validation: {failure_count}')
        logger.info(f'Total pipeline execution time: {pipeline_end_time - pipeline_start_time:.2f} seconds')
        logger.info('=============================================================')


if __name__ == '__main__':
    pipeline = AnalysisPipeline()
    pipeline.run()

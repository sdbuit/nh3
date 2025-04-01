"""
Orchestrates the analysis pipeline using classes:
    - Finds datasets,
    - processes them in parallel,
    - validates,
    - and saves results.
"""

import concurrent.futures
import time
import os
import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

import polars as pl
from tqdm import tqdm

try:
    from nh3.preprocessing import (
        preprocess_ecm_file,
        load_route_elevation_data
    )

    from nh3.config import (
        DATA_DIR,
        OUTPUT_DIR,
        DRIVE_CYCLE_ROUTE_PATH,
        COLUMN_ALIASES,
        REQUIRED_COLUMNS_POST_PROCESSING,
        MAX_WORKERS
    )

except ImportError as e:
     logging.critical(f"Failed to import necessary modules from 'nh3' package: {e}. Ensure PYTHONPATH is set correctly or run from project root.", exc_info=True)
     DATA_DIR = Path('data')
     OUTPUT_DIR = Path('test_processed_output')
     DRIVE_CYCLE_ROUTE_PATH = Path('data/geo/drive_cycle_route.json')
     COLUMN_ALIASES = {}
     REQUIRED_COLUMNS_POST_PROCESSING = set()
     MAX_WORKERS = 1
     def preprocess_ecm_file(*args, **kwargs): return None
     def load_route_elevation_data(*args, **kwargs): return None, None
     import sys
     sys.exit(1)


# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


class DatasetFinder:
    """Finds ECM datasets based on Year/Make/Model structure under DATA_DIR/vehicle."""
    def __init__(self, base_data_path: Path):
        self.vehicle_path = Path(base_data_path) / 'vehicle_v2'
        if not self.vehicle_path.is_dir():
            logger.warning(f"Base vehicle data directory not found or is not a directory: {self.vehicle_path}")
            self.vehicle_path = None

    def find_datasets(self) -> List[Dict[str, Any]]:
        """Walks Year/Make/Model structure and globs for *_ECM.csv files."""
        datasets_info = []
        if not self.vehicle_path:
             logger.error('Vehicle path is invalid, cannot find datasets.')
             return datasets_info
        logger.info(f'Searching for datasets under: {self.vehicle_path}')
        # rglob for recursive search within the 'vehicle' directory
        file_pattern = '*/*/*/*_ECM.csv' # Year/Make/Model/Filename
        potential_files = list(self.vehicle_path.rglob(file_pattern))
        logger.info(f"Found {len(potential_files)} potential ECM files using rglob pattern '{file_pattern}'.")
        for ecm_file_path in potential_files:
            try:
                relative_path = ecm_file_path.relative_to(self.vehicle_path)
                parts = relative_path.parts
                if len(parts) == 4:
                    year, make, model, filename = parts
                    if not (year.isdigit() and len(year) == 4):
                         logger.warning(f'Skipping file, expected 4-digit year directory: {ecm_file_path}')
                         continue
                    match = re.match(r'(\d+)_ECM\.csv', filename, re.IGNORECASE)
                    trial_number = match.group(1) if match else 'unknown'
                    dataset_info = {
                        'filepath': str(ecm_file_path.resolve()),
                        'year': year, 'make': make, 'model': model,
                        'trial_number': trial_number, 'relative_path': str(relative_path)
                    }
                    datasets_info.append(dataset_info)
                else:
                     logger.warning(f'Skipping file with unexpected path structure relative to vehicle dir ({len(parts)} parts): {ecm_file_path}')
            except ValueError as e:
                 logger.error(f'Error calculating relative path for {ecm_file_path} (is it under {self.vehicle_path}?): {e}')
            except Exception as e:
                 logger.error(f'Error processing path {ecm_file_path}: {e}', exc_info=True)
        logger.info(f'Found {len(datasets_info)} valid ECM datasets.')
        return datasets_info


# AnalysisPipeline Class
class AnalysisPipeline:
    """Orchestrates the loading, processing, validation, and saving."""
    def __init__(self):
        # Get config values directly from imported config module
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        self.route_file_path = DRIVE_CYCLE_ROUTE_PATH
        self.aliases = COLUMN_ALIASES
        self.required_columns = REQUIRED_COLUMNS_POST_PROCESSING
        self.max_workers = MAX_WORKERS
        self.finder = DatasetFinder(self.data_dir)
        self.route_tree = None
        self.route_elevs = None

    def _load_shared_resources(self) -> bool:
        """Loads resources needed by all processing tasks (e.g., KDTree)."""
        logger.info('Loading shared resources (Route Elevation Data & KDTree)...')
        try:
            self.route_elevs, self.route_tree = load_route_elevation_data(str(self.route_file_path)) # Use corrected attribute
            if self.route_tree is None or self.route_elevs is None:
                raise RuntimeError('Failed to load route elevation data or build KDTree (check logs in preprocessing).')
            logger.info('Shared resources loaded successfully.')
            return True
        except Exception as e:
            logger.critical(f"Fatal error loading shared resources using path '{self.route_file_path}': {e}", exc_info=True)
            return False

    def _run_parallel_processing(self, datasets_to_process: List[Dict[str, Any]]) -> List[Tuple[Optional[pl.DataFrame], Dict[str, Any]]]:
        """Processes datasets in parallel, returning (DataFrame, metadata) tuples."""
        results_with_meta = []
        num_datasets = len(datasets_to_process)
        logger.info(f'Starting parallel processing for {num_datasets} datasets with {self.max_workers} workers.')
        if self.route_tree is None or self.route_elevs is None or self.aliases is None:
             logger.error('Cannot start processing: Shared resources not loaded.')
             return [(None, meta) for meta in datasets_to_process]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_meta = {
                executor.submit(preprocess_ecm_file, d['filepath'], self.aliases, self.route_tree, self.route_elevs): d
                for d in datasets_to_process}
            logger.info('Waiting for preprocessing tasks to complete...')
            for future in tqdm(concurrent.futures.as_completed(future_to_meta),
                               total=num_datasets, desc='Preprocessing Files', unit='file', ncols=100):
                metadata = future_to_meta[future]
                filepath = metadata['filepath']
                try:
                    processed_df = future.result()
                    results_with_meta.append((processed_df, metadata))
                except Exception as exc:
                    logger.error(f"File '{os.path.basename(filepath)}' generated an exception during processing task: {exc}", exc_info=True)
                    results_with_meta.append((None, metadata))
        processed_count = sum(1 for df, meta in results_with_meta if df is not None)
        logger.info(f'Parallel processing finished. Successfully processed {processed_count} out of {num_datasets} files.')
        return results_with_meta

    def _validate_result(self, df: Optional[pl.DataFrame], metadata: Dict[str, Any]) -> bool:
        """Validates the processed DataFrame based on config requirements."""
        filename = os.path.basename(metadata['filepath'])
        if df is None:
            logger.warning(f'Validation failed for {filename}: Processing returned None.')
            return False
        if df.height == 0:
            logger.warning(f'Validation failed for {filename}: DataFrame is empty.')
            return False
        missing_cols = self.required_columns - set(df.columns)
        if missing_cols:
            logger.warning(f'Validation failed for {filename}: Missing required columns {sorted(list(missing_cols))}.')
            return False
        logger.debug(f'Validation passed for {filename}.')
        return True

    def _generate_filename(self, meta: Dict[str, Any], extension: str = '.parquet') -> str:
        """Generates an output filename based on metadata (Year_Make_Model_Trial)."""
        year = re.sub(r'\W+', '', meta.get('year', 'unk_yr'))
        make = re.sub(r'\W+', '', meta.get('make', 'unk_make'))
        model = re.sub(r'\W+', '', meta.get('model', 'unk_model'))
        trial = re.sub(r'\W+', '', meta.get('trial_number', 'unk_trial'))
        base_name = f'processed_{year}_{make}_{model}_T{trial}'
        return base_name + extension

    def _save_result(self, df: pl.DataFrame, metadata: Dict[str, Any]):
        """Saves the processed DataFrame to the output directory."""
        try:
            output_format = 'parquet'
            extension = '.parquet'
            output_filename = self._generate_filename(metadata, extension=extension)
            output_path = self.output_dir / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f'Attempting to save {output_format} to: {output_path}')
            if output_format == 'parquet':
                df.write_parquet(output_path, compression='zstd')
            # else: df.write_csv(output_path)
            logger.info(f'Saved {output_format}: {output_path} ({df.height} records)')
        except Exception as e:
            logger.error(f'Failed to save result for {metadata.get('filepath', 'unknown file')}: {e}', exc_info=True)

    def run(self):
        """Executes the full analysis pipeline."""
        pipeline_start_time = time.time()
        logger.info('================ Starting Analysis Pipeline Run ================')
        if not self._load_shared_resources():
            logger.critical('Failed to load shared resources. Aborting pipeline.')
            return
        datasets_to_process = self.finder.find_datasets()
        if not datasets_to_process:
            logger.warning('No datasets found to process. Exiting.')
            return
        preprocessing_results = self._run_parallel_processing(datasets_to_process)
        success_count, failure_count = 0, 0
        logger.info('Starting validation and saving of processed results...')
        for df, metadata in tqdm(preprocessing_results, desc='Validating & Saving', unit='result', ncols=100):
            if self._validate_result(df, metadata):
                self._save_result(df, metadata)
                success_count += 1
            else:
                failure_count += 1
                # logger.warning(f'Skipping save for {metadata.get('filepath', 'unknown file')} due to failure.') # Already logged
        pipeline_end_time = time.time()
        logger.info('================= Analysis Pipeline Summary =================')
        logger.info(f'Total datasets found: {len(datasets_to_process)}')
        logger.info(f'Successfully processed & saved: {success_count}')
        logger.info(f'Failed processing or validation: {failure_count}')
        logger.info(f'Total pipeline execution time: {pipeline_end_time - pipeline_start_time:.2f} seconds')
        logger.info('=============================================================')


if __name__ == '__main__':
    pipeline = AnalysisPipeline()
    pipeline.run()

import csv
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional

import numpy as np
import polars as pl

from nh3 import config as project_config
from nh3.preprocessing import preprocess_data
from nh3.vsp_compute import VSPCalculator
from optimized_processor import OptimizedProcessor


logger = logging.getLogger(__name__)


class DataProcessor:
    """Data Processing Pipeline"""
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        self.config = project_config.config  # Use the singleton Config instance by default
        if config_override:
            for key, value in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.info(f'Overriding config: {key} = {value}')
                else:
                    logger.warning(f'Attempted to override non-existent config key: {key}')
        self.vsp_calculator = VSPCalculator(
            vehicle_mass=getattr(self.config, 'VEHICLE_MASS', 1500.0),
            rolling_resistance_coef=getattr(
                self.config, 'ROLLING_RESISTANCE_COEF', 0.0135),
            drag_coef=getattr(self.config, 'DRAG_COEF', 0.369),
            frontal_area=getattr(self.config, 'FRONTAL_AREA', 2.27),
            drivetrain_efficiency=getattr(
                self.config, 'DRIVETRAIN_EFFICIENCY', 0.92),
            air_density=getattr(self.config, 'AIR_DENSITY', 1.207))
        self.optimized_processor = OptimizedProcessor(self.config)

    def load_data(self, file_path: Union[str, Path]) -> pl.DataFrame:
        file_path = Path(file_path)
        logger.info(f'Attempting to load data and auto-detect header from {file_path}')
        # Auto-detect header using values from config.COLUMN_ALIASES
        # flatten the list of lists for easier checking:
        # all_aliases = set(item for sublist in self.config.COLUMN_ALIASES.values() for item in sublist)
        essential_headers = {
            'Latitude', 'Latitude(degrees)', 'Latitude(deg)',
            'Longitude', 'Longitude(degrees)', 'Longitude(deg)',
            'Altitude_m(m)', 'Altitude(m)',
            'Speed(kmh)', 'Speed_kmh(km/h)',
            'Elapse Time (sec)'}
        rows_to_skip = 0
        header_found = False
        max_lines_to_check = 30
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i in range(max_lines_to_check):
                    try:
                        line = f.readline()
                    except EOFError:
                        break
                    if not line:
                        break
                    reader = csv.reader([line])
                    try:
                        potential_headers = [h.strip() for h in next(reader)]
                    except StopIteration:
                        continue # Skip empty lines
                    # Heuristic: Check if at least 3 essential col names in
                    # headers are present.
                    found_count = 0
                    for header in potential_headers:
                        if header in essential_headers:
                            found_count += 1
                    if found_count >= 3:
                        rows_to_skip = i
                        header_found = True
                        logger.info(f'Auto-detected header on line {i + 1} (skipping {rows_to_skip} rows).')
                        break
        except FileNotFoundError:
            logger.error(f'File not found: {file_path}')
            raise
        except Exception as e:
            logger.error(f'Error reading file header {file_path}: {e}')
            raise
        if not header_found:
            logger.warning(f'Could not auto-detect header row within the first {max_lines_to_check} lines. '
                           f'Attempting to load with skip_rows=0. Available columns might be incorrect.')
            rows_to_skip = 0
        try:
            df = pl.read_csv(
                file_path,
                skip_rows=rows_to_skip,
                has_header=True,
                infer_schema_length=1000,
                ignore_errors=True)
            logger.info(f'Loaded {len(df)} records from {file_path} after skipping {rows_to_skip} rows.')
            if df.is_empty():
                 logger.warning(f'Loaded DataFrame is empty from {file_path}')
            elif len(df.columns) <= 1:
                 logger.warning(f'Loaded DataFrame from {file_path} has only {len(df.columns)} columns, check delimiter or file format.')
            return df
        except Exception as e:
            logger.error(f'Error loading data from {file_path} even after header detection: {e}')
            raise

    def process_file(
        self, file_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        df = self.load_data(file_path)
        logger.info(f'Initial columns: {df.columns}')
        # Pass the DataFrame and the Config instance to preprocess_data
        # preprocess_data.py now handles the column renaming internally
        df, grades, metadata = preprocess_data(df, self.config)
        logger.info(f'Columns after preprocessing and cleaning: {df.columns}')
        required_cols_for_vsp = [speed_col:='speed_kmh',
                                 time_col:='elapse_sec',
                                 grade_col:='grade']
        missing_cols = [col for col in required_cols_for_vsp if col not in df.columns]
        if missing_cols:
            logger.error(f'Missing expected columns after preprocessing for VSP calculation: {missing_cols}. Available: {df.columns}')
            raise KeyError(f'Missing required columns after preprocessing for VSP calculation: {missing_cols}')
        speeds = df[speed_col].to_numpy()
        timestamps = df[time_col].to_numpy()
        grades_for_vsp = df[grade_col].to_numpy()
        vsp_values, accelerations = self.optimized_processor.calculate_vsp(
            speeds_kph=speeds, timestamps=timestamps, grades=grades_for_vsp)
        df = df.with_columns([
            pl.Series(name='vsp', values=vsp_values),
            pl.Series(name='acceleration', values=accelerations)])
        vsp_modes = self.optimized_processor.classify_vsp_modes(vsp_values)
        df = df.with_columns(pl.Series(name='vsp_mode', values=vsp_modes))
        metadata.update({'vsp_min': np.min(vsp_values),
                         'vsp_max': np.max(vsp_values),
                         'vsp_mean': np.mean(vsp_values),
                         'vsp_std': np.std(vsp_values),
                         'acceleration_min': np.min(accelerations),
                         'acceleration_max': np.max(accelerations),
                         'acceleration_mean': np.mean(accelerations),
                         'acceleration_std': np.std(accelerations),})
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            input_filename = Path(file_path).stem
            output_path = output_dir / f'{input_filename}_processed.csv'
            df.write_csv(output_path)
            logger.info(f'Saved processed data to {output_path}')
            metadata['output_path'] = str(output_path)
        return df, metadata

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = '*.csv'
    ) -> Dict[str, Dict[str, Any]]:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_files = list(input_dir.glob(file_pattern))
        logger.info(f'Found {len(input_files)} files matching pattern {file_pattern} in {input_dir}')
        results = {}
        for file_path in input_files:
            try:
                _, metadata = self.process_file(file_path, output_dir)
                results[file_path.name] = metadata
                logger.info(f'Successfully processed {file_path.name}')
            except Exception as e:
                logger.error(f'Error processing {file_path.name}: {e}')
                results[file_path.name] = {'error': str(e)}
        return results

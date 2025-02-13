# main.py V2.0 (updated) 
import sys
from typing import List, Dict, Tuple

import polars as pl

from config import config
from compute_vsp import ComputeVSP
from dataset import DatasetLoader


class DataProcessor:
    def __init__(self):
        self._vsp = ComputeVSP()

    def process_dataset(self, df: pl.DataFrame, meta: Dict) -> pl.DataFrame:
        """Process a single dataset."""
        data = df.to_dicts()        
        vsp_values = self._vsp.compute_vsp(data)
        df = df.with_columns([
            pl.Series(name="vsp", values=vsp_values),
            pl.col("elapse_sec").diff().alias("delta_t"),
            pl.col("alt_m").diff().alias("delta_alt")])
        return self._clean_dataframe(df)

    def _clean_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Remove unneeded columns that have empty names, all nulls, or all zeroes
        (for numeric data).
        """
        cols_to_drop = []
        for col in df.columns:
            if not col.strip():
                cols_to_drop.append(col)
                continue
            s = df[col]
            if s.null_count() == len(s):
                cols_to_drop.append(col)
                continue
            if s.dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                non_null = s.drop_nulls()
                if not non_null.is_empty() and (non_null == 0).all():
                    cols_to_drop.append(col)
        if cols_to_drop:
            df = df.drop(cols_to_drop)
        return df


class PipelineController:
    def __init__(self):
        self.loader = DatasetLoader(config.DATA_DIR / 'vehicle_v2')
        self.processor = DataProcessor()

    def run(self):
        datasets = self._load_datasets()
        for df, meta in datasets:
            if self._validate_dataset(df):
                processed_df = self.processor.process_dataset(df, meta)
                self._save_results(processed_df, meta)
                
    def _load_datasets(self) -> List[Tuple[pl.DataFrame, Dict]]:
        """Load and filter ECM datasets"""
        all_data = self.loader.load_all()
        return [(df, meta) for df, meta in all_data
                if meta.get('dataset_type') == 'ECM']
        
    def _validate_dataset(self, df: pl.DataFrame) -> bool:
        """Validate required columns for geographical drive cycle profile."""
        required = {'lat_deg', 'lon_deg', 'alt_m', 'elapse_sec'}
        return all(col in df.columns for col in required)
        
    def _save_results(self, df: pl.DataFrame, meta: Dict):
        """Save processed results with metadata"""
        filename = self._generate_filename(meta)
        df.write_csv(config.OUTPUT_DIR / filename)
        print(f'Processed {filename} with {len(df)} records')

    def _generate_filename(self, meta: Dict) -> str:
        """
        Generate output filename from metadata with trial number and proper 
        prefix.
        """
        base_parts = [
            meta.get('year', 'unknown'),
            meta.get('make', 'unknown'),
            meta.get('model', 'unknown')]
        trial = meta.get('trial_number', '')
        if trial:
            base_parts.append(trial)
        return f'processed_{'_'.join(base_parts)}.csv'


if __name__ == '__main__':
    controller = PipelineController()
    controller.run()
    sys.exit(0)

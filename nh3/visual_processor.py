"""Extends the DataProcessor class to include visualization capabilities."""

import os
import logging
from typing import List, Dict, Tuple
from pathlib import Path

import polars as pl

from .data_processor import DataProcessor
from .visualization import EmissionsVisualizer

logger = logging.getLogger(__name__)


class VisualDataProcessor(DataProcessor):
    """Extended DataProcessor with data visualization capabilities."""
    def __init__(self, visualization_dir: str = None):
        super().__init__()
        self.visualization_dir = visualization_dir
        self.visualizer = EmissionsVisualizer(output_dir=visualization_dir)

    def process_and_visualize_file(
        self,
        file_path: str,
        output_dir: str = None,
        visualization_dir: str = None,
        prefix: str = None,
    ) -> Tuple[pl.DataFrame, Dict, List[str]]:
        df, meta = self.process_file(file_path)
        if output_dir and not df.is_empty():
            os.makedirs(output_dir, exist_ok=True)
            filename = Path(file_path).stem + "_processed.csv"
            output_path = os.path.join(output_dir, filename)
            df.write_csv(output_path)
            logger.info(f"Saved processed file to {output_path}")
        vis_dir = visualization_dir or self.visualization_dir
        if vis_dir and not df.is_empty():
            os.makedirs(vis_dir, exist_ok=True)
            if prefix is None:
                parts = [
                    meta.get("year", "unknown"),
                    meta.get("make", "unknown"),
                    meta.get("model", "unknown"),
                ]
                trial = meta.get("trial_number", "")
                if trial:
                    parts.append(trial)
                prefix = "_".join(parts)
            vis_paths = self.visualizer.create_visualization_report(df, output_dir=vis_dir, prefix=prefix)
            logger.info(f"Generated {len(vis_paths)} visualizations in {vis_dir}")
            return df, meta, vis_paths
        return df, meta, []

    def process_and_visualize_directory(
        self, directory: str, output_dir: str = None, visualization_dir: str = None
    ) -> List[Tuple[pl.DataFrame, Dict, List[str]]]:
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv"):
                    file_paths.append(os.path.join(root, file))
        results = []
        for file_path in file_paths:
            logger.info(f"Processing and visualizing file: {file_path}")
            result = self.process_and_visualize_file(file_path, output_dir, visualization_dir)
            results.append(result)
        return results

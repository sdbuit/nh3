from pathlib import Path
import logging
import sys
import re

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from nh3.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from nh3.config import OUTPUT_DIR, REPORTS_DIR, PLOTS_DIR
except ImportError:
    logging.error('Failed to import paths from nh3.config. Using default relative paths.')
    REPORTS_DIR = PROJECT_ROOT / 'reports'
    PLOTS_DIR = REPORTS_DIR / 'plots'
    OUTPUT_DIR = PROJECT_ROOT / 'test_processed_output'
parquet_file_path_str = r'../test_processed_output/processed_2017_Dodge_RAM2500_T01.parquet'
parquet_file_path = Path(parquet_file_path_str)

# Optional: Clip grade for better visualization of typical range
# Set to None to disable clipping
GRADE_CLIP_PERCENT = 20 # e.g., clip values outside +/- 20%

def parse_metadata_from_filename(filename: str) -> dict:
    """Parses Year, Make, Model, Trial from processed filename."""
    pattern = r'processed_(\d{4})_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_T(\d+)\.parquet'
    match = re.match(pattern, filename, re.IGNORECASE)
    if match:
        return {
            'year': match.group(1),
            'make': match.group(2),
            'model': match.group(3),
            'trial': match.group(4)
        }
    else:
        logger.warning(f'Could not parse metadata from filename: {filename}')
        return {'year': 'unknown', 'make': 'unknown', 'model': 'unknown', 'trial': 'unknown'}

def generate_plot_filename(metadata: dict, plot_type: str) -> str:
    """Generates the output plot filename."""
    # Use f-string for cleaner formatting
    return f"{metadata['year']}_{metadata['make']}_{metadata['model']}_Trial_{metadata['trial']}_{plot_type}.png"

def plot_and_save_elevation_grade(df: pl.DataFrame, input_filename: str, output_plot_path: Path):
    """Creates and saves a plot with elevation and grade vs. time."""
    # Columns to plot
    time_col = 'time_s'
    alt_col = 'altitude_m_mapped'
    grade_col = 'grade_percent'
    if df.is_empty():
        logger.warning(f'DataFrame for {input_filename} is empty. Skipping plot.')
        return
    required_cols = {time_col, alt_col, grade_col}
    if not required_cols.issubset(df.columns):
        logger.error(f'Missing required columns in {input_filename}: {required_cols - set(df.columns)}. Cannot plot.')
        return
    try:
        df_sorted = df.sort(time_col)
        df_plot = df_sorted.select([time_col, alt_col, grade_col]).drop_nulls()
        if df_plot.is_empty():
             logger.warning(f'No valid (non-null) data points found for plotting in {input_filename}.')
             return
        time_data = df_plot[time_col].to_numpy()
        alt_data = df_plot[alt_col].to_numpy()
        grade_data = df_plot[grade_col].to_numpy()
    except Exception as e:
        logger.error(f'Error preparing data using Polars for {input_filename}: {e}', exc_info=True)
        return
    logger.info(f'Plotting data for {input_filename} ({len(df_plot)} points)...')
    if GRADE_CLIP_PERCENT is not None:
        original_min, original_max = np.min(grade_data), np.max(grade_data)
        grade_data = np.clip(grade_data, a_min=-GRADE_CLIP_PERCENT, a_max=GRADE_CLIP_PERCENT)
        clipped_min, clipped_max = np.min(grade_data), np.max(grade_data)
        if original_min < clipped_min or original_max > clipped_max:
             logger.info(f'Grade data clipped to [{-GRADE_CLIP_PERCENT}%, {GRADE_CLIP_PERCENT}%] for visualization.')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Elevation and Grade Profile\n{input_filename}', fontsize=14)
    try:
        ax1.plot(time_data, alt_data, label='Mapped Altitude', color='blue', linewidth=1)
        ax1.set_ylabel('Altitude (m)')
        ax1.set_title('Mapped Elevation Profile')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax1.ticklabel_format(style='plain', axis='y', useOffset=False)

        ax2.plot(time_data, grade_data, label='Calculated Grade', color='red', linewidth=1)
        ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, label='0% Grade')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Grade (%)')
        ax2.set_title('Calculated Road Grade')
        if GRADE_CLIP_PERCENT is not None:
             ax2.set_ylim(-GRADE_CLIP_PERCENT - 2, GRADE_CLIP_PERCENT + 2)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'Saving plot to: {output_plot_path}')
        fig.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    except Exception as e:
         logger.error(f'Error during plotting or saving for {input_filename}: {e}', exc_info=True)
    finally:
        plt.close(fig)


if __name__ == '__main__':
    logger.info(f'Attempting to load Parquet file: {parquet_file_path}')
    if not parquet_file_path.is_file():
        logger.error(f'Input file not found: {parquet_file_path}')
        sys.exit(1)
    try:
        data_df = pl.read_parquet(parquet_file_path)
        logger.info(f'Successfully loaded data. Shape: {data_df.shape}')
        logger.info(f'Columns: {data_df.columns}')
        input_filename = parquet_file_path.name
        metadata = parse_metadata_from_filename(input_filename)
        plot_filename = generate_plot_filename(metadata,
                                               plot_type='grade_profile')
        # Construct output path using PLOTS_DIR imported from config
        output_path = PLOTS_DIR / plot_filename
        # Generate and save the plot
        plot_and_save_elevation_grade(data_df, input_filename, output_path)
        logger.info('Plotting complete.')
    except pl.exceptions.PolarsError as e:
         logger.error(f'Polars error reading {parquet_file_path}: {e}',
                      exc_info=True)
    except FileNotFoundError:
         logger.error(f'File not found error for {parquet_file_path}')
    except Exception as e:
        logger.error(f'An unexpected error occurred in main block: {e}',
                     exc_info=True)

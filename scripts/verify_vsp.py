"""
Verifies grade/elevation, and compares VSP among different runs.
"""

import os
# from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

test_processed_output = '../test_processed_output/'
csv_files = [
    'processed_2016_Chevrolet_Colorado_T01.csv',
    'processed_2016_Chevrolet_Colorado_T02.csv',
    'processed_2016_Chevrolet_Colorado_T03.csv',
    'processed_2016_Jeep_Wrangler_T01.csv',
    'processed_2016_Jeep_Wrangler_T02.csv',
    'processed_2016_Jeep_Wrangler_T03.csv',
    'processed_2017_Dodge_RAM2500_T01.csv',
    'processed_2017_Dodge_RAM2500_T02.csv',
    'processed_2017_Dodge_RAM2500_T03.csv']

def load_dataframes(base_path: str,
                    filenames: list[str]) -> dict[str, pl.DataFrame]:
    data_dict = {}
    for fname in filenames:
        path = os.path.join(base_path, fname)
        try:
            df = pl.read_csv(path)
            data_dict[fname] = df
        except Exception as e:
            print(f'Could not load {fname}: {e}')
    return data_dict

# def summarize_grade_and_elevation(df: pl.DataFrame) -> None:
#     """
#     Prints summary statistics for elevation and grade_percent columns.
#     """
#     stats = df.select([
#         pl.col('elevation_m').mean().alias('mean_elevation_m'),
#         pl.col('elevation_m').min().alias('min_elevation_m'),
#         pl.col('elevation_m').max().alias('max_elevation_m'),
#         pl.col('grade_percent').mean().alias('mean_grade_percent'),
#         pl.col('grade_percent').min().alias('min_grade_percent'),
#         pl.col('grade_percent').max().alias('max_grade_percent'),])
#     print(stats)

def compare_vsp_stats(dfs: dict[str, pl.DataFrame]) -> None:
    """Compares average or max VSP across different files."""
    results = []
    for fname, df in dfs.items():
        if 'vsp_kw_tonne' in df.columns:
            mean_vsp = df['vsp_kw_tonne'].mean()
            max_vsp = df['vsp_kw_tonne'].max()
            results.append((fname, mean_vsp, max_vsp))
        else:
            results.append((fname, None, None))
    print('VSP Summary')
    for (fname, mean_vsp, max_vsp) in results:
        print(f'{fname} -> Mean VSP: {mean_vsp}, Max VSP: {max_vsp}')

def plot_elevation_vs_time(df: pl.DataFrame, title_prefix: str) -> None:
    """elevation [m] vs. time"""
    if 'time_s' not in df.columns or 'elevation_m' not in df.columns:
        print(f'{title_prefix} -> Missing time_s or elevation_m columns.')
        return
    plt.figure()
    plt.plot(df['time_s'].to_numpy(), df['elevation_m'].to_numpy())
    plt.title(f'{title_prefix} Elevation vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Elevation (m)')
    plt.show()

def plot_grade_vs_time(df: pl.DataFrame, title_prefix: str) -> None:
    """grade [%] vs. time"""
    if 'time_s' not in df.columns or 'grade_percent' not in df.columns:
        print(f'{title_prefix} -> Missing time_s or grade_percent columns.')
        return
    plt.figure()
    plt.plot(df['time_s'].to_numpy(), df['grade_percent'].to_numpy())
    plt.title(f'{title_prefix} Grade vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Grade (%)')
    plt.show()

def plot_vsp_vs_time(df: pl.DataFrame, title_prefix: str) -> None:
    """
    VSP [kw/tonne] vs. time.
    """
    if 'time_s' not in df.columns or 'vsp_kw_tonne' not in df.columns:
        print(f'{title_prefix} -> Missing time_s or vsp_kw_tonne columns.')
        return

    plt.figure()
    plt.plot(df['time_s'].to_numpy(), df['vsp_kw_tonne'].to_numpy())
    plt.title(f'{title_prefix} VSP vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('VSP (kW/tonne)')
    plt.show()

def main():
    dfs = load_dataframes(test_processed_output, csv_files)
    for fname, df in dfs.items():
        if df is None or df.is_empty():
            print(f'Skipping empty or failed DataFrame for {fname}')
            continue
        print(f'Analysis for {fname}')
        # TODO fix summarize_grade_and_elevation()
        # summarize_grade_and_elevation(df)
        plot_elevation_vs_time(df, title_prefix=fname)
        plot_grade_vs_time(df, title_prefix=fname)
        plot_vsp_vs_time(df, title_prefix=fname)
    compare_vsp_stats(dfs)

if __name__ == '__main__':
    # dfs = load_dataframes(test_processed_output, csv_files)
    import sys
    main()
    sys.exit(1)


import os
import sys

import polars as pl

import dataset as ds
import geospatial as geo
from dataset import DatasetLoader


def comp_vsp(df: pl.DataFrame) -> pl.DataFrame:
    """TODO"""
    required_cols = {'elapse_sec', 'gps_speed_m_s'}
    if not required_cols.issubset(df.columns):
        return df
    
    df = df.with_columns([
        (pl.col('elapse_sec').shift(-1) - pl.col('elapse_sec'))
        .fill_null(0)
        .alias('dt_s'),  # Delta time
        (pl.col('gps_speed_m_s').shift(-1) - pl.col('gps_speed_m_s'))
        .fill_null(0)
        .alias('dv_m_s')])  # e
    
    df = df.with_columns([
        (pl.col('dv_m_s') / pl.col('dt_s'))
        .fill_nan(None)
        .alias('accel_m_s2')])
    
    return df

def main():
    base_path = './data/vehicle_v2'
    geo_elev_model_path = './data/geo/drive_cycle_route.json'
    output_folder = './data/test_process_output'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    loader = DatasetLoader(base_path=base_path)
    results = loader.load_all_datasets_parallel()
    ecms = [(df, meta) for df, meta in results if meta.get('dataset_type') == 'ECM']
    
    groups = {}
    for df, meta in ecms:
        key = (meta.get('year'), meta.get('make'), meta.get('model'))
        groups.setdefault(key, []).append((df, meta))

    for key, items in groups.items():
        year, make, model = key
        items.sort(key=lambda x: x[1].get('file'))
        
        for i, (df, meta) in enumerate(items, start=1):
            file_index = f'{i:02d}'
            print(f'Processing ECM file: {meta.get("file")}')
            
            if 'speed_mph' in df.columns:
                df = df.with_columns((pl.col('speed_mph') * 0.44704).alias('gps_speed_m_s'))
            df = ds.process_nox_cols(df, shift_count=5)

            if {'lat_deg', 'lon_deg', 'alt_m'}.issubset(df.columns):
                out_path_alt = os.path.join(output_folder, f'mapped_{year}_{make}_{model}_ECM_{file_index}.csv')
                try:
                    df = geo.run_altitude_mapping(
                        vehicle_df=df,
                        lat_col='lat_deg',
                        lon_col='lon_deg',
                        alt_col='alt_m',
                        geo_elev_model_path=geo_elev_model_path,
                        use_process_map=True,
                        output_path=out_path_alt,
                    )
                    df = geo.process_for_grade(df)
                except geo.DataValidationError as e:
                    print(f'[Error] Grade calculation failed: {e}')
            else:
                print(f'[Warning] Missing geodetic columns in {meta.get("file")}')

            df = comp_vsp(df)
            final_path = os.path.join(output_folder, f'final_{year}_{make}_{model}_ECM_{file_index}.csv')
            df.write_csv(final_path, null_value='')
            print(f'[{file_index}] Completed pipeline, wrote {final_path}')


if __name__ == '__main__':
    main()
    sys.exit(0)

import sys
import polars as pl
import math

from dataset import (DatasetLoader, vehicle_geodetic_to_coord, update_dataset)
import geospatial as geo


def process_nox_nh3_columns(df: pl.DataFrame, shift_count: int = 5) -> pl.DataFrame:
    rename_map = {}
    if 'nox_cant' in df.columns and 'nox_2' in df.columns:
        rename_map = {'nox_cant': 'NOX_NODE10', 'nox_2': 'NOX_NODE12'}
    if rename_map:
        df = df.rename(rename_map)
    if 'NOX_NODE10' not in df.columns or 'NOX_NODE12' not in df.columns:
        return df
    df = df.with_column(pl.col('NOX_NODE10').shift(shift_count).alias('NOX_NODE10_shifted'))
    df = df.with_column((pl.col('NOX_NODE12') - pl.col('NOX_NODE10_shifted'))
        .clip_min(0.0)
        .alias('nox_ppm'))
    def calibrate_nh3(x: float) -> float:
        if x is None or math.isnan(x):
            return None
        return 1.0553 * x + 0.3223
    df = df.with_column(pl.col('NOX_NODE10_shifted').apply(calibrate_nh3).alias('nh3_ppm'))
    return df

def process_for_vsp(df: pl.DataFrame) -> pl.DataFrame:
    if 'elapse_sec' not in df.columns or 'gps_speed_m_s' not in df.columns:
        return df
    df = df.with_columns([
        (pl.col('elapse_sec').shift(-1) - pl.col('elapse_sec')).alias('dt_s'),
        (pl.col('gps_speed_m_s').shift(-1) - pl.col('gps_speed_m_s')).alias('dv_m_s'),])
    df = df.with_column(pl.when((pl.col('dt_s') == 0) | pl.col('dt_s').is_null())
        .then(None)
        .otherwise(pl.col('dv_m_s') / pl.col('dt_s'))
        .alias('accel_m_s2'))
    return df

def run_altitude_mapping(vehicle_df: pl.DataFrame, lat_col: str, lon_col: str, 
    alt_col: str, geo_elev_model_path: str, use_process_map: bool, 
    output_path: str) -> pl.DataFrame:
    required = {lat_col, lon_col, alt_col}
    missing = required - set(vehicle_df.columns)
    if missing:
        print(f'[Warning] Missing required columns for altitude mapping: {missing}')
        print('Returning the original DataFrame unchanged.')
        return vehicle_df
    geo_elev_model_coord = geo.load_geo_elev_model_v2(geo_elev_model_path)
    coords = vehicle_geodetic_to_coord(vehicle_df)
    mapper = geo.CoordinateMapper(
        altitude_data=geo_elev_model_coord,
        distance_calculator=geo.HaversineDistance(),
        use_kdtree=True)
    mapped_elevs = mapper.map_elevations_v2(coords, use_process_map=use_process_map)
    updated_coords = geo.merge_altitudes(coords, mapped_elevs)
    updated_vehicle_df = update_dataset(vehicle_df, updated_coords, lat_col, lon_col, alt_col)
    updated_vehicle_df = updated_vehicle_df.fill_nan(None)
    updated_vehicle_df.write_csv(output_path, null_value='')
    return updated_vehicle_df


def main():
    base_path = './data/vehicle_v2'
    geo_elev_model_path = './data/geo/drive_cycle_route.json'
    loader = DatasetLoader(base_path=base_path)
    results = loader.load_all_datasets_parallel()
    year_filter = '2007'
    make_filter = 'Dodge'
    model_filter = 'RAM1500'
    filtered_results = []
    for df, meta in results:
        if (meta.get('year') == year_filter and
            meta.get('make') == make_filter and
            meta.get('model') == model_filter):
            filtered_results.append((df, meta))
    if not filtered_results:
        print(f'No matching datasets found for {year_filter} {make_filter} {model_filter}.')
        sys.exit(0)

    print(f'Found {len(filtered_results)} datasets for {year_filter}/{make_filter}/{model_filter}.')

    for i, (df, meta) in enumerate(filtered_results, start=1):
        df = process_nox_nh3_columns(df, shift_count=5)
        if 'speed_mph' in df.columns:
            df = df.with_columns((pl.col('speed_mph') * 0.44704).alias('gps_speed_m_s'))
        out_path_alt = f'mapped_{meta['year']}_{meta['make']}_{meta['model']}_{i}.csv'
        updated_df = run_altitude_mapping(
            vehicle_df=df,
            lat_col='lat_deg',
            lon_col='lon_deg',
            alt_col='alt_m',
            geo_elev_model_path=geo_elev_model_path,
            use_process_map=True,
            output_path=out_path_alt)
        final_df = process_for_vsp(updated_df)
        final_path = f'final_{meta['year']}_{meta['make']}_{meta['model']}_{i}.csv'
        final_df.write_csv(final_path, null_value='')
        print(f'[{i}] Completed pipeline, wrote {out_path_alt} and {final_path}')


if __name__ == '__main__':
    main()
    sys.exit(0)

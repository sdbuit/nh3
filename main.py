import os
import sys
import math
import re
import polars as pl

from dataset import (DatasetLoader, vehicle_geodetic_to_coord, update_dataset, 
                    fix_duplicate_columns)
import geospatial as geo


def process_nox_cols(df: pl.DataFrame, shift_count: int = 5) -> pl.DataFrame:
    nox_count = df.columns.count('NOX(ppm)')    
    if nox_count >= 2:
        new_cols = []
        count = 0
        for col in df.columns:
            if col == 'NOX(ppm)':
                count += 1
                new_cols.append(f'nox_node_{count:02d}')
            else:
                new_cols.append(col)
        df.columns = new_cols
    elif nox_count == 1:
        new_cols = []
        for col in df.columns:
            if col == 'NOX(ppm)':
                new_cols.append('nox_node_01')
            else:
                new_cols.append(col)
        df.columns = new_cols
        print("[Warning] Only one 'NOX(ppm)' column found. 'nox_node_02' is missing; 'nox_ppm' column will be set to null.")
    else:
        print("[Warning] No 'NOX(ppm)' column found in the DataFrame.")
        return df

    def calibrate_nh3(x: float) -> float:
        if x is None or math.isnan(x):
            return None
        return 1.0553 * x + 0.3223

    df = df.with_columns([pl.col('nox_node_01')
                          .shift(shift_count)
                          .alias('nox_node_01_shifted')])
    if 'nox_node_02' in df.columns:
        diff_expr = pl.col('nox_node_02') - pl.col('nox_node_01_shifted')
        df = df.with_columns([
            pl.when(diff_expr < 0)
              .then(0.0)
              .otherwise(diff_expr)
              .alias('nox_ppm')
        ])
    else:
        df = df.with_columns([pl.lit(None).alias('nox_ppm')])
    df = df.with_columns([pl.col('nox_node_01_shifted')
                          .map(calibrate_nh3)
                          .alias('nh3_ppm')])

    return df

def process_for_vsp(df: pl.DataFrame) -> pl.DataFrame:
    if 'elapse_sec' not in df.columns or 'gps_speed_m_s' not in df.columns:
        return df
    df = df.with_columns([
        (pl.col('elapse_sec').shift(-1) - pl.col('elapse_sec')).alias('dt_s'),
        (pl.col('gps_speed_m_s').shift(-1) - pl.col('gps_speed_m_s')).alias('dv_m_s'),
    ])
    df = df.with_columns([
        pl.when((pl.col('dt_s') == 0) | pl.col('dt_s').is_null())
          .then(None)
          .otherwise(pl.col('dv_m_s') / pl.col('dt_s'))
          .alias('accel_m_s2')
    ])
    return df

def run_altitude_mapping(vehicle_df: pl.DataFrame, lat_col: str, lon_col: str,
                         alt_col: str, geo_elev_model_path: str, use_process_map: bool,
                         output_path: str) -> pl.DataFrame:
    required = {lat_col, lon_col, alt_col}
    missing = required - set(vehicle_df.columns)
    if missing:
        print(f'[Warning] Missing coordinates for altitude mapping: {missing}')
        return vehicle_df
    geo_elev_model_coord = geo.load_geo_elev_model_v2(geo_elev_model_path)
    coords = vehicle_geodetic_to_coord(vehicle_df)
    mapper = geo.CoordinateMapper(altitude_data=geo_elev_model_coord,
                                  distance_calculator=geo.HaversineDistance(), use_kdtree=True)
    mapped_elevs = mapper.map_elevations_v2(coords, use_process_map=use_process_map)
    if len(mapped_elevs) > 1 and abs(mapped_elevs[0] - mapped_elevs[1]) > 10:
        mapped_elevs[0] = mapped_elevs[1]    
    updated_coords = geo.merge_altitudes(coords, mapped_elevs)
    updated_vehicle_df = update_dataset(vehicle_df, updated_coords, 
                                        lat_col, lon_col, alt_col)
    updated_vehicle_df = updated_vehicle_df.fill_nan(None)
    # updated_vehicle_df.write_csv(output_path, null_value='')

    return updated_vehicle_df

def process_for_grade(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col('lat_deg').shift(1).alias('prev_lat'),
        pl.col('lon_deg').shift(1).alias('prev_lon'),
        pl.col('alt_m').shift(1).alias('prev_alt'),
    ])
    hav = geo.HaversineDistance(radius=6371000.0)
    def compute_distance(s: dict) -> float:
        try:
            if s['prev_lat'] is None or s['prev_lon'] is None:
                return None
            p1 = geo.Point(lat=float(s['prev_lat']), lon=float(s['prev_lon']))
            p2 = geo.Point(lat=float(s['lat_deg']), lon=float(s['lon_deg']))
            return float(hav.calculate(p1, p2))
        except Exception:
            return None
    temp_df = df.select(['prev_lat', 'prev_lon', 'lat_deg', 'lon_deg'])
    distances = [compute_distance(row) for row in temp_df.to_dicts()]
    df = df.with_columns(pl.Series('dist_m', distances))
    df = df.with_columns([
        ((pl.col('alt_m') - pl.col('prev_alt')) / pl.col('dist_m') * 100).alias('grade_pct')])
    df = df.drop(['prev_lat', 'prev_lon', 'prev_alt', 'dist_m'])
    return df

def correct_headers(df: pl.DataFrame) -> pl.DataFrame:
    orig_cols = df.columns
    new_names = []
    keep_cols = []
    seen = set()
    for col in orig_cols:
        new_name = col
        if re.match(r'^_duplicated_\d+$', col):
            continue
        elif re.match(r'^NOX\(ppm\)_duplicated_\d+$', col):
            new_name = 'nox_node_02'
        elif col == 'O2R(%)_duplicated_0':
            new_name = 'o2r'
        elif col in {'0Ch Engine RPM(rpm)', '0Fh Intake air temperature(∞C)'}:
            new_name = 'nox_node_02'
        if new_name in seen:
            continue
        seen.add(new_name)
        keep_cols.append(col)
        new_names.append(new_name)
    df = df.select(keep_cols)
    df.columns = new_names
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
            df = process_nox_cols(df, shift_count=5)
            if {'lat_deg', 'lon_deg', 'alt_m'}.issubset(set(df.columns)):
                out_path_alt = os.path.join(output_folder, f'mapped_{year}_{make}_{model}_ECM_{file_index}.csv')
                df = run_altitude_mapping(
                    vehicle_df=df,
                    lat_col='lat_deg',
                    lon_col='lon_deg',
                    alt_col='alt_m',
                    geo_elev_model_path=geo_elev_model_path,
                    use_process_map=True,
                    output_path=out_path_alt,
                )
                df = process_for_grade(df)
            else:
                print(f'[Warning] Missing geodetic columns in {meta.get("file")}; skipping altitude mapping and grade.')
            df = process_for_vsp(df)
            df = fix_duplicate_columns(df)
            df = correct_headers(df)
            final_path = os.path.join(output_folder, f'final_{year}_{make}_{model}_ECM_{file_index}.csv')
            df.write_csv(final_path, null_value='')
            print(f'[{file_index}] Completed pipeline, wrote {final_path}')


if __name__ == '__main__':
    main()
    sys.exit(0)

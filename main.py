import sys
import math

import polars as pl
import numpy as np

from dataset import load_dataset, vehicle_geodetic_to_coord, update_dataset
import geospatial as geo


def delta(values: list[float]) -> list[float]:
    out = []
    for i in range(len(values) - 1):
        out.append(values[i + 1] - values[i])
    out.append(np.nan)
    return out

def get_accel(dv: list[float], dt: list[float]) -> list[float]:
    if len(dv) != len(dt):
        raise ValueError('dv and dt must be the same length.')
    out = []
    for dv_i, dt_i in zip(dv, dt):
        if dt_i == 0 or math.isnan(dt_i):
            out.append(np.nan)
        else:
            out.append(dv_i / dt_i)
    return out

def get_dist_trav(speed: list[float], dt: list[float]) -> list[float]:
    if len(speed) != len(dt):
        raise ValueError('speed and dt must be the same length.')
    dist = []
    d_sum = 0.0
    for v, delta_t in zip(speed, dt):
        if not math.isnan(v) and not math.isnan(delta_t):
            d_sum += v * delta_t
        dist.append(d_sum / 1000.0)
    return dist


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float, 
    radius: float = 6371000.0) -> float:
    rlat1, rlon1 = math.radians(lat1), math.radians(lon1)
    rlat2, rlon2 = math.radians(lat2), math.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = (math.sin(dlat / 2)**2
         + math.cos(rlat1)*math.cos(rlat2)*(math.sin(dlon / 2)**2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def get_grade(lat: list[float], lon: list[float], dh: list[float]) -> list[float]:
    if not (len(lat) == len(lon) == len(dh)):
        raise ValueError('lat, lon, dh must be the same length.')
    grades = []
    for i in range(len(lat) - 1):
        run_m = haversine_meters(lat[i], lon[i], lat[i + 1], lon[i + 1])
        if run_m == 0 or math.isnan(run_m):
            grades.append(np.nan)
        else:
            grades.append(dh[i] / run_m)
    grades.append(np.nan)
    return grades


def compute_vsp(speed_m_s: float, accel_m_s2: float, grade: float) -> float:
    if any(map(math.isnan, [speed_m_s, accel_m_s2, grade])):
        return np.nan
    g = 9.81
    return speed_m_s * (1.1*accel_m_s2 + g*grade + 0.132) + 0.000302*(speed_m_s**3)


def get_vsp(speed_array: list[float], accel_array: list[float], 
        grade_array: list[float]) -> list[float]:
    if not (len(speed_array) == len(accel_array) == len(grade_array)):
        raise ValueError('speed, accel, grade arrays must be the same length.')
    out = []
    for v, a, g in zip(speed_array, accel_array, grade_array):
        out.append(compute_vsp(v, a, g))
    return out


def calibrate_sensor(nh3: list[float]) -> list[float]:
    out = []
    for val in nh3:
        if math.isnan(val):
            out.append(np.nan)
        else:
            out.append(1.0553*val + 0.3223)
    return out


def process_df_for_vsp(df: pl.DataFrame,time_col='elapse_sec',
        speed_col='gps_speed_m_s',alt_col='alt_m',lat_col='lat_deg',
        lon_col='lon_deg',is_gasoline=True) -> pl.DataFrame:
    t_vals = df[time_col].to_numpy()
    speed_vals = df[speed_col].to_numpy()
    alt_vals = df[alt_col].to_numpy()
    lat_vals = df[lat_col].to_numpy()
    lon_vals = df[lon_col].to_numpy()
    f_lat, f_lon, f_alt, f_t, f_speed = geo.filter_trackpoints(
        lat_vals, lon_vals, alt_vals, t_vals, speed_vals,dist_thresh=5.0,
        elev_thresh=0.5)
    dt_vals = delta(f_t)
    dv_vals = delta(f_speed)
    dh_vals = delta(f_alt)
    a_vals = get_accel(dv_vals, dt_vals)
    dist_km = get_dist_trav(f_speed, dt_vals)
    grade_vals = get_grade(f_lat, f_lon, dh_vals)
    vsp_vals = get_vsp(f_speed, a_vals, grade_vals)
    filtered_df = pl.DataFrame({time_col: f_t,speed_col: f_speed,
        alt_col: f_alt,lat_col: f_lat,lon_col: f_lon,'dt': dt_vals,
        'dv': dv_vals,'a': a_vals,'dist_km': dist_km,'dh': dh_vals,
        'grade': grade_vals,'vsp': vsp_vals})
    if is_gasoline and 'NOX_NODE10' in df.columns and 'NOX_NODE12' in df.columns:
        node10_list = df['NOX_NODE10'].to_numpy()
        node12_list = df['NOX_NODE12'].to_numpy()
        node10_shifted = node10_list[5:] + [np.nan]*5
        nox_vals = []
        nh3_vals = []
        for val12, val10 in zip(node12_list, node10_shifted):
            if not math.isnan(val12) and not math.isnan(val10):
                nox_vals.append(val12 - val10)
                nh3_vals.append(val10)
            else:
                nox_vals.append(np.nan)
                nh3_vals.append(np.nan)
        # nh3_cal = calibrate_sensor(nh3_vals)
        # nox_clipped = [0 if (v < 0) else v for v in nox_vals]

    return filtered_df

def run_altitude_mapping(vehicle_df: pl.DataFrame,lat_col: str,lon_col: str,
        alt_col: str,geo_elev_model_path: str,use_process_map: bool, 
        output_path: str) -> pl.DataFrame:
    geo_elev_model_coord = geo.load_geo_elev_model_v2(geo_elev_model_path)
    coords = vehicle_geodetic_to_coord(vehicle_df.select([lat_col, lon_col, alt_col]))
    mapper = geo.CoordinateMapper(altitude_data=geo_elev_model_coord,
                distance_calculator=geo.HaversineDistance(), use_kdtree=True)
    mapped_elevs = mapper.map_elevations_v2(coords, use_process_map=use_process_map)
    from geospatial import merge_altitudes
    updated_coords = merge_altitudes(coords, mapped_elevs)
    updated_vehicle_df = update_dataset(vehicle_df, updated_coords, lat_col, lon_col, alt_col)
    updated_vehicle_df = updated_vehicle_df.fill_nan(None)
    updated_vehicle_df.write_csv(output_path, null_value='')

    return updated_vehicle_df

def main():    
    vehicle_data_path    = '01_ECM.csv'
    geo_elev_model_path  = './data/geo/drive_cycle_route.json'
    output_path_alt      = 'process-updated_v1_01_ECM.csv'
    output_path_final    = 'final_processed.csv'
    vehicle_df = load_dataset(dataset_path=vehicle_data_path)
    updated_vehicle_df = run_altitude_mapping(vehicle_df=vehicle_df,
        lat_col='lat_deg',lon_col='lon_deg',alt_col='alt_m',
        geo_elev_model_path=geo_elev_model_path,use_process_map=True,
        output_path=output_path_alt)
    if 'speed_mph' in updated_vehicle_df.columns:
        updated_vehicle_df = updated_vehicle_df.with_columns(
            (pl.col('speed_mph') * 0.44704).alias('gps_speed_m_s'))
    processed_df = process_df_for_vsp(df=updated_vehicle_df,
            time_col='elapse_sec',speed_col='gps_speed_m_s',alt_col='alt_m',
            lat_col='lat_deg',lon_col='lon_deg',is_gasoline=False)
    processed_df = processed_df.fill_nan(None).slice(1, None)
    processed_df.write_csv(output_path_final, null_value='')


if __name__ == '__main__':    
    main()
    sys.exit(0)

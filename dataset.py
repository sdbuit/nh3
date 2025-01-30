import os
import sys
import re
from typing import Tuple, Dict, List, Optional
from tqdm.contrib.concurrent import thread_map

import polars as pl

from geospatial import Coordinate, Point


COLUMN_ALIASES = {
    'timestamp_raw': ['timestamp_raw', 'time_raw', 'Timestamp Raw', 'Date & Time'],
    'elapse_hms': ['elapse_hms', 'elapsed_hms', 'Elapse Time (hh:mm:ss)'],
    'elapse_sec': ['elapse_sec', 'elapsed_seconds', 'time_s', 'Elapse Time (sec)'],
    
    'tc1_degC': ['tc1_degC', 'TC1(degC)'],
    'tc2_degC': ['tc2_degC', 'TC2(degC)'],
    'tc3_degC': ['tc3_degC', 'TC3(degC)'],
    'tc4_degC': ['tc4_degC', 'TC4(degC)'],
    
    'canopen_state_0x11': ['canopen_state_0x11', 'CANopenState_0x11()'],
    'canopen_error_code_0x11': ['canopen_error_code_0x11', 'CANopen_Error_Code_0x11()'],
    'canopen_error_reg_0x11': ['canopen_error_reg_0x11', 'CANopen_Error_Reg_0x11()'],
    
    'nox_cant': ['nox_cant', 'noxCANt', 'NOX(ppm)'],
    'lam': ['lam', 'LAM()'],
    'o2r': ['o2r', 'O2R(%)'],
    'rpvs_ohms': ['rpvs_ohms', 'RPVS(ohms)'],
    
    'canopen_state_0x12': ['canopen_state_0x12', 'CANopenState_0x12()'],
    'canopen_error_code_0x12': ['canopen_error_code_0x12', 'CANopen_Error_Code_0x12()'],
    'canopen_error_reg_0x12': ['canopen_error_reg_0x12', 'CANopen_Error_Reg_0x12()'],
    'ecm_errcode1_0x12': ['ecm_errcode1_0x12', 'ECM_ErrCode1_0x12()'],
    'ecm_auxiliary_0x12': ['ecm_auxiliary_0x12', 'ECM_Auxiliary_0x12()'],
    'ecm_errcode2_0x12': ['ecm_errcode2_0x12', 'ECM_ErrCode2_0x12()'],
    
    'nh3_ppm': ['nh3_ppm', 'NH3(ppm)'],
    'mode_hex': ['mode_hex', 'MODE(hex)'],
    'vh_volt': ['vh_volt', 'VH(V)'],

    'canopen_state_0x15': ['canopen_state_0x15', 'CANopenState_0x15()'],
    'canopen_error_code_0x15': ['canopen_error_code_0x15', 'CANopen_Error_Code_0x15()'],
    'canopen_error_reg_0x15': ['canopen_error_reg_0x15', 'CANopen_Error_Reg_0x15()'],
    'ecm_errcode1_0x15': ['ecm_errcode1_0x15', 'ECM_ErrCode1_0x15()'],
    'ecm_auxiliary_0x15': ['ecm_auxiliary_0x15', 'ECM_Auxiliary_0x15()'],
    'ecm_errcode2_0x15': ['ecm_errcode2_0x15', 'ECM_ErrCode2_0x15()'],
    
    'speed_kmh': ['speed_kmh', 'Speed_kmh(km/h)'],
    'speed_mph': ['speed_mph', 'Speed_mph(mph)'],
    'course_deg': ['course_deg', 'Course(degrees)'],
    'lat_deg': ['lat_deg', 'latitude', 'lat', 'Latitude', 'LAT', 'Latitude(degrees)'],
    'lon_deg': ['lon_deg', 'longitude', 'lon', 'Longitude', 'LON', 'Longitude(degrees)'],
    'alt_m': ['alt_m', 'altitude_m', 'altitude', 'Alt', 'ALT_m', 'Altitude_m(m)'],
    'alt_ft': ['alt_ft', 'Altitude_ft(ft)'],
    'sat_type': ['sat_type', 'SatType()'],
    'sat_count': ['sat_count', 'SatCount()'],
    
    'rpm_00': ['rpm_00', 'RPM_$00(rpm)'],
    'vss_00': ['vss_00', 'VSS_$00(km/h)'],
    'load_pct_00': ['load_pct_00', 'LOAD_PCT_$00(%)'],
    'o2s12_00': ['o2s12_00', 'O2S12_$00(V)'],
    'obdsup_00': ['obdsup_00', 'OBDSUP_$00()'],
    'mil_dist_00': ['mil_dist_00', 'MIL_DIST_$00(km)'],
    'rpm_01': ['rpm_01', 'RPM_$01(rpm)']
}

ECM_COL_NAMES = [
    'timestamp_raw', 'elapse_hms', 'elapse_sec',  # Time columns
    'tc1_degC', 'tc2_degC', 'tc3_degC', 'tc4_degC',  # Thermocouples
    'nox_cant', 'lam', 'o2r', 'rpvs_ohms',  # Gas sensor data
    'canopen_state_0x11', 'canopen_error_code_0x11', 'canopen_error_reg_0x11',
    'canopen_state_0x12', 'canopen_error_code_0x12', 'canopen_error_reg_0x12',
    'ecm_errcode1_0x12', 'ecm_auxiliary_0x12', 'ecm_errcode2_0x12',
    'nh3_ppm', 'mode_hex', 'vh_volt',
    'canopen_state_0x15', 'canopen_error_code_0x15', 'canopen_error_reg_0x15',
    'ecm_errcode1_0x15', 'ecm_auxiliary_0x15', 'ecm_errcode2_0x15',
    'speed_kmh', 'speed_mph', 'course_deg', 'lat_deg', 'lon_deg', 'alt_m', 'alt_ft', 'sat_type', 'sat_count',
    'rpm_00', 'vss_00', 'load_pct_00', 'o2s12_00', 'obdsup_00', 'mil_dist_00', 'rpm_01'
]

ECM_COL_INDICIES = list(range(len(ECM_COL_NAMES)))

SCHEMA_OVERRIDE = {
    'RPM_$01(rpm)': pl.Float64,  # RPM should be float, not int
    'Speed_kmh(km/h)': pl.Float64,
    'Speed_mph(mph)': pl.Float64,
    'Altitude_m(m)': pl.Float64,
    'Altitude_ft(ft)': pl.Float64,
    'O2S12_$00(V)': pl.Float64,
    'RPVS(ohms)': pl.Float64,
}

# ECM_COL_NAMES = [
#     # index 0..2
#     'timestamp_raw',
#     'elapse_hms',
#     'elapse_sec',
#     # index 3..7
#     'nox_cant',
#     'o2r_1',
#     'afr',
#     'pkpa',
#     # index 8..13
#     'canopen_state_0x10',
#     'canopen_error_code_0x10',
#     'canopen_error_reg_0x10',
#     'ecm_errcode1_0x10',
#     'ecm_auxiliary_0x10',
#     'ecm_errcode2_0x10',
#     # indexes 14..17
#     'nox_2',
#     'lam',
#     'o2r_2',
#     'rpvs_ohms',
#     # index 18..20
#     'can_state_1',
#     'can_state_2',
#     'can_state_3',
#     # index 21..23
#     'ecm_errcode1',
#     'ecm_auxiliary',
#     'ecm_errcode2',
#     # index 24..30
#     'speed_mph',
#     'course_deg',
#     'lat_deg',
#     'lon_deg',
#     'alt_m',
#     'sat_type',
#     'sat_count',
#     # index 31..34
#     'can_bus_47',
#     'can_bus_48',
#     'can_bus_49',
#     'can_bus_50'
# ]

# ECM_COL_INDICIES = [
#     0, 1, 2,                        # Timestamps
#     4, 5, 6, 7,                     # nox_cant, o2r_1, afr, pkpa
#     8, 9, 10, 11, 12, 13,           # CANopen / ECM
#     26, 27, 28, 29,                 # nox_2, lam, o2r_2, rpvs_ohms
#     30, 31, 32,                     # can_state_1..3
#     33, 34, 35,                     # ecm_errcode1, ecm_auxiliary, ecm_errcode2
#     37, 38, 39, 40, 41, 42, 43,     # speed_mph, course_deg, lat_deg, lon_deg, alt_m, sat_type, sat_count
#     47, 48, 49, 50                  # can_bus_47..50
# ]

def find_best_skip_rows(dataset_path: str, max_skip: int = 10) -> Optional[int]:
    recognized_aliases = set()
    for alias_list in COLUMN_ALIASES.values():
        recognized_aliases.update(alias_list)
    for skip_val in range(max_skip + 1):
        try:
            sample_df = pl.read_csv(dataset_path, has_header=True, n_rows=1, skip_rows=skip_val)
            csv_cols = sample_df.columns
            if any(col in recognized_aliases for col in csv_cols):
                return skip_val
        except Exception:
            continue
    return None

def apply_aliases(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {}
    for canonical_name, alias_list in COLUMN_ALIASES.items():
        for alias in alias_list:
            if alias in df.columns:
                rename_map[alias] = canonical_name
                break
    if rename_map:
        df = df.rename(rename_map)
    return df


class DatasetLoader:
    ECM_PATTERN = re.compile(r'\d{2}_ECM\.csv')
    AUTO5GAS_FILENAME = None
    DATASET_COLUMNS = {'ECM': ECM_COL_NAMES,'AUTO5GAS': None}

    def __init__(self, base_path: str):
        self.base_path = base_path

    def _identify_dataset_type(self, filename: str) -> str:
        if self.ECM_PATTERN.match(filename):
            return 'ECM'
        elif filename == self.AUTO5GAS_FILENAME:
            return 'AUTO5GAS'
        return 'UNKNOWN'

    def load_dataset(self, dataset_path: str) -> Tuple[pl.DataFrame, Dict]:
        filename = os.path.basename(dataset_path)
        dataset_type = self._identify_dataset_type(filename)
        if dataset_type == 'UNKNOWN':
            print(f'[Error] Unsupported dataset type: {filename}')
            return pl.DataFrame(), {}
        skip_guess = find_best_skip_rows(dataset_path, max_skip=15)
        if skip_guess is None:
            print(f'[Warning] Could not find a valid header in {dataset_path} (up to 15 lines).')
            return pl.DataFrame(), {}
        try:
            df = pl.read_csv(dataset_path, has_header=True, skip_rows=skip_guess)
        except Exception as e:
            print(f'[Error] reading {dataset_path}: {e}')
            return pl.DataFrame(), {}
        df = apply_aliases(df)
        expected_columns = self.DATASET_COLUMNS[dataset_type]
        parts = os.path.normpath(dataset_path).split(os.sep)
        metadata = {
            'year': parts[-4] if len(parts) >= 4 else 'Unknown',
            'make': parts[-3] if len(parts) >= 3 else 'Unknown',
            'model': parts[-2] if len(parts) >= 2 else 'Unknown',
            'dataset_type': dataset_type,
            'dataset_filename': filename,
            'skip_rows_used': skip_guess}

        return df, metadata

    def load_all_datasets_parallel(self) -> List[Tuple[pl.DataFrame, Dict]]:
        dataset_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.base_path)
            for file in files if self._identify_dataset_type(file) != 'UNKNOWN'
        ]
        results = thread_map(self.load_dataset, dataset_paths, chunksize=5)
        
        return results

def load_dataset(dataset_path: str, has_headers: bool = True, 
    skip_rows: int = None, columns=ECM_COL_INDICIES, 
    new_columns=ECM_COL_NAMES) -> pl.DataFrame:
    skip_guess = find_best_skip_rows(dataset_path, max_skip=20)
    df = pl.read_csv(dataset_path, has_header=has_headers, skip_rows=skip_guess,
        columns=columns, new_columns=new_columns)
    df = df.drop([c for c in df.columns if df[c].null_count() == df.height])
    
    return df

def vehicle_geodetic_to_coord(df: pl.DataFrame) -> List[Coordinate]:
    coord_list = []
    if not {'lat_deg', 'lon_deg', 'alt_m'}.issubset(df.columns):
        return coord_list
    for row in df.to_dicts():
        coord_list.append(
            Coordinate(
                point=Point(
                    lat=row['lat_deg'],
                    lon=row['lon_deg']
                ),
                altitude=row['alt_m']
            )
        )
    return coord_list

def update_dataset(df: pl.DataFrame, updated_coords: List[Coordinate], 
    lat: str, lon: str, alt: str) -> pl.DataFrame:
    if not updated_coords:
        return df
    new_data = {
        lat: [float(c.point.lat) for c in updated_coords],
        lon: [float(c.point.lon) for c in updated_coords],
        alt: [float(c.altitude) for c in updated_coords],}
    updated_df = pl.DataFrame(new_data)
    existing_cols = set(df.columns)
    drop_cols = [c for c in [lat, lon, alt] if c in existing_cols]
    df = df.drop(drop_cols).hstack(updated_df)
    return df


if __name__ == '__main__':
    test_path = '.\\data\\vehicle_v2\\2007\\Dodge\\RAM1500\\01_ECM.csv'
    skip_guess = find_best_skip_rows(test_path, max_skip=10)
    df_test = load_dataset(test_path)
    sys.exit(0)

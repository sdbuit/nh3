import os
import polars as pl
from typing import Tuple, Dict, List, Optional
from tqdm.contrib.concurrent import thread_map
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
    'canopen_error_code_0x11': ['CANopen_Error_Code_0x11()'],
    'canopen_error_reg_0x11': ['canopen_error_reg_0x11', 'CANopen_Error_Reg_0x11()'],

    'ecm_errcode1_0x11': ['ECM_ErrCode1_0x11()'],
    'ecm_auxiliart_0x11': ['ECM_Auxiliary_0x11()'],
    'ecm_errcode_0x11': ['ECM_ErrCode2_0x11()'],    
    
    'nox_cant': ['nox_cant', 'noxCANt', 'NOX(ppm)'],
    'lam': ['lam', 'LAM()'],
    'o2r': ['o2r', 'O2R(%)'],
    'rpvs_ohms': ['rpvs_ohms', 'RPVS(ohms)'],
    'pkpa_kpa': ['PKPA(kPa)'],
    
    					
    'canopen_state_ox10': ['CANopenState_0x10()'],
    'canopen_error_code_ox10': ['CANopen_Error_Code_0x10()'],
    'canopen_error_reg_0x10': ['CANopen_Error_Reg_0x10()'],
    'ecm_errcode1_0x10': ['ECM_ErrCode1_0x10()'],
    'ecm_auxiliary_0x10': ['ECM_Auxiliary_0x10()'],
    'ecm_errcode2_0x10': ['ECM_ErrCode2_0x10()'],

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


    'speed_kmh': ['Speed(kmh)', 'Speed_kmh(km/h)'],
    'speed_mph': ['Speed(mph)', 'Speed_mph(mph)'],
    'course_deg': ['course_deg', 'Course(degrees)'],
    'lat_deg': ['lat_deg', 'Latitude(degrees)', 'Latitude(deg)'],
    'lon_deg': ['lon_deg', 'Longitude(degrees)', 'Longitude(deg)'],
    'alt_m': ['Altitude_m(m)', 'Altitude(m)'],
    'alt_ft': ['alt_ft', 'Altitude_ft(ft)'],
    'sat_type': ['SatType()'],
    'sat_count': ['SatCount()'],

    'obd_rpm': ['rpm_00', 'RPM_$00(rpm)', '0Ch Engine RPM(rpm)'],
    'obd_vss_kmh': ['vss_00', 'VSS_$00(km/h)', 'VSS_$01(km/h)'],
    'obd_engine_load_pct': ['load_pct_00', 'LOAD_PCT_$00(%)', '04h Calculated engine load(%)'],
    'obd_o2s12': ['o2s12_00', 'O2S12_$00(V)'],
    'obd_sup': ['obdsup_00', 'OBDSUP_$00()'],
    'obd_mil_dist': ['mil_dist_00', 'MIL_DIST_$00(km)'],
    'obd_rpm': ['RPM_$01(rpm)'],
    'obd_clt_dist_km': ['CLR_DIST_$00(km)'],
    'obd_cat_temp': ['CATEMP11_$00(âˆžC)', 'CATEMP11_$00(Â°C)', 'CATEMP11_$00(∞C)'],
    'obd_cat_temp_bank_1': ['3Ch Catalyst Temperature: Bank 1'],
    'obd_o2_sensor_bank2': ['13h O2 sensors present (in 2 banks)()'],
    'obd_O2_S12_v': ['O2S12_$00(V)'],
    'obd_maf_gs': ['MAF_$00(g/s)'],
    'obd_mil_dist_km': ['MIL_DIST_$00(km)'],
    'obd_iat': ['IAT_$00(âˆžC)'],
    'obd_cat_rdy': ['CAT_RDY_$00()'],
    'obd_speed_kmh': ['0Dh Vehicle speed(km/h)'],
}

SCHEMA_OVERRIDE = {
    'RPM_$01(rpm)': pl.Float64,
    'Speed_kmh(km/h)': pl.Float64,
    'Speed_mph(mph)': pl.Float64,
    'Altitude_m(m)': pl.Float64,
    'Altitude_ft(ft)': pl.Float64
}


def detect_header_row(file_path: str, max_skip: int = 100) -> Optional[int]:
    recognized_aliases = {alias.lower().strip() for aliases in COLUMN_ALIASES.values() for alias in aliases}
    candidate = None
    for skip in range(max_skip + 1):
        try:
            df_sample = pl.read_csv(file_path, has_header=True, n_rows=1, skip_rows=skip)
            header = [col.lower().strip() for col in df_sample.columns]
            score = sum(1 for col in header if col in recognized_aliases)
            if score >= 4:
                candidate = skip
        except Exception:
            continue
    if candidate is None:
        print(f'[Warning] No valid header found in {file_path} (checked {max_skip} lines).')
        return 0
    return candidate

def apply_aliases(df: pl.DataFrame) -> pl.DataFrame:
    new_names = []
    for col in df.columns:
        normalized = col.lower().strip()
        renamed = None
        for canonical, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                if normalized == alias.lower().strip():
                    renamed = canonical
                    break
            if renamed:
                break
        new_names.append(renamed if renamed is not None else col)
    df.columns = new_names
    return df

def fix_duplicate_columns(df: pl.DataFrame) -> pl.DataFrame:
    unique_cols = []
    seen = set()
    for col in df.columns:
        if col.strip() == '':
            continue
        if col in seen:
            continue
        seen.add(col)
        unique_cols.append(col)
    return df.select(unique_cols)

def load_dataset(dataset_path: str) -> Tuple[pl.DataFrame, Dict]:
    filename = os.path.basename(dataset_path)
    header_row = detect_header_row(dataset_path)
    if header_row is None:
        print(f'[Warning] Falling back to using the first row as header for {dataset_path}')
        header_row = 0
    try:
        df = pl.read_csv(
            dataset_path,
            has_header=True,
            skip_rows=header_row,
            dtypes=SCHEMA_OVERRIDE,
            infer_schema_length=10000,
            ignore_errors=True,
            quote_char="'",
            null_values=['', 'NULL']
        )
        df = apply_aliases(df)
        df = fix_duplicate_columns(df)
        parts = os.path.normpath(dataset_path).split(os.sep)
        try:
            idx = parts.index('vehicle_v2')
            year = parts[idx + 1] if len(parts) > idx + 1 else None
            make = parts[idx + 2] if len(parts) > idx + 2 else None
            model = parts[idx + 3] if len(parts) > idx + 3 else None
        except ValueError:
            year = make = model = None
        if 'ECM' in filename:
            dataset_type = 'ECM'
        else:
            dataset_type = 'unknown'
        meta = {
            'file': dataset_path,
            'skip_rows': header_row,
            'year': year,
            'make': make,
            'model': model,
            'dataset_type': dataset_type
        }
        return df, meta
    except Exception as e:
        print(f'[Error] Failed to read {dataset_path}: {e}')
        return pl.DataFrame(), {}

def load_all_datasets(base_path: str) -> List[Tuple[pl.DataFrame, Dict]]:
    dataset_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(base_path)
        for file in files if file.endswith('.csv')
    ]
    return thread_map(load_dataset, dataset_paths, chunksize=5)

class DatasetLoader:
    def __init__(self, base_path: str):
        self.base_path = base_path
    def load_all(self) -> List[Tuple[pl.DataFrame, Dict]]:
        return load_all_datasets(self.base_path)
    def load_all_datasets_parallel(self) -> List[Tuple[pl.DataFrame, Dict]]:
        return load_all_datasets(self.base_path)

def vehicle_geodetic_to_coord(df: pl.DataFrame) -> List[Coordinate]:
    """
    Convert a vehicle DataFrame (with columns like 'lat_deg', 'lon_deg', 
    'alt_m') into a list of Coordinate objects.
    """
    coords = []
    for row in df.iter_rows(named=True):
        try:
            lat = float(row['lat_deg'])
            lon = float(row['lon_deg'])
            alt = float(row['alt_m']) if row.get('alt_m') is not None else 0.0
            coords.append(Coordinate(point=Point(lat=lat, lon=lon), altitude=alt))
        except Exception:
            continue
    return coords

def update_dataset(vehicle_df: pl.DataFrame, updated_coords: List[Coordinate],
                   lat_col: str, lon_col: str, alt_col: str) -> pl.DataFrame:
    """
    Update the given vehicle DataFrame with new altitude values from 
    updated_coords.
    """
    altitudes = [coord.altitude for coord in updated_coords]
    return vehicle_df.with_columns([pl.Series(name=alt_col, values=altitudes)])

if __name__ == '__main__':
    base_path = './data/vehicle_v2'
    loader = DatasetLoader(base_path)
    datasets = loader.load_all()
    for df, meta in datasets:
        if not df.is_empty():
            print(f'Loaded {meta.get("file")} with {df.shape[0]} rows and {df.shape[1]} columns.')
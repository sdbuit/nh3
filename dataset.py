import os
import polars as pl
from typing import Tuple, Dict, List, Optional
from tqdm.contrib.concurrent import thread_map
from geospatial import Coordinate, Point

COLUMN_ALIASES = {
    'timestamp_raw': ['Date & Time'],
    'elapse_hms': ['Elapse Time (hh:mm:ss)'],
    'elapse_sec': ['Elapse Time (sec)'],

    'tc1_degC': ['tc1_degC', 'TC1(degC)'],
    'tc2_degC': ['tc2_degC', 'TC2(degC)'],
    'tc3_degC': ['tc3_degC', 'TC3(degC)'],
    'tc4_degC': ['tc4_degC', 'TC4(degC)'],

    # CANopen/ECM fields for 0x10
    'canopen_state_0x10': ['CANopenState_0x10()'],
    'canopen_error_code_0x10': ['CANopen_Error_Code_0x10()'],
    'canopen_error_reg_0x10': ['CANopen_Error_Reg_0x10()'],
    'ecm_errcode1_0x10': ['ECM_ErrCode1_0x10()'],
    'ecm_auxiliary_0x10': ['ECM_Auxiliary_0x10()'],
    'ecm_errcode2_0x10': ['ECM_ErrCode2_0x10()'],

    # CANopen/ECM fields for 0x11
    'canopen_state_0x11': ['CANopenState_0x11()'],
    'canopen_error_code_0x11': ['CANopen_Error_Code_0x11()'],
    'canopen_error_reg_0x11': ['CANopen_Error_Reg_0x11()'],
    'ecm_errcode1_0x11': ['ECM_ErrCode1_0x11()'],
    'ecm_auxiliary_0x11': ['ECM_Auxiliary_0x11()'],
    'ecm_errcode2_0x11': ['ECM_ErrCode2_0x11()'],

    # CANopen/ECM fields for 0x12
    'canopen_state_0x12': ['CANopenState_0x12()'],
    'canopen_error_code_0x12': ['CANopen_Error_Code_0x12()'],
    'canopen_error_reg_0x12': ['CANopen_Error_Reg_0x12()'],
    'ecm_errcode1_0x12': ['ECM_ErrCode1_0x12()'],
    'ecm_auxiliary_0x12': ['ECM_Auxiliary_0x12()'],
    'ecm_errcode2_0x12': ['ECM_ErrCode2_0x12()'],
   
    # CANopen/ECM fields for 0x15
    'canopen_state_0x15': ['CANopenState_0x15()'],
    'canopen_error_code_0x15': ['CANopen_Error_Code_0x15()'],
    'canopen_error_reg_0x15': ['CANopen_Error_Reg_0x15()'],
    'ecm_errcode1_0x15': ['ECM_ErrCode1_0x15()'],
    'ecm_auxiliary_0x15': ['ECM_Auxiliary_0x15()'],
    'ecm_errcode2_0x15': ['ECM_ErrCode2_0x15()'],   
    'nox_node_01': ['NOX(ppm)'],
    'lam': ['lam', 'LAM()'],
    'o2r': ['o2r', 'O2R(%)'],
    'AFR': ['AFR()'],
    'rpvs_ohms': ['rpvs_ohms', 'RPVS(ohms)'],
    'pkpa_kpa': ['PKPA(kPa)'],
    'nh3_ppm': ['NH3(ppm)'],
    'mode_hex': ['mode_hex', 'MODE(hex)'],
    'vh_volt': ['vh_volt', 'VH(V)'],
    'speed_kmh': ['Speed(kmh)', 'Speed_kmh(km/h)'],
    'speed_mph': ['Speed(mph)', 'Speed_mph(mph)'],
    'course_deg': ['Course(degrees)', 'Course(deg)'],
    'lat_deg': ['Latitude(degrees)', 'Latitude(deg)'],
    'lon_deg': ['Longitude(degrees)', 'Longitude(deg)'],
    'alt_m': ['Altitude_m(m)', 'Altitude(m)'],
    'alt_ft': ['alt_ft', 'Altitude_ft(ft)'],
    'sat_type': ['SatType()'],
    'sat_count': ['SatCount()'],
    
    'obd_rpm': ['rpm_00', 'RPM_$00(rpm)', '0Ch Engine RPM(rpm)', '0Ch Engine RPM(rpm)'],
    'obd_vss_kmh_0': ['vss_00', 'VSS_$00(km/h)'],
    'obd_vss_kmh_1': ['VSS_$01(km/h)'],
    'obd_engine_load_pct': ['load_pct_00', 'LOAD_PCT_$00(%)', '04h Calculated engine load(%)'],
    'obd_o2s12': ['O2S12_$00(V)'],
    'obd_sup': ['OBDSUP_$00()'],
    'obd_mil_dist': ['mil_dist_00'],
    'obd_mil_dist_km': ['21h Distance traveled with MIL on(km)', 'MIL_DIST_$00(km)'],
    'obd_rpm': ['RPM_$01(rpm)'],
    'obd_clt_dist_km': ['CLR_DIST_$00(km)'],
    'obd_cat_temp': ['CATEMP11_$00(âˆžC)', 'CATEMP11_$00(Â°C)', 'CATEMP11_$00(∞C)'],
    'obd_cat_temp_bank_1': ['3Ch Catalyst Temperature: Bank 1'],
    'obd_o2_sensor_bank2': ['13h O2 sensors present (in 2 banks)()'],
    'obd_O2_S12_v': ['O2S12_$00(V)'],
    'obd_maf_gs': ['MAF_$00(g/s)'],
    'obd_mil_dist_km': ['MIL_DIST_$00(km)', '21h Distance traveled with MIL on(km)'],
    'obd_iat': ['IAT_$00(âˆžC)'],
    'obd_cat_rdy': ['CAT_RDY_$00()'],
    'obd_speed_kmh': ['0Dh Vehicle speed(km/h)'],
    'obd_intake_air_temp': ['0Fh Intake air temperature(âˆžC)']
}

SCHEMA_OVERRIDE = {}

def detect_header_row(file_path: str, max_skip: int = 100) -> Optional[int]:
    recognized_aliases = {alias.lower().strip() for aliases in COLUMN_ALIASES.values() for alias in aliases}
    candidate = None
    for skip in range(max_skip + 1):
        try:
            df_sample = pl.read_csv(file_path, has_header=False, n_rows=1, skip_rows=skip)
            row = df_sample.row(0)
            header = [str(x).strip() for x in row]
            if sum(1 for col in header if col) < 4:
                continue
            score = sum(1 for col in header if col.lower() in recognized_aliases)
            if score >= 4:
                candidate = skip
                break
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
    seen = {}
    new_cols = []
    cols_to_select = []
    for i, col in enumerate(df.columns):
        count = seen.get(col, 0)
        if count:
            if df[col].drop_nulls().height == 0:
                seen[col] += 1
                continue
            else:
                new_name = f'{col}_{count}'
                new_cols.append(new_name)
                cols_to_select.append((i, col))
                seen[col] += 1
        else:
            seen[col] = 0
            new_cols.append(col)
            cols_to_select.append((i, col))
    cols_to_select.sort(key=lambda x: x[0])
    selected_col_names = [col for i, col in cols_to_select]
    df = df.select(selected_col_names)
    df.columns = new_cols
    
    return df

def load_dataset(dataset_path: str) -> Tuple[pl.DataFrame, Dict]:
    filename = os.path.basename(dataset_path)
    header_row = detect_header_row(dataset_path)
    if header_row is None:
        print(f'[Warning] Using the first row as header for {dataset_path}')
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
    altitudes = [coord.altitude for coord in updated_coords]
    return vehicle_df.with_columns([pl.Series(name=alt_col, values=altitudes)])

if __name__ == '__main__':
    base_path = './data/vehicle_v2'
    loader = DatasetLoader(base_path)
    datasets = loader.load_all()
    for df, meta in datasets:
        if not df.is_empty():
            print(f'Loaded {meta.get("file")} with {df.shape[0]} rows and {df.shape[1]} columns.')

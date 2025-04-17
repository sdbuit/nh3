import os
from pathlib import Path
from typing import Dict, Set, List


CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'test_processed_output'
GEO_DATA_DIR = PROJECT_ROOT / 'data' / 'geo'
EXTENSION_DIR = CONFIG_DIR / 'extensions'
DRIVE_CYCLE_ROUTE_PATH = GEO_DATA_DIR / 'drive_cycle_route.json'

REPORTS_DIR = PROJECT_ROOT / 'reports'
PLOTS_DIR = REPORTS_DIR / 'plots'

COLUMN_ALIASES: Dict[str, List[str]] = {
    # Time
    'time_s': ['Elapse Time (sec)'],
    'timestamp_raw': ['Date & Time'],
    'elapse_hms': ['Elapse Time (hh:mm:ss)'],

    # Thermocouples
    'tc1_degC': ['tc1_degC', 'TC1(degC)'],
    'tc2_degC': ['tc2_degC', 'TC2(degC)'],
    'tc3_degC': ['tc3_degC', 'TC3(degC)'],
    'tc4_degC': ['tc4_degC', 'TC4(degC)'],

    # CANopen/ECM Status
    'canopen_state_0x10': ['CANopenState_0x10()'],
    'canopen_error_code_0x10': ['CANopen_Error_Code_0x10()'],
    'canopen_error_reg_0x10': ['CANopen_Error_Reg_0x10()'],
    'ecm_errcode1_0x10': ['ECM_ErrCode1_0x10()'],
    'ecm_auxiliary_0x10': ['ECM_Auxiliary_0x10()'],
    'ecm_errcode2_0x10': ['ECM_ErrCode2_0x10()'],
    'canopen_state_0x11': ['CANopenState_0x11()'],
    'canopen_error_code_0x11': ['CANopen_Error_Code_0x11()'],
    'canopen_error_reg_0x11': ['CANopen_Error_Reg_0x11()'],
    'ecm_errcode1_0x11': ['ECM_ErrCode1_0x11()'],
    'ecm_auxiliary_0x11': ['ECM_Auxiliary_0x11()'],
    'ecm_errcode2_0x11': ['ECM_ErrCode2_0x11()'],
    'canopen_state_0x12': ['CANopenState_0x12()'],
    'canopen_error_code_0x12': ['CANopen_Error_Code_0x12()'],
    'canopen_error_reg_0x12': ['CANopen_Error_Reg_0x12()'],
    'ecm_errcode1_0x12': ['ECM_ErrCode1_0x12()'],
    'ecm_auxiliary_0x12': ['ECM_Auxiliary_0x12()'],
    'ecm_errcode2_0x12': ['ECM_ErrCode2_0x12()'],
    'canopen_state_0x15': ['CANopenState_0x15()'],
    'canopen_error_code_0x15': ['CANopen_Error_Code_0x15()'],
    'canopen_error_reg_0x15': ['CANopen_Error_Reg_0x15()'],
    'ecm_errcode1_0x15': ['ECM_ErrCode1_0x15()'],
    'ecm_auxiliary_0x15': ['ECM_Auxiliary_0x15()'],
    'ecm_errcode2_0x15': ['ECM_ErrCode2_0x15()'],

    # Emissions/Engine Parameters
    'nox_ppm': ['NOX(ppm)'],
    'lambda': ['lam', 'LAM()'],
    'o2r_pct': ['O2R(%)'],
    'afr': ['AFR()'],
    'rpvs_ohms': ['rpvs_ohms', 'RPVS(ohms)'],
    'pkpa_kpa': ['PKPA(kPa)'],
    'nh3_ppm': ['NH3(ppm)'],
    'mode_hex': ['mode_hex', 'MODE(hex)'],
    'vh_volt': ['vh_volt', 'VH(V)'],

    # GPS and Vehicle OBD Patameters
    'speed_kmh': ['Speed(kmh)', 'Speed_kmh(km/h)', '0Dh Vehicle speed(km/h)', 'vss_00', 'VSS_$00(km/h)', 'VSS_$01(km/h)'],
    'speed_mph': ['Speed(mph)', 'Speed_mph(mph)'],
    'course_deg': ['Course(degrees)', 'Course(deg)'],
    'latitude': ['Latitude', 'Latitude(degrees)', 'Latitude(deg)'],
    'longitude': ['Longitude', 'Longitude(degrees)', 'Longitude(deg)'],
    'altitude_m': ['Altitude_m(m)', 'Altitude(m)'],
    'altitude_ft': ['alt_ft', 'Altitude_ft(ft)'],
    'sat_type': ['SatType()'],
    'sat_count': ['SatCount()'],

    'engine_rpm': ['0Ch Engine RPM(rpm)', 'rpm_00', 'RPM_$00(rpm)', 'RPM_$01(rpm)', 'EngRPM'],
    'engine_load_pct': ['load_pct_00', 'LOAD_PCT_$00(%)', '04h Calculated engine load(%)'],
    'o2s12_v': ['O2S12_$00(V)'],
    'obd_support': ['OBDSUP_$00()'],
    'mil_dist_km': ['mil_dist_00', '21h Distance traveled with MIL on(km)', 'MIL_DIST_$00(km)'],
    'clr_dist_km': ['CLR_DIST_$00(km)'],
    'cat_temp_c': ['CATEMP11_$00(âˆžC)', 'CATEMP11_$00(Â°C)', 'CATEMP11_$00(∞C)', 'CATEMP21_$00(∞C)', '3Ch Catalyst Temperature: Bank 1'],
    'o2_sensor_banks': ['13h O2 sensors present (in 2 banks)()'],
    'maf_gs': ['MAF_$00(g/s)'],
    'intake_air_temp_c': ['IAT_$00(âˆžC)', 'IAT_$00(∞C)', '0Fh Intake air temperature(âˆžC)', '0Fh Intake air temperature(∞C)'],
    'cat_ready': ['CAT_RDY_$00()'],
    'abs_baro_pressure_kpa': ['33h Absolute Barometric Pressure(kPa)'],
}

HEADER_DETECTION_KEYWORDS: List[str] = [
    'Time', 'Speed', 'Lat', 'Lon', 'Alt', 'RPM', 'Date', 'Course', 'Sat'
]

EARTH_RADIUS = 6371000 # meters
DEFAULT_VEHICLE_PARAMS: Dict[str, float] = {
    'vehicle_mass_kg': 1900.0,          # Kilograms (kg) - Adjust for typical vehicle in your fleet
    'rolling_resistance_coeff': 0.013,  # Unitless (Crr) - Typical range 0.01 to 0.015
    'drag_coeff': 0.35,                 # Unitless (Cd) - Typical range 0.25 to 0.4+
    'frontal_area_m2': 2.5,             # Square meters (m^2) - Estimate based on vehicle type
    'drivetrain_efficiency': 0.92,      # Unitless (eta) - Efficiency from engine to wheels
    'air_density_kg_m3': 1.225          # Kilograms per cubic meter (kg/m^3) - Standard sea level
}

# TODO Required columns needed after preprocessing and aliasing for validation
REQUIRED_COLUMNS_POST_PROCESSING: Set[str] = {
    'time_s',
    'speed_mph',
    'latitude',
    'longitude',
    'altitude_m_mapped', # The altitude mapped from route data
    'grade_percent',
    # TODO add 'engine_rpm', 'vsp', ...
}

REQUIRED_COLUMNS_POST_PROCESSING: Set[str] = {
    'time_s',
    'speed_mph', # Or speed_kmh
    'latitude',
    'longitude',
    'altitude_m_mapped',
    'grade_percent',
    'vsp_kw_tonne'
}

# Multiprocessing Configuration
CPU_COUNT = os.cpu_count() // 2
MAX_WORKERS = max(1, CPU_COUNT - 1) if CPU_COUNT else 1 # Leave a core free

MAX_WORKERS = max(1, CPU_COUNT - 1) if CPU_COUNT else 1 # Leave a core free

if 'grade_percent' not in REQUIRED_COLUMNS_POST_PROCESSING:
     # Note: Modifying a set requires adding, not appending
     REQUIRED_COLUMNS_POST_PROCESSING.add('grade_percent')

if __name__ == '__main__':
    print(f'Project Root: {PROJECT_ROOT}')
    print(f'Data Directory: {DATA_DIR}')
    print(f'Output Directory: {OUTPUT_DIR}')
    print(f'Route Data Directory: {GEO_DATA_DIR}')
    print(f'Route File Path: {DRIVE_CYCLE_ROUTE_PATH}')
    print(f'Route File Exists: {DRIVE_CYCLE_ROUTE_PATH.is_file()}')

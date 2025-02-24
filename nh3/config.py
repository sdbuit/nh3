# config.py
import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np


@dataclass
class Point:
    lat: np.float32
    lon: np.float32
    deg: bool = True


@dataclass
class Coordinate:
    point: Point
    altitude: np.float32


@dataclass
class ProcessingConfig:
    EARTH_RADIUS: int=6_371_000  # Meters
    KD_TREE_ENABLED: bool=True
    ELEVATION_MISSING_THRESHOLD: float=10.0  # Meters
    CHUNK_SIZE: int=50
    MAX_WORKERS: int=os.cpu_count()-1
    GEO_MODEL_PATH: Path=Path('geo/drive_cycle_route.json')  


class Config:
    def __init__(self):        
        self.BASE_DIR=Path(__file__).parent
        self.DATA_DIR=self.BASE_DIR / 'data'
        self.GEO_MODEL_PATH=self.DATA_DIR / 'geo/drive_cycle_route.json'
        self.OUTPUT_DIR=self.DATA_DIR / 'test_process_output'
        self.EARTH_RADIUS=6_371_000  # Meters
        self.KD_TREE_ENABLED=True
        self.ELEVATION_MISSING_THRESHOLD=10.0  # Meters
        self.CHUNK_SIZE=50
        self.MAX_WORKERS=os.cpu_count()-1
        self.COLUMN_ALIASES = {
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
            'o2r': ['O2R(%)'],
            'afr': ['AFR()'],
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

            'obd_rpm': ['0Ch Engine RPM(rpm)','rpm_00', 'RPM_$00(rpm)',
                        'RPM_$00(rpm)'],
            'obd_vss_kmh_0': ['vss_00', 'VSS_$00(km/h)'],
            'obd_vss_kmh_1': ['VSS_$01(km/h)'],
            'obd_engine_load_pct': ['load_pct_00', 'LOAD_PCT_$00(%)', 
                                    '04h Calculated engine load(%)'],
            'obd_o2s12': ['O2S12_$00(V)'],
            'obd_sup': ['OBDSUP_$00()'],
            'obd_mil_dist': ['mil_dist_00'],
            'obd_mil_dist_km': ['21h Distance traveled with MIL on(km)', 
                                'MIL_DIST_$00(km)'],
            'obd_rpm': ['RPM_$01(rpm)'],
            'obd_clt_dist_km': ['CLR_DIST_$00(km)'],
            'obd_cat_temp': ['CATEMP11_$00(âˆžC)', 'CATEMP11_$00(Â°C)', 
                             'CATEMP11_$00(∞C)', 'CATEMP21_$00(∞C)'],
            'obd_cat_temp_bank_1': ['3Ch Catalyst Temperature: Bank 1'],
            'obd_o2_sensor_bank2': ['13h O2 sensors present (in 2 banks)()'],
            'obd_O2_S12_v': ['O2S12_$00(V)'],
            'obd_maf_gs': ['MAF_$00(g/s)'],
            'obd_mil_dist_km': ['MIL_DIST_$00(km)', 
                                '21h Distance traveled with MIL on(km)'],
            'obd_iat': ['IAT_$00(âˆžC)', 'IAT_$00(∞C)'],
            'obd_cat_rdy': ['CAT_RDY_$00()'],
            'obd_speed_kmh': ['0Dh Vehicle speed(km/h)'],
            'obd_intake_air_temp': ['0Fh Intake air temperature(âˆžC)', 
                                    '0Fh Intake air temperature(∞C)'],
            'obd_abs_barometric_kpa':['33h Absolute Barometric Pressure(kPa)']
}

config = Config()


if __name__ == '__main__':
    print(f'Base Dir: {config.BASE_DIR}')

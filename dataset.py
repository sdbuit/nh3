import os
import polars as pl
from dataclasses import dataclass
from geospatial import Coordinate, Point
from typing import List
from geospatial import Coordinate, Point # , update_vehicle_data_with_elevation
import polars.selectors as cs

ecm_col_indices = [                     # Indices of columns in the CSV
        0, 1, 2,                        # Timestamps
        4, 5, 6, 7,                     # NOxCANt, O2R, AFR, PKPA
        8, 9, 10, 11, 12, 13,           # CANopen / ECM
        26, 27, 28, 29,                 # NOX, LAM, O2R, RPVS
        30, 31, 32,                     # CAN states
        33, 34, 35,                     # ECM states
        37, 38, 39, 40, 41, 42, 43,     # Speed, Course, Lat, Lon, Alt, Sat...
        47, 48, 49, 50                  # CAN_Bus
        ]

ecm_col_names = [                # Matching those indices above
    'timestamp_raw',             # index 0
    'elapse_hms',                # index 1
    'elapse_sec',                # index 2
    'nox_cant',                  # index 4
    'o2r_1',                     # index 5
    'afr',                       # index 6
    'pkpa',                      # index 7
    'canopen_state_0x10',
    'canopen_error_code_0x10',
    'canopen_error_reg_0x10',
    'ecm_errcode1_0x10',
    'ecm_auxiliary_0x10',
    'ecm_errcode2_0x10',         # indexes 8..13
    'nox_2',                     # index 26
    'lam',                       # index 27
    'o2r_2',                     # index 28
    'rpvs_ohms',                 # index 29
    'can_state_1',               # index 30
    'can_state_2',               # index 31
    'can_state_3',               # index 32
    'ecm_errcode1',              # index 33
    'ecm_auxiliary',             # index 34
    'ecm_errcode2',              # index 35
    'speed_mph',                 # index 37
    'course_deg',                # index 38
    'lat_deg',                   # index 39
    'lon_deg',                   # index 40
    'alt_m',                     # index 41
    'sat_type',                  # index 42
    'sat_count',                 # index 43
    'can_bus_47',                # index 47
    'can_bus_48',                # index 48
    'can_bus_49',                # index 49
    'can_bus_50'                 # index 50
]

ecm_col_dtypes = {
    'timestamp_raw': pl.Utf8,
    'elapse_hms': pl.Utf8,
    'elapse_sec': pl.Float64,
    'nox_cant': pl.Float64,
    'o2r_1': pl.Float64,
    'afr': pl.Float64,
    'pkpa': pl.Float64,
    'canopen_state_0x10': pl.Int32,
    'canopen_error_code_0x10': pl.Int32,
    'canopen_error_reg_0x10': pl.Int32,
    'ecm_errcode1_0x10': pl.Int32,
    'ecm_auxiliary_0x10': pl.Int32,
    'ecm_errcode2_0x10': pl.Int32,
    'nox_2': pl.Float64,
    'lam': pl.Float64,
    'o2r_2': pl.Float64,
    'rpvs_ohms': pl.Float64,
    'can_state_1': pl.Int32,
    'can_state_2': pl.Int32,
    'can_state_3': pl.Int32,
    'ecm_errcode1': pl.Int32,
    'ecm_auxiliary': pl.Int32,
    'ecm_errcode2': pl.Int32,
    'speed_mph': pl.Float64,
    'course_deg': pl.Float64,
    'lat_deg': pl.Float64,
    'lon_deg': pl.Float64,
    'alt_m': pl.Float64,
    'sat_type': pl.Utf8,
    'sat_count': pl.Int32,
    'can_bus_47': pl.Float64,
    'can_bus_48': pl.Float64,
    'can_bus_49': pl.Float64,
    'can_bus_50': pl.Float64,
}

def vheicle_geodetic2coord(df: pl.DataFrame):
    vehicle_coord_list = []
    for row in df.to_dicts():
        vehicle_coord_list.append(
            Coordinate(
                point=Point(
                    lat=row["lat_deg"],
                    lon=row["lon_deg"]
                ),
                altitude=row["alt_m"]  # or 0 if missing
            )
        )
    return vehicle_coord_list 


def get_geodetic_data(df: pl.DataFrame, lat_col: str, lon_col: str, 
                        alt_col: str) -> List[Coordinate]:
    """Extracts coordinates from a Polars DataFrame."""
    return [Coordinate(point=Point(lat=row[lat_col], lon=row[lon_col]),
                       altitude=row[alt_col])
            for row in df.iter_rows(named=True)]
    
def update_dataset(df: pl.DataFrame, updated_coords: List[Coordinate], 
                   lat: str, lon: str, alt: str) -> pl.DataFrame:
    """Updates dataset with new altitude values."""
    new_data = {
        lat: [float(c.point.lat) for c in updated_coords],
        lon: [float(c.point.lon) for c in updated_coords],
        alt: [float(c.altitude) for c in updated_coords],
    }
    updated_df = pl.DataFrame(new_data)
    return df.drop([lat, lon, alt]).hstack(updated_df)

def load_dataset(dataset_path:str='01_ESM.csv', has_headers:bool=True, 
                 skip_rows:int=6, columns=ecm_col_indices, 
                 new_colums=ecm_col_names) -> pl.DataFrame:
    dataset_path = '01_ECM.csv'
    df = pl.read_csv(dataset_path, has_header=True, skip_rows=6,
                     columns=ecm_col_indices, new_columns=ecm_col_names)
    df.drop([c for c in df.columns if df[c].null_count() == df.height])
    
    return df



if __name__ == '__main__':
    df = load_dataset()
    print(df)
    print(df.columns)
    # df.select(pl.col('struct_col').struct.field(''))
    
    # dataset_path = '01_ECM.csv'
    # df = pl.read_csv(dataset_path,
    #                  has_header=True, 
    #                  skip_rows=6,
    #                  columns=ecm_col_indices,
    #                  new_columns=ecm_col_names
    #                 )
    # df.drop([c for c in df.columns if df[c].null_count() == df.height])
    # schema = df.collect_schema
    # print(df.collect_schema())
    # print(df.collect_schema)
    
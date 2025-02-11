import math
import json
import logging
from functools import partial
from typing import List, Optional
from abc import ABC, abstractmethod
from tqdm.contrib.concurrent import process_map, thread_map

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

import dataset as ds
from config import config, Point, Coordinate


class GeoProcessingError(Exception):
    """Base exception for geospatial processing errors"""

class DataValidationError(GeoProcessingError):
    """TODO"""


class DistanceCalculator(ABC):
    @abstractmethod
    def calculate(self, p1: Point, p2: Point) -> np.float32:
        """Computes distance between two points"""


class HaversineDistance(DistanceCalculator):
    def __init__(self, radius: float = config.EARTH_RADIUS):
        self.radius = radius

    def _to_radians(self, point: Point) -> Point:
        return Point(
            lat=np.radians(point.lat),
            lon=np.radians(point.lon),
            deg=False
        ) if point.deg else point

    def calculate(self, p1: Point, p2: Point) -> np.float32:
        p1_rad = self._to_radians(p1)
        p2_rad = self._to_radians(p2)
        dlat = p2_rad.lat - p1_rad.lat
        dlon = p2_rad.lon - p1_rad.lon
        a = math.sin(dlat/2)**2 + math.cos(p1_rad.lat) * math.cos(p2_rad.lat) * math.sin(dlon/2)**2
        return np.float32(self.radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))


class CoordinateMapper:
    def __init__(self, altitude_data: List[Coordinate], 
                 distance_calculator: DistanceCalculator, 
                 use_kdtree: bool = config.KD_TREE_ENABLED):
        self.altitude_data = altitude_data
        self.distance_calculator = distance_calculator
        self.kdtree = self._build_kdtree() if use_kdtree else None

    def _build_kdtree(self) -> cKDTree:
        try:
            points = np.array([[c.point.lat, c.point.lon] for c in self.altitude_data], dtype=np.float64)
            
            return cKDTree(points)
        
        except Exception as e:
            raise GeoProcessingError('KDTree initialization failed') from e

    def find_closest(self, target: Point) -> Optional[Coordinate]:
        if self.kdtree:
            _, idx = self.kdtree.query([target.lat, target.lon])
            
            return self.altitude_data[idx]
        
        closest = min(
            self.altitude_data,
            key=lambda c: self.distance_calculator.calculate(target, c.point),
            default=None
        )
        
        return closest
    
    def map_elevations(self, targets: List[Coordinate], 
                       use_process_map: bool = True) -> List[np.float32]:
        mapper = partial(self._single_point_mapper, self.find_closest)
        executor = process_map if use_process_map else thread_map
        results = executor(mapper, [t.point for t in targets], chunksize=50)
        
        return [r.altitude if r else np.nan for r in results]

    @staticmethod
    def _single_point_mapper(func, point: Point) -> Optional[Coordinate]:
        return func(point)


def load_geo_elev_model(json_path: str) -> List[Coordinate]:
    ft_to_m = 0.3048
    schema = {'lat': pl.Float32, 'lon': pl.Float32, 'elev': pl.Float32}
    df = pl.read_json(json_path, schema=schema)
    df = df.with_columns(pl.col('elev').fill_null(0.0))
    coords = []
    for row in df.to_dicts():
        coords.append(
            Coordinate(
                point=Point(
                    lat=row['lat'],
                    lon=row['lon'],
                ),
                altitude=row['elev'] * ft_to_m,
            )
        )

    return coords

def merge_altitudes(original_coords: List[Coordinate], 
                    altitudes: List[np.float32]) -> List[Coordinate]:
    updated = []
    for old_coord, new_alt in zip(original_coords, altitudes):
        updated.append(
            Coordinate(
                point=Point(
                    lat=old_coord.point.lat,
                    lon=old_coord.point.lon,
                ),
                altitude=new_alt,
            )
        )

    return updated

def calculate_elevation_gain_pipeline(df: pl.DataFrame, window_size=50,  
        threshold=config.ELEVATION_MISSING_THRESHOLD) -> float:
    if 'alt_m' not in df.columns:
        return 0.0

    df_smoothed = df.with_columns(pl.col('alt_m').rolling_mean(window_size).alias('alt_m_smooth'))
    alt_list = df_smoothed['alt_m_smooth'].to_list()
    total_gain = 0.0
    for i in range(1, len(alt_list)):
        delta = alt_list[i] - alt_list[i - 1]
        if delta > threshold:
            total_gain += delta

    return total_gain

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

def run_altitude_mapping(vehicle_df: pl.DataFrame, lat_col: str, lon_col: str,
                         alt_col: str, geo_elev_model_path: str, use_process_map: bool,
                         output_path: str) -> pl.DataFrame:
    required = {lat_col, lon_col, alt_col}
    missing = required - set(vehicle_df.columns)
    if missing:
        print(f'[Warning] Missing coordinates for altitude mapping: {missing}')

        return vehicle_df

    geo_elev_model_coord = load_geo_elev_model(geo_elev_model_path)

    coords = vehicle_geodetic_to_coord(vehicle_df)
    
    mapper = CoordinateMapper(altitude_data=geo_elev_model_coord,
                              distance_calculator=HaversineDistance(),
                              use_kdtree=True)
    
    mapped_elevs = mapper.map_elevations(coords, use_process_map=use_process_map)
    
    if len(mapped_elevs) > 1 and abs(mapped_elevs[0] - mapped_elevs[1]) > 10:
        mapped_elevs[0] = mapped_elevs[1]
          
    updated_coords = merge_altitudes(coords, mapped_elevs)
    
    updated_vehicle_df = ds.update_dataset(vehicle_df, updated_coords, alt_col)
    
    return updated_vehicle_df.fill_nan(None)

def process_for_grade(df: pl.DataFrame) -> pl.DataFrame:
    required_cols = {'lat_deg', 'lon_deg', 'alt_m'}
    if not required_cols.issubset(df.columns):
        raise DataValidationError(f'Missing required columns: {required_cols - set(df.columns)}')

    df = df.with_columns([
        pl.col('lat_deg').shift(1).alias('_prev_lat'),
        pl.col('lon_deg').shift(1).alias('_prev_lon'),
        pl.col('alt_m').shift(1).alias('_prev_alt'),
    ])

    hav = HaversineDistance(radius=config.EARTH_RADIUS)
    
    df = df.with_columns(
        pl.struct(['_prev_lat', '_prev_lon', 'lat_deg', 'lon_deg'])
        .map_elements(
            lambda x: hav.calculate(
                Point(lat=x['_prev_lat'], lon=x['_prev_lon']),
                Point(lat=x['lat_deg'], lon=x['lon_deg'])
            ) if (x['_prev_lat'] is not None and x['_prev_lon'] is not None) else None,
            return_dtype=pl.Float64
        )
        .alias('_dist_m')
    )

    df = df.with_columns([
        ((pl.col('alt_m') - pl.col('_prev_alt')) / 
         pl.col('_dist_m').replace(0, None) * 100
        ).alias('grade_pct')
    ]).drop(['_prev_lat', '_prev_lon', '_prev_alt', '_dist_m'])
    
    return df.with_columns([
        pl.when(pl.col('grade_pct').abs() > 100)
          .then(None)
          .otherwise(pl.col('grade_pct'))
          .alias('grade_pct')
    ]).drop_nulls(subset=['grade_pct'])


if __name__ == '__main__':
    base_path = './data/vehicle_v2'

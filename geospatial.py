# geospatial.py V2.0 (TESTING)
import math
import logging
from typing import List, Optional
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from tqdm.contrib.concurrent import process_map, thread_map

from config import config, Point, Coordinate


logger = logging.getLogger(__name__)


class GeoProcessingError(Exception):
    """Base exception for geospatial processing errors"""


class DataValidationError(GeoProcessingError):
    """Raised when input data fails validation checks"""


class DistanceCalculator(ABC):
    @abstractmethod
    def calculate(self, p1: Point, p2: Point) -> float:
        """Calculate distance between two geographic points in meters"""


class HaversineCalculator(DistanceCalculator):
    def __init__(self, radius: float = config.EARTH_RADIUS):
        self.radius = radius

    def calculate(self, p1: Point, p2: Point) -> float:
        """Calculate great-circle distance using Haversine formula"""
        lat1 = math.radians(p1.lat) if p1.deg else p1.lat
        lon1 = math.radians(p1.lon) if p1.deg else p1.lon
        lat2 = math.radians(p2.lat) if p2.deg else p2.lat
        lon2 = math.radians(p2.lon) if p2.deg else p2.lon

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (math.sin(dlat/2)**2 
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        return self.radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class ElevationMapper:
    def __init__(self, reference_data: List[Coordinate]):
        self.reference = reference_data
        self.kdtree = self._build_kdtree() if config.KD_TREE_ENABLED else None
        self.distance_calculator = HaversineCalculator()

    def _build_kdtree(self) -> cKDTree:
        """Build KDTree for fast spatial queries"""
        try:
            points = np.array([(c.point.lat, c.point.lon) 
                              for c in self.reference], dtype=np.float64)
            return cKDTree(points)
        except Exception as e:
            logger.error('KDTree construction failed: %s', e)
            raise GeoProcessingError('Failed to build KDTree') from e

    def map_elevations(self, targets: List[Coordinate], 
                      parallel: bool = True) -> List[Optional[float]]:
        """Map elevations to target coordinates using nearest neighbor search"""
        if self.kdtree:
            return self._kdtree_search(targets)
        return self._linear_search(targets, parallel)

    def _kdtree_search(self, targets: List[Coordinate]) -> List[Optional[float]]:
        """KDTree-based elevation mapping"""
        try:
            _, indices = self.kdtree.query(
                [(t.point.lat, t.point.lon) for t in targets]
            )
            return [self.reference[i].altitude if i < len(self.reference) else None 
                    for i in indices]
        except Exception as e:
            logger.error('KDTree query failed: %s', e)
            return [None] * len(targets)

    def _linear_search(self, targets: List[Coordinate],
                      parallel: bool) -> List[Optional[float]]:
        """Brute-force search with optional parallelization"""
        mapper = partial(self._find_nearest_sequential)
        executor = process_map if parallel else thread_map
        return executor(mapper, targets, chunksize=config.CHUNK_SIZE)

    def _find_nearest_sequential(self, target: Coordinate) -> Optional[float]:
        """Find nearest reference point for a single target"""
        try:
            closest = min(
                self.reference,
                key=lambda ref: self.distance_calculator.calculate(
                    target.point, ref.point
                )
            )
            return closest.altitude
        except ValueError:
            return None


def load_elevation_model(path: str) -> List[Coordinate]:
    """Load elevation model from JSON file"""
    try:
        df = pl.read_json(path, schema={'lat': pl.Float32, 'lon': pl.Float32, 'elev': pl.Float32})
        df = df.with_columns(pl.col('elev').fill_null(0.0))
        return [Coordinate(
            point=Point(lat=row['lat'], lon=row['lon']),
            altitude=row['elev']*0.3048  # Convert ft. to meters
            )
            for row in df.to_dicts()]
    except Exception as e:
        logger.error('Failed to load elevation model: %s', e)
        raise GeoProcessingError('Elevation model loading failed') from e

def calculate_grade(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate road grade percentage between consecutive points"""
    required = {'lat_deg', 'lon_deg', 'alt_m'}
    if missing := required - set(df.columns):
        raise DataValidationError(f'Missing required columns: {missing}')
    calculator = HaversineCalculator()
    return df.with_columns([
        pl.col('lat_deg').shift().alias('_prev_lat'),
        pl.col('lon_deg').shift().alias('_prev_lon'),
        pl.col('alt_m').shift().alias('_prev_alt')
    ]).with_columns([
        pl.struct(['_prev_lat', '_prev_lon', 'lat_deg', 'lon_deg'])
        .map_elements(
            lambda x: calculator.calculate(
                Point(x['_prev_lat'], x['_prev_lon']),
                Point(x['lat_deg'], x['lon_deg'])
            ) if x['_prev_lat'] is not None else None,
            return_dtype=pl.Float64
        ).alias('distance_m'),
        ((pl.col('alt_m') - pl.col('_prev_alt')) / 
         pl.col('distance_m').replace(0, None) * 100
        ).alias('grade_pct')
    ]).drop(['_prev_lat', '_prev_lon', '_prev_alt', 'distance_m'])

def process_geospatial_data(df: pl.DataFrame, 
                            elevation_model_path: str) -> pl.DataFrame:
    """Full geospatial processing pipeline"""
    elevation_data = load_elevation_model(elevation_model_path)
    vehicle_coords = [
        Coordinate(Point(row['lat_deg'], row['lon_deg']),
                   row.get('alt_m', 0.0))
        for row in df.iter_rows(named=True)]
    mapper = ElevationMapper(elevation_data)
    elevations = mapper.map_elevations(vehicle_coords)
    
    return df.with_columns(
        pl.Series('alt_m', elevations).fill_null(0.0))

# geospatial.py V1.0 (NOT USING CURRENTLY)
# class GeoProcessingError(Exception):
#     """Base exception for geospatial processing errors"""

# class DataValidationError(GeoProcessingError):

# class DistanceCalculator(ABC):
#     @abstractmethod
#     def calculate(self, p1: Point, p2: Point) -> np.float32:
#         """Computes distance between two points"""


# class HaversineDistance(DistanceCalculator):
#     def __init__(self, radius: float = config.EARTH_RADIUS):
#         self.radius = radius

#     def _to_radians(self, point: Point) -> Point:
#         return Point(
#             lat=np.radians(point.lat),
#             lon=np.radians(point.lon),
#             deg=False
#         ) if point.deg else point

#     def calculate(self, p1: Point, p2: Point) -> np.float32:
#         p1_rad = self._to_radians(p1)
#         p2_rad = self._to_radians(p2)
#         dlat = p2_rad.lat - p1_rad.lat
#         dlon = p2_rad.lon - p1_rad.lon
#         a = math.sin(dlat/2)**2 + math.cos(p1_rad.lat) * math.cos(p2_rad.lat) * math.sin(dlon/2)**2
#         return np.float32(self.radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))


# class CoordinateMapper:
#     def __init__(self, altitude_data: List[Coordinate], 
#                  distance_calculator: DistanceCalculator, 
#                  use_kdtree: bool = config.KD_TREE_ENABLED):
#         self.altitude_data = altitude_data
#         self.distance_calculator = distance_calculator
#         self.kdtree = self._build_kdtree() if use_kdtree else None

#     def _build_kdtree(self) -> cKDTree:
#         try:
#             points = np.array([[c.point.lat, c.point.lon] for c in self.altitude_data], dtype=np.float64)
            
#             return cKDTree(points)
        
#         except Exception as e:
#             raise GeoProcessingError('KDTree initialization failed') from e

#     def find_closest(self, target: Point) -> Optional[Coordinate]:
#         if self.kdtree:
#             _, idx = self.kdtree.query([target.lat, target.lon])
            
#             return self.altitude_data[idx]
        
#         closest = min(
#             self.altitude_data,
#             key=lambda c: self.distance_calculator.calculate(target, c.point),
#             default=None
#         )
        
#         return closest
    
#     def map_elevations(self, targets: List[Coordinate], 
#                        use_process_map: bool = True) -> List[np.float32]:
#         mapper = partial(self._single_point_mapper, self.find_closest)
#         executor = process_map if use_process_map else thread_map
#         results = executor(mapper, [t.point for t in targets], chunksize=50)
        
#         return [r.altitude if r else np.nan for r in results]

#     @staticmethod
#     def _single_point_mapper(func, point: Point) -> Optional[Coordinate]:
#         return func(point)


# def load_geo_elev_model(json_path: str) -> List[Coordinate]:
#     ft_to_m = 0.3048
#     schema = {'lat': pl.Float32, 'lon': pl.Float32, 'elev': pl.Float32}
#     df = pl.read_json(json_path, schema=schema)
#     df = df.with_columns(pl.col('elev').fill_null(0.0))
#     coords = []
#     for row in df.to_dicts():
#         coords.append(
#             Coordinate(
#                 point=Point(
#                     lat=row['lat'],
#                     lon=row['lon'],
#                 ),
#                 altitude=row['elev'] * ft_to_m,
#             )
#         )

#     return coords

# def merge_altitudes(original_coords: List[Coordinate], 
#                     altitudes: List[np.float32]) -> List[Coordinate]:
#     updated = []
#     for old_coord, new_alt in zip(original_coords, altitudes):
#         updated.append(
#             Coordinate(
#                 point=Point(
#                     lat=old_coord.point.lat,
#                     lon=old_coord.point.lon,
#                 ),
#                 altitude=new_alt,
#             )
#         )

#     return updated

# def calculate_elevation_gain_pipeline(df: pl.DataFrame, window_size=50,  
#         threshold=config.ELEVATION_MISSING_THRESHOLD) -> float:
#     if 'alt_m' not in df.columns:
#         return 0.0

#     df_smoothed = df.with_columns(pl.col('alt_m').rolling_mean(window_size).alias('alt_m_smooth'))
#     alt_list = df_smoothed['alt_m_smooth'].to_list()
#     total_gain = 0.0
#     for i in range(1, len(alt_list)):
#         delta = alt_list[i] - alt_list[i - 1]
#         if delta > threshold:
#             total_gain += delta

#     return total_gain

# def vehicle_geodetic_to_coord(df: pl.DataFrame) -> List[Coordinate]:
#     coords = []
#     for row in df.iter_rows(named=True):
#         try:
#             lat = float(row['lat_deg'])
#             lon = float(row['lon_deg'])
#             alt = float(row['alt_m']) if row.get('alt_m') is not None else 0.0
#             coords.append(Coordinate(point=Point(lat=lat, lon=lon), altitude=alt))
#         except Exception:
#             continue

#     return coords

# def run_altitude_mapping(vehicle_df: pl.DataFrame, lat_col: str, lon_col: str,
#                          alt_col: str, geo_elev_model_path: str, use_process_map: bool,
#                          output_path: str) -> pl.DataFrame:
#     required = {lat_col, lon_col, alt_col}
#     missing = required - set(vehicle_df.columns)
#     if missing:
#         print(f'[Warning] Missing coordinates for altitude mapping: {missing}')

#         return vehicle_df

#     geo_elev_model_coord = load_geo_elev_model(geo_elev_model_path)

#     coords = vehicle_geodetic_to_coord(vehicle_df)
    
#     mapper = CoordinateMapper(altitude_data=geo_elev_model_coord,
#                               distance_calculator=HaversineDistance(),
#                               use_kdtree=True)
    
#     mapped_elevs = mapper.map_elevations(coords, use_process_map=use_process_map)
    
#     if len(mapped_elevs) > 1 and abs(mapped_elevs[0] - mapped_elevs[1]) > 10:
#         mapped_elevs[0] = mapped_elevs[1]
          
#     updated_coords = merge_altitudes(coords, mapped_elevs)
    
#     updated_vehicle_df = ds.update_dataset(vehicle_df, updated_coords, alt_col)
    
#     return updated_vehicle_df.fill_nan(None)

# def process_for_grade(df: pl.DataFrame) -> pl.DataFrame:
#     required_cols = {'lat_deg', 'lon_deg', 'alt_m'}
#     if not required_cols.issubset(df.columns):
#         raise DataValidationError(f'Missing required columns: {required_cols - set(df.columns)}')

#     df = df.with_columns([
#         pl.col('lat_deg').shift(1).alias('_prev_lat'),
#         pl.col('lon_deg').shift(1).alias('_prev_lon'),
#         pl.col('alt_m').shift(1).alias('_prev_alt'),
#     ])

#     hav = HaversineDistance(radius=config.EARTH_RADIUS)
    
#     df = df.with_columns(
#         pl.struct(['_prev_lat', '_prev_lon', 'lat_deg', 'lon_deg'])
#         .map_elements(
#             lambda x: hav.calculate(
#                 Point(lat=x['_prev_lat'], lon=x['_prev_lon']),
#                 Point(lat=x['lat_deg'], lon=x['lon_deg'])
#             ) if (x['_prev_lat'] is not None and x['_prev_lon'] is not None) else None,
#             return_dtype=pl.Float64
#         )
#         .alias('_dist_m')
#     )

#     df = df.with_columns([
#         ((pl.col('alt_m') - pl.col('_prev_alt')) / 
#          pl.col('_dist_m').replace(0, None) * 100
#         ).alias('grade_pct')
#     ]).drop(['_prev_lat', '_prev_lon', '_prev_alt', '_dist_m'])
    
#     return df.with_columns([
#         pl.when(pl.col('grade_pct').abs() > 100)
#           .then(None)
#           .otherwise(pl.col('grade_pct'))
#           .alias('grade_pct')
#     ]).drop_nulls(subset=['grade_pct'])


# if __name__ == '__main__':
#     base_path = './data/vehicle_v2'

import logging
import math
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from tqdm.contrib.concurrent import thread_map, process_map
import numpy as np
import polars as pl

from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Point:
    lat: np.float32
    lon: np.float32
    deg: bool = True


@dataclass
class Coordinate:
    point: Point
    altitude: np.float32


def load_geo_elev_model(json_path: str) -> list[Coordinate]:
    ft_to_m = pl.Float32(0.3048)  # foot -> meter
    with open(json_path, 'r') as f:
        data = json.load(f)
    elevation_profile = []
    for item in data:
        elevation_profile.append(
            Coordinate(
                point=Point(
                    lat=np.float32(item['lat']),
                    lon=np.float32(item['lon'])
                ),
                altitude=np.float32(item['elev'] * ft_to_m)
            )
        )
    return elevation_profile


def load_geo_elev_model_v2(json_path: str) -> List[Coordinate]:
    ft_to_m = 0.3048
    schema = {'lat': pl.Float32, 'lon': pl.Float32, 'elev': pl.Float32}
    df = pl.read_json(json_path, schema=schema)
    df = df.with_columns(pl.col('elev').fill_null(0.0))
    coord_list = []
    for row in df.to_dicts():
        coord_list.append(
            Coordinate(
                point=Point(
                    lat=row['lat'],
                    lon=row['lon']
                ),
                altitude=row['elev']*ft_to_m
            )
        )
    return coord_list


class DistanceCalculator(ABC):
    """Abstract base class for calculating distances."""
    @abstractmethod
    def calculate(self, p1: Point, p2: Point) -> np.float32:
        pass


class HaversineDistance(DistanceCalculator):
    """
    Haversine formula:  great-circle distances computation.
     
    Assumes:
        lat/lon in degrees,
        Earths radius is ~6367.0 km.
    """
    def __init__(self, radius: np.float32 = 6367.0):
        self.radius = radius

    def point_radian(self, point: Point) -> Point:
        if point.deg:
            return Point(lat=np.radians(point.lat),lon=np.radians(point.lon),deg=False)
        
        return point

    def calculate(self, p1: Point, p2: Point) -> np.float32:
        p1 = self.point_radian(p1)
        p2 = self.point_radian(p2)
        dlat = p2.lat - p1.lat
        dlon = p2.lon - p1.lon
        a = math.sin(dlat / 2)**2 + \
            math.cos(p1.lat) * math.cos(p2.lat) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return np.float32(self.radius * c)


class Euclidean(DistanceCalculator):
    def calculate(self, p1: Point, p2: Point) -> np.float32:
        return np.sqrt((p2.lat - p1.lat)**2 + (p2.lon - p1.lon)**2)


class CoordinateMapper:
    """Maps vehicle coordinates to the closest altitude point"""
    def __init__(self,altitude_data: List[Coordinate],
        distance_calculator: DistanceCalculator,use_kdtree: bool = False):
        self.altitude_data = altitude_data
        self.distance_calculator = distance_calculator
        self.use_kdtree = use_kdtree
        if use_kdtree:
            arr = []
            for coord in altitude_data:
                arr.append([float(coord.point.lat), float(coord.point.lon)])
            arr = np.array(arr, dtype=np.float64)  # shape (N, 2)
            self._kdtree = cKDTree(arr)
        else:
            self._kdtree = None

    def find_closest_point(self, target: Point) -> Coordinate:
        if self._kdtree is not None:
            dist, idx = self._kdtree.query([float(target.lat), float(target.lon)])
            return self.altitude_data[idx]
        else:
            closest_coord = None
            min_distance = np.float32(np.inf)
            for coord in self.altitude_data:
                distance = self.distance_calculator.calculate(target, coord.point)
                if distance < min_distance:
                    min_distance = distance
                    closest_coord = coord
            return closest_coord

    def map_elevations(self, vehicle_data: List[Coordinate]) -> List[np.float32]:
        mapped_elevations = []
        for vehicle_coord in vehicle_data:
            closest_coord = self.find_closest_point(vehicle_coord.point)
            if closest_coord:
                mapped_elevations.append(closest_coord.altitude)
            else:
                mapped_elevations.append(np.nan)
                logging.warning(f'Could not map elevation for vehicle coordinate {vehicle_coord}')
        
        return mapped_elevations

    def map_elevations_v2(self,vehicle_data: List[Coordinate],
        use_process_map: bool = False) -> List[np.float32]:
        points = [v.point for v in vehicle_data]
        if use_process_map:
            logging.info('Using process_map for mapping with possible KD-tree...')
            closest_coords = process_map(self.find_closest_point, points, chunksize=100)
        else:
            logging.info('Using thread_map for mapping with possible KD-tree...')
            closest_coords = thread_map(self.find_closest_point, points, chunksize=100)
        missing_count = sum(1 for coord in closest_coords if coord is None)
        logging.info(f'Total missing closest points: {missing_count}')

        return [coord.altitude if coord else np.nan for coord in closest_coords]


def update_vehicle_data_with_elevation(vehicle_data: List[Coordinate], 
    mapped_elevations: List[np.float32]) -> List[Coordinate]:
    if len(vehicle_data) != len(mapped_elevations):
        logging.error('Mismatch between vehicle data and mapped elevations')
        raise ValueError('Mismatch between vehicle data and mapped elevations.')
    updated_data = []
    for coord, alt in zip(vehicle_data, mapped_elevations):
        updated_data.append(
            Coordinate(
                point=coord.point,
                altitude=alt
            )
        )
    return updated_data


def merge_altitudes(
    original_coords: List[Coordinate],
    altitudes: List[float]
) -> List[Coordinate]:
    updated = []
    for old_coord, new_alt in zip(original_coords, altitudes):
        updated.append(
            Coordinate(
                point=Point(
                    lat=old_coord.point.lat,
                    lon=old_coord.point.lon
                ),
                altitude=new_alt
            )
        )
    return updated


def filter_trackpoints(lat: list[float],lon: list[float],alt: list[float],
    time_s: list[float],speed_m_s: list[float],dist_thresh: float = 5.0,
    elev_thresh: float = 0.0):
    if not (len(lat) == len(lon) == len(alt) == len(time_s) == len(speed_m_s)):
        raise ValueError('All input lists must have same length.')
    f_lat = [lat[0]]
    f_lon = [lon[0]]
    f_alt = [alt[0]]
    f_time = [time_s[0]]
    f_speed = [speed_m_s[0]]

    for i in range(1, len(lat)):
        dist_m = HaversineDistance(radius=6371000.0).calculate(
            Point(lat=np.float32(f_lat[-1]), lon=np.float32(f_lon[-1])),
            Point(lat=np.float32(lat[i]),     lon=np.float32(lon[i])))
        d_alt = abs(alt[i] - f_alt[-1])
        if dist_m < dist_thresh and d_alt < elev_thresh:
            continue
        f_lat.append(lat[i])
        f_lon.append(lon[i])
        f_alt.append(alt[i])
        f_time.append(time_s[i])
        f_speed.append(speed_m_s[i])

    return f_lat, f_lon, f_alt, f_time, f_speed


def calculate_elevation_gain_pipeline(df: pl.DataFrame,window_size: int = 5,
        threshold: float = 1.0) -> float:
    df_smoothed = df.with_columns(pl.col('alt_m').rolling_mean(window_size).alias('alt_m_smooth'))
    alt_list = df_smoothed['alt_m_smooth'].to_list()
    total_gain = 0.0
    for i in range(1, len(alt_list)):
        delta = alt_list[i] - alt_list[i - 1]
        if delta > threshold:
            total_gain += delta

    return total_gain

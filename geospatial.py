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



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    """Loads dataset taken from DEM (digital elevation model) database."""
    ft_to_m=pl.Float32(0.3048)  # Scaler for converting foot to meters.
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
                altitude=np.float32(item['elev']*ft_to_m)
            )
        )
    return elevation_profile

def load_geo_elev_model_v2(json_path: str) -> List[Coordinate]:
    """Loads JSON data using polars and returns List[Coordinate]."""
    # ft_to_m = np.float32(0.3048)
    ft_to_m=0.3048
    schema={'lat': pl.Float32,'lon': pl.Float32,'elev': pl.Float32}
    df=pl.read_json(json_path, schema=schema)
    df=df.with_columns(pl.col('elev').fill_null(0.0))
    coord_list = []
    for row in df.to_dicts():
        coord_list.append(
            Coordinate(
                point=Point(
                    lat=row['lat'],
                    lon=row['lon']
                ),
                altitude=row["elev"]*ft_to_m
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
    Calculates great-circle distances using the Haversine formula.
    Assumes lat/lon in degrees, Earth radius ~6367.0 km by default.
    """
    def __init__(self, radius: np.float32 = 6367.0):
        self.radius = radius

    def point_radian(self, point: Point) -> Point:
        """Converts lat/lon from degrees to radians if deg=True."""
        if point.deg:
            return Point(
                lat=np.radians(point.lat),
                lon=np.radians(point.lon),
                deg=False
            )
        return point

    def calculate(self, p1: Point, p2: Point) -> np.float32:
        p1 = self.point_radian(p1)  # Convert (Lat, Lon) to Radians
        p2 = self.point_radian(p2)
        dlat = p2.lat - p1.lat
        dlon = p2.lon - p1.lon
        a=math.sin(dlat/2)**2 + \
            math.cos(p1.lat)*math.cos(p2.lat)*math.sin(dlon/2)**2
        c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return np.float32(self.radius*c)


class Euclidean(DistanceCalculator):
    """Calculates planar Euclidean distance."""
    def calculate(self, p1: Point, p2: Point) -> np.float32:
        return np.sqrt((p2.lat-p1.lat)**2 + (p2.lon-p1.lon)**2)
    

class CoordinateMapper:
    """
    Maps vehicle coordinates to the closest altitude point using a specified
    distance calculation (e.g., Haversine or Euclidean).
    """
    def __init__(self, altitude_data: List[Coordinate], distance_calculator: DistanceCalculator):
        """
        args:
            altitude_data: A list of known Coordinates with altitudes.
            distance_calculator: An instance of DistanceCalculator to measure distances.
        """        
        self.altitude_data = altitude_data
        self.distance_calculator = distance_calculator

    def find_closest_point(self, target: Point) -> Coordinate:
        """
        Finds the single closest altitude Coordinate to the given target point.
        
        args:
            target: The point for which we want the nearest altitude.
        
        returns:
            The Coordinate (point + altitude) that is closest to `target`.
        """
        closest_coord = None
        min_distance = np.float32(np.inf)

        for coord in self.altitude_data:
            distance = self.distance_calculator.calculate(target, coord.point)
            if distance < min_distance:
                min_distance = distance
                closest_coord = coord

        return closest_coord
    
    def map_elevations(self, vehicle_data: List[Coordinate]) -> List[np.float32]:
        """
        Maps a list of vehicle Coordinates to their closest altitudes.

        args:
            vehicle_data: A list of vehicle coordinates to map.

        returns:
            A list of altitudes (floats), each matching the closest point in altitude_data.
        """
        mapped_elevations = []
        for vehicle_coord in vehicle_data:
            closest_coord = self.find_closest_point(vehicle_coord.point)
            if closest_coord:
                mapped_elevations.append(closest_coord.altitude)
            else:
                mapped_elevations.append(np.nan)
                logging.warning(f'Could not map elevation for vehicle coordinate {vehicle_coord}')
        
        return mapped_elevations
    
    def map_elevations_v2(self, vehicle_data: List[Coordinate], 
                          use_process_map: bool = False) -> List[np.float32]:
        """
        Maps a list of vehicle Coordinates to their closest altitudes, 
        optionally using multi-processing (process_map) for speed.
    
        args:
            vehicle_data: List of vehicle coordinates.
            use_process_map: If True, use process_map; else use thread_map.
        
        returns: A list of altitudes (floats).
        """
        points = [v.point for v in vehicle_data]
        if use_process_map:
            logging.info('Using process_map for mapping...')
            closest_coords = process_map(self.find_closest_point, points,
                                         chunksize=100)
        else:
            logging.info('Using thread_map for mapping...')
            closest_coords = thread_map(self.find_closest_point, points,
                                        chunksize=100)

        missing_count = sum(1 for coord in closest_coords if coord is None)
        logging.info(f'Total missing closest points: {missing_count}')
          
        return [
            coord.altitude if coord else np.nan 
            for coord in closest_coords
        ]
    

def update_vehicle_data_with_elevation(vehicle_data: List[Coordinate],
                                       mapped_elevations: List[np.float32]) -> List[Coordinate]:
    """
    Updates a list of Coordinates by replacing each altitude with the mapped 
    altitude.
    
    Args:
        vehicle_data: Original list of Coordinates.
        mapped_elevations: A list of floats (altitudes) of the same length.

    Returns:
        A new list of Coordinates with updated altitudes.
    """
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
    """
    Merges new altitude values with existing Coordinates, preserving lat/lon.
    
    Args:
        original_coords: The original list of Coordinates.
        altitudes: A list of new altitude values (in the same order).
    
    Returns:
        A new list of Coordinates with updated altitudes.
    """
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

def calculate_elevation_gain_pipeline(
    df: pl.DataFrame,
    window_size: int = 5,
    threshold: float = 1.0
) -> float:
    """
    source: https://www.gpsvisualizer.com/tutorials/elevation_gain.html
    
    A pipeline function that:
    Smooths data with a rolling mean, extracts altitude data, and computes
    total elevation gain above a threshold.

    Args:
        df: Polars DataFrame with 'alt_m' column (time-ordered or 
            distance-ordered).
        window_size: The window size for the moving average filter.
        threshold: Minimum altitude change to be counted as "gain".

    Returns:
        Total cumulative elevation gain (float).
        
    NOTE:  If the difference between consecutive points is smaller, its 
        assumed to be noise.  This parameterize view threshold.
        For example, threshold = 1.0 (meter or ft.) 
    """
    df_smoothed = df.with_columns(
        pl.col('alt_m').rolling_mean(window_size).alias('alt_m_smooth'))

    alt_list = df_smoothed['alt_m_smooth'].to_list()

    total_gain = 0.0
    for i in range(1, len(alt_list)):
        delta = alt_list[i] - alt_list[i-1]
        if delta > threshold:
            total_gain += delta

    return total_gain
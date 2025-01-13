import logging
import math
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
# from pathlib import Path
from typing import List, Union
from tqdm.contrib.concurrent import thread_map, process_map
import numpy as np
import polars as pl


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
    """Calculates great-circle distances using the Haversine formula."""
    def __init__(self, radius: np.float32 = 6367.0):
        self.radius = radius

    def point_radian(self, point: Point) -> Point:
        """Converts a dataclss Point of lat and lon to radians."""
        return Point(np.radians(point.lat), np.radians(point.lon))

    def calculate(self, p1: Point, p2: Point) -> np.float32:
        p1 = self.point_radian(p1)  # Convert (Lat, Lon) to Radians
        p2 = self.point_radian(p2)
        dlat = p2.lat - p1.lat
        dlon = p2.lon - p1.lon
        a=math.sin(dlat/2)**2 + \
            math.cos(p1.lat)*math.cos(p2.lat)*math.sin(dlon/2)**2
        c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return self.radius*c


class Euclidean(DistanceCalculator):
    """Calculates distance using simple Euclidean formula."""
    def calculate(self, p1: Point, p2: Point) -> np.float32:
        return np.sqrt((p2.lat-p1.lat)**2 + (p2.lon-p1.lon)**2)
    

class CoordinateMapper:
    """
    Maps vehicle coordinates to the closest altitude points from a dataset
    using a specified distance calculation algorithm.
    """
    def __init__(self, altitude_data: List[Coordinate], distance_calculator: DistanceCalculator):
        self.altitude_data = altitude_data
        self.distance_calculator = distance_calculator

    def find_closest_point(self, target: Point) -> Coordinate:
        """Custom implementation to find the closest point without lambda."""
        closest_coord = None
        min_distance = np.float32(np.inf)

        for coord in self.altitude_data:
            distance = self.distance_calculator.calculate(target, coord.point)
            if distance < min_distance:
                min_distance = distance
                closest_coord = coord

        return closest_coord
    
    # def map_elevations(self, vehicle_data: List[Coordinate]) -> List[np.float32]:
    #     """Maps vehicle coordinates to the closest elevation with a progress bar."""
    #     points = [v.point for v in vehicle_data]
    #     closest_coords = thread_map(self.find_closest_point, points, chunksize=100)
    #     return [coord.altitude if coord else np.nan for coord in closest_coords]
    

    def map_elevations(self, vehicle_data: List[Coordinate]) -> List[np.float32]:
        """Maps vehicle coordinates to the closest elevation."""
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
        Maps vehicle coordinates to the closest elevation.
        
        Args:
            vehicle_data: List of vehicle coordinates.
            use_process_map: If True, use `process_map`. Otherwise, use `thread_map`.
        """
        points = [v.point for v in vehicle_data]
        if use_process_map:
            logging.info('Using process_map for mapping...')
                                        # Vehicle Point(lat,lon), altitude)
            closest_coords = process_map(self.find_closest_point, points,
                                         chunksize=100)
        else:
            logging.info('Using thread_map for mapping...')
            closest_coords = thread_map(self.find_closest_point, points,
                                        chunksize=100)
        
        logging.info(f'Total missing closest points: {sum(1 for coord in closest_coords if coord is None)}')
        
        return [coord.altitude if coord else np.nan for coord in closest_coords]
    

def update_vehicle_data_with_elevation(vehicle_data: List[Coordinate], 
                                       mapped_elevations: List[np.float32]) -> List[Coordinate]:
    """Updates vehicle data by replacing the altitude with mapped elevations."""
    if len(vehicle_data) != len(mapped_elevations):
        logging.error('Mismatch between vehicle data and mapped elevations')
        raise ValueError('Mismatch between vehicle data and mapped elevations.')


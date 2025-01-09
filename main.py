import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
# import polars as pl


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'vehicle' / 'geo' / ''


@dataclass
class Point:
    lat: np.float32
    lon: np.float32


@dataclass
class Coordinate:
    point: Point
    altitude: np.float32


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
        # deg_to_rad = math.pi / 180.0
        # dlat = (p2.lat - p1.lat) * deg_to_rad
        # dlon = (p2.lon - p1.lon) * deg_to_rad
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
        return np.sqrt((p2.lat - p1.lat) ** 2 + (p2.lon - p1.lon) ** 2)
    

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

    def map_elevations(self, vehicle_data: List[Coordinate]) -> List[np.float32]:
        """Maps vehicle coordinates to the closest elevation."""
        mapped_elevations = []
        for vehicle_coord in vehicle_data:
            closest_coord = self.find_closest_point(vehicle_coord.point)
            mapped_elevations.append(closest_coord.altitude)
        return mapped_elevations
    
def update_vehicle_data_with_elevation(vehicle_data: List[Coordinate], mapped_elevations: List[np.float32]) -> List[Coordinate]:
    """Updates vehicle data by replacing the altitude with mapped elevations."""
    updated_data = []
    for i, vehicle_coord in enumerate(vehicle_data):
        updated_data.append(Coordinate(vehicle_coord.point, mapped_elevations[i]))
    return updated_data


if __name__ == '__main__':
    # print(BASEDIR)
    # print(VEHICLE_DATA_DIR)
    # 'data/vehicle/2007/Dodge/RAM1500/01_ECM.csv'
    # print(BASEDIR.joinpath('vehicle/2007_Dodge_Ram'))
    # veh_07_dodge_ram1500 = DATA_DIR.joinpath('2007/Dodge/RAM1500/')
    
    # Example Usage:
    pass

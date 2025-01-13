import numpy as np
from typing import List
from scipy.spatial import cKDTree

from geospatial import Coordinate

class KDTreeCoordinateMapper:
    """
    Uses scipy's cKDTree to quickly find nearest altitudes for each vehicle coordinate.
    """
    def __init__(self, altitude_data: List[Coordinate]):
        """
        Builds the KD-tree from the altitude data (lat, lon).
        
        Args:
            altitude_data: A list of Coordinate objects.
        """
        self.altitude_data = altitude_data

        # Build an Nx2 array of [lat, lon]
        coords_array = np.array([
            [c.point.lat, c.point.lon]
            for c in altitude_data
        ], dtype=np.float64)

        self.kdtree = cKDTree(coords_array)

    def map_elevations(self, vehicle_coords: List[Coordinate]) -> List[float]:
        """
        Finds altitudes for each vehicle coordinate using the KD-tree nearest
        neighbor lookup.
        
        Args:
            vehicle_coords: List of Coordinates (point + altitude).
        
        Returns:
            List of floats corresponding to the altitude of the nearest 
            altitude_data point.
        """
        # Build Nx2 array for the vehicle coordinates
        query_points = np.array([
            [v.point.lat, v.point.lon]
            for v in vehicle_coords
        ], dtype=np.float64)

        # Query returns distances and indices (of altitude_data)
        distances, indices = self.kdtree.query(query_points)

        results = [self.altitude_data[idx].altitude for idx in indices]
        return results

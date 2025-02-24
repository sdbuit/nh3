"""
compute_vsp.py

Temporarily separation of ComputeVSP 
"""

import math
from typing import List, Dict, Tuple
from config import config
from config import Point

from geospatial import HaversineCalculator

class ComputeVSP:
    def __init__(self):
        self.haversine = HaversineCalculator()

    def compute_vsp(self, data: List[Dict]) -> List[float]:
        vsp_values = []
        prev_point = None
        for i, row in enumerate(data):
            try:
                speed = self._get_speed(row)
                acceleration = self._get_acceleration(data, i, speed)
                grade = self._get_grade(data, i, prev_point)
                # TODO Separate force components in the VSP formular (maybe)
                vsp = speed * (1.1 * acceleration + 9.81 * grade + 0.132) + 0.000302 * speed**3
                vsp_values.append(round(vsp, 2))
                prev_point = (row['lat_deg'], row['lon_deg'], row['alt_m'])
            except (KeyError, TypeError, ValueError):
                vsp_values.append(None)
        return vsp_values

    def _get_speed(self, row: Dict) -> float:
        if 'gps_speed_m_s' in row:
            return row['gps_speed_m_s']
        if 'speed_mph' in row:
            return row['speed_mph'] * 0.44704
        if 'speed_kmh' in row:
            return row['speed_kmh'] * 0.27778
        raise ValueError('No valid speed column found')

    def _get_acceleration(self, data: List[Dict], index: int, current_speed: float) -> float:
        if index == 0 or index >= len(data)-1:
            return 0.0
        dt = self._get_time_difference(data, index)
        if dt <= 0:
            return 0.0
        prev_speed = self._get_speed(data[index-1]) if index > 0 else 0.0
        next_speed = self._get_speed(data[index+1]) if index < len(data)-1 else 0.0

        return (next_speed - prev_speed) / (2 * dt)

    def _get_grade(self, data: List[Dict], index: int, prev_point: Tuple[float]) -> float:
        """
        Computes road grade using altitude from drive cycle elevation model. 
        For each test trial a GPS sensor equiped to the vehicle provided 
        coordinates that were used for obtaining elevation data from a third 
        party database.
        
        TODO
            1. IMMEDIATE ATTENTION: Grade may need some post-processing done on
               grade calculations -> refer interpolation methods.
            2. Compare drive cycle on-road grade results with other credible
               geographical databases.
        """
        if index == 0 or not prev_point:
            return 0.0
        current_alt = data[index]['alt_m']
        prev_alt = prev_point[2]
        delta_alt = current_alt - prev_alt
        current_point = (data[index]['lat_deg'], data[index]['lon_deg'])
        distance = self.haversine.calculate(
            Point(lat=prev_point[0], lon=prev_point[1]),
            Point(lat=current_point[0], lon=current_point[1]))
        # TODO Confirm return type by checking VSP data.  Percent grade
        # is required for the VSP formula.
        return delta_alt / distance if distance > 0 else 0.0

    def _get_time_difference(self, data: List[Dict], index: int) -> float:
        """Get delta t (dt) seconds"""
        # TODO Verify, visualize, validate the default 1 second.  One second
        # is a long time for a telemtry system collecting at high speed.
        if index == 0 or index >= len(data)-1:
            return 1.0  # TODO Default to 1 second to avoid division by zero.
        current_time = data[index]['elapse_sec']
        prev_time = data[index-1]['elapse_sec']
        next_time = data[index+1]['elapse_sec']
        return (next_time - prev_time) / 2

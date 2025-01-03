import os
import json
import dataclasses
from pathlib import Path
from dataclasses import dataclass
from typing import Type, TypeVar, Optional, Any

import polars as pl


@dataclass
class StoredDataType:
    name: str
    short_name: str
    unit: str

def snake_to_camel(s: str) -> str:
    """Convert snake_case to camelCase."""
    parts = s.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])

def camel_to_snake(s: str) -> str:
    """Convert camelCase to snake_case."""
    snake = ''
    for char in s:
        if char.isupper():
            snake += '_' + char.lower()
        else: snake += char
    return snake


T = TypeVar('T')
class Dataclass:
    def to_json(self, include_null: bool = False) -> dict:
        # return dataclasses.asdict(self, dict_factory=lambda fields: {
        #     snake_to_camel(key): value
        #     for (key, value) in fields
        #     if value is not None or include_null
        #     },
        # )
        def my_factory(fields):
            result = {}
            for (field_name, value) in fields:
                if value is None and not include_null:
                    continue
                if isinstance(value, StoredDataType):
                    result[value.short_name] = value.unit
                else:
                    result[snake_to_camel(field_name)] = value
            
            return result
            
        return dataclasses.asdict(self, dict_factory=my_factory)

    @classmethod
    def from_json(cls: Type[T], json_data: dict) -> T:
        """Constructs a dataclass instance from a JSON-like dictionary.
        
        Args:
            json_data:  A Dict JSON dictionary with camelCase keys.
        
        Returns:  
            T a new dataclass instance.
        
        Raises:  ValueError if `cls` is not a dataclass.
        """
        if not dataclasses.is_dataclass(cls):
            raise ValueError(f'{cls.__name__} must be a dataclass')

        field_names = {field.name for field in dataclasses.fields(cls)}
        kwargs = {
            camel_to_snake(key): value
            for key, value in json_data.items()
            if camel_to_snake(key) in field_names
        }
        return cls(**kwargs)

@dataclass
class GeodeticConfig(Dataclass):
    """
    Stores geodetic coordinates for a given point in the drive cycle route.
    """
    lat: float = 0.0
    lon: float = 0.0
    elev: float = 0.0


def load_geodetic_csv_to_list(csv_path: str) -> list[GeodeticConfig]:
    """
    Reads a CSV file containing known columns like [lat, lon, elev] and then 
    constructs a list of GeodeticConfig objects.
    """
    df = pl.read_csv(csv_path)
    # df = df.rename({'latitude': 'lat', 'longitude': 'lon', 'alt': 'elev'})
    geodetic_list = []
    # latitude,longitude,altitude (ft)
    for row in df.iter_rows(named=True):
        # row is a dict-like mapping: {'lat': x, 'lon': y, 'elev': z, ...}
        geodetic_list.append(GeodeticConfig(lat=row['latitude'], 
                                            lon=row['longitude'], 
                                            elev=row['altitude (ft)']))
    return geodetic_list
    

if __name__ == '__main__':
    # BASEDIR = os.path.abspath(os.path.dirname(__file__))
    # DATA_DIR = os.path.join(BASEDIR, 'data')
    # geodetic_data = os.path.join(DATA_DIR, 'geo', 'usu_drive_cycle_path_elev_profile.csv')
    BASEDIR = Path(__file__).resolve().parent
    DATA_DIR = BASEDIR / 'data' / 'geo'
    geodetic_csv_path = os.path.join(DATA_DIR,
                                 'usu_drive_cycle_path_elev_profile.csv')
    geodetic_data = load_geodetic_csv_to_list(geodetic_csv_path)
    json_list = [x.to_json() for x in geodetic_data]
    print(json.dumps(json_list, indent=2))
    # with open('drive_cycle_route.json', 'w') as f:
    #     json.dump(json_list, f, indent=2)
    
    
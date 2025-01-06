import os
import sys
import json
import shutil
import dataclasses
from pathlib import Path
from dataclasses import dataclass
from typing import Type, TypeVar, Optional, Any

import polars as pl
from docx import Document


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
class StoredDataType:
    name: str
    short_name: str
    unit: str


@dataclass
class GeodeticConfig(Dataclass):
    """
    Stores geodetic coordinates for a given point in the drive cycle route.
    """
    file_path: Optional[str] = None
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

def extract_docx_to_json(docx_path, json_path):
    """Extract table data from a .docx file and save it as a .json file."""
    try:
        doc = Document(docx_path)
        data = {}
        for table_index, table in enumerate(doc.tables):
            table_data = []
            for row_index, row in enumerate(table.rows):
                cells = [cell.text.strip() for cell in row.cells]
                if len(cells) == 2:
                    key, value = cells
                    data[key] = value
                else:
                    table_data.append(cells)
            if table_data:
                data[f'table_{table_index + 1}'] = table_data
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f'Extracted JSON: {json_path}')
    except Exception as e:
        print(f'Failed to extract data from {docx_path}: {e}')

def reorganize_vehicle_data(base_dir: str, output_dir: str):
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    if not base_path.exists():
        print(f'Base directory {base_path} does not exist.')
        return
    for vehicle_dir in base_path.iterdir():
        if not vehicle_dir.is_dir():
            continue
        parts = vehicle_dir.name.split('_')
        if len(parts) < 3:
            print(f'Skipping {vehicle_dir.name}: Unable to parse year, make, and model.')
            continue
        year = parts[1]
        make = parts[2]
        model = '_'.join(parts[3:]) if len(parts) > 3 else 'Unknown_Model'
        target_dir = output_path / year / make / model
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in vehicle_dir.iterdir():
            target_path = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target_path, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target_path)

                if item.suffix == '.docx':
                    json_output_path = target_path.with_suffix('.json')
                    extract_docx_to_json(target_path, json_output_path)

        print(f'Reorganized {vehicle_dir.name} -> {target_dir}')


if __name__ == '__main__':
    BASEDIR = Path(__file__).resolve().parent

    GEODETIC_DATA_DIR = BASEDIR / 'data' / 'geo'
    geodetic_csv_path = os.path.join(GEODETIC_DATA_DIR,
                                 'usu_drive_cycle_path_elev_profile.csv')
    geodetic_data = load_geodetic_csv_to_list(geodetic_csv_path)
    json_list = [x.to_json() for x in geodetic_data]
    print(json.dumps(json_list, indent=2))
    # with open('drive_cycle_route.json', 'w') as f:
    #     json.dump(json_list, f, indent=2)

    # BASE_VEHICLE_DIR = BASEDIR / 'data' / 'archive' / 'UWRL_ON_ROAD_Measurments' / 'UWRL_On_Road_Measurements_Gasoline'
    BASE_VEHICLE_DIR = BASEDIR / 'data' / 'archive' / 'UWRL_ON_ROAD_Measurments' / 'UWRL_On_Road_Measurements_Diesel'
    OUTPUT_DIR = BASEDIR / 'data' / 'vehicle'
    reorganize_vehicle_data(BASE_VEHICLE_DIR, OUTPUT_DIR)
    sys.exit()

import polars as pl
import sys

from typing import List
from geospatial import (CoordinateMapper, HaversineDistance, 
                        load_geo_elev_model_v2, merge_altitudes)
from dataset import (update_dataset, load_dataset, vehicle_geodetic_to_coord)


def run_mapping_cycle(
    label: str,
    mapper: CoordinateMapper,
    vehicle_coords: List,
    vehicle_df: pl.DataFrame,
    use_process_map: bool,
    output_path: str
) -> None:
    """
    Runs one full cycle of mapping altitudes to vehicle coords and writing 
    to CSV.
    """
    print(f"{label}")
    mapped_elevations = mapper.map_elevations_v2(
        vehicle_coords,
        use_process_map=use_process_map
    )
    updated_coords = merge_altitudes(vehicle_coords, mapped_elevations)
    updated_vehicle_data = update_dataset(
        vehicle_df,
        updated_coords,
        'lat_deg',
        'lon_deg',
        'alt_m'
    )

    updated_vehicle_data.write_csv(output_path)
    mode = "process_map" if use_process_map else "thread_map"
    print(f"Updated dataset saved with {mode} to {output_path}\n")

def main():
    vehicle_data_path =     '01_ECM.csv'
    geo_elev_model_path =   './data/geo/drive_cycle_route.json'
    output_path_process =   'process-updated_v1_01_ECM.csv'
    output_path_thread =    'thread-updated_v1_01_ECM.csv'
    
    vehicle_df = load_dataset()
    geo_elev_model_coord = load_geo_elev_model_v2(geo_elev_model_path)

    vehicle_coord = vehicle_geodetic_to_coord(vehicle_df.select(
        ['lat_deg', 'lon_deg', 'alt_m'])
    )
    
    mapper = CoordinateMapper(geo_elev_model_coord, HaversineDistance())
    
    print('Process Map Test')
    mapped_elevations_process = mapper.map_elevations_v2(vehicle_coord, 
                                                         use_process_map=True)
    updated_coords_process = merge_altitudes(vehicle_coord, 
                                             mapped_elevations_process)
    updated_vehicle_data = update_dataset(
        vehicle_df,
        updated_coords_process,
        'lat_deg',
        'lon_deg',
        'alt_m'
    )
    
    updated_vehicle_data.write_csv(output_path_process)
    print(f'Updated dataset saved with process_map to {output_path_process}')
    
    # NOTE: Process map is a lot faster compared to threading option.
    # print('Thread Map Test')
    # mapped_elevations_thread = mapper.map_elevations_v2(
    #     vehicle_coord,
    #     use_process_map=False
    # )
    # print('Thread Map Test:')
    # mapped_elevations_thread = mapper.map_elevations_v2(vehicle_coord,
    #                                                     use_process_map=False)
    # updated_coords_thread = merge_altitudes(vehicle_coord, mapped_elevations_thread)
    # updated_vehicle_data = update_dataset(
    #     vehicle_df,
    #     updated_coords_thread,
    #     'lat_deg',
    #     'lon_deg',
    #     'alt_m'
    # )
    # updated_vehicle_data.write_csv(output_path_thread)
    # print(f'Updated dataset saved with thread_map to {output_path_thread}')
    
    sys.exit(0)


if __name__ == '__main__':
    main()
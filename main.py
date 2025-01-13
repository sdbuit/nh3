import polars as pl
from geospatial import (CoordinateMapper, HaversineDistance,load_geo_elev_model_v2)
from dataset import update_dataset, load_dataset
import sys
import polars as pl
from geospatial import Coordinate, Point
import polars.selectors as cs

def vheicle_geodetic2coord(df: pl.DataFrame):
    vehicle_coord_list = []
    for row in df.to_dicts():
        vehicle_coord_list.append(
            Coordinate(
                point=Point(
                    lat=row["lat_deg"],
                    lon=row["lon_deg"]
                ),
                altitude=row["alt_m"]  # or 0 if missing
            )
        )
    return vehicle_coord_list

def main():
    vehicle_data_path =     '01_ECM.csv'
    geo_elev_model_path =   './data/geo/drive_cycle_route.json'
    output_path_process =   'process-updated_v1_01_ECM.csv'
    output_path_thread =    'thread-updated_v1_01_ECM.csv'
    
    vehicle_df = load_dataset()
    geo_elev_model_coord = load_geo_elev_model_v2(geo_elev_model_path)
    # Select vehicle coordinates: 'lat_deg', 'lon_deg', 'alt_m'
    # vehicle_coord = vehicle_df.select(['lat_deg', 'lon_deg', 'alt_m'])    
    # vehicle_coord = vheicle_geodetic2coord(vehicle_coord)
    vehicle_coord = vheicle_geodetic2coord(vehicle_df.select(
        ['lat_deg', 'lon_deg', 'alt_m'])
    )
        
    mapper = CoordinateMapper(geo_elev_model_coord, HaversineDistance())
    print('Process Map Test: 2')
    mapped_elevations_process = mapper.map_elevations_v2(vehicle_coord, 
                                                         use_process_map=True)
    updated_coords = []
    for old_coord, new_alt in zip(vehicle_coord, mapped_elevations_process):
        updated_coords.append(
            Coordinate(
                point=Point(
                    lat=old_coord.point.lat,
                    lon=old_coord.point.lon
                ),
                altitude=new_alt
            )
        )
    updated_vehicle_data = update_dataset(vehicle_df, updated_coords, 
                                          'lat_deg', 'lon_deg', 'alt_m')
    # alt_col='alt_m'
    # updated_vehicle_data = update_dataset(vehicle_df,
    #                                       mapped_elevations_process,
    #                                       'lat_deg', 'lon_deg', alt_col)
    # updated_vehicle_data  = updated_vehicle_data.with_columns(
    #     pl.Series(name='alt_m', values=mapped_elevations_process))
    updated_vehicle_data.write_csv(output_path_process)
    print(f'Updated dataset saved with process_map to {output_path_process}')
    
    # #########################################################################

    # Option 2: Using `thread_map`
    print('Thread Map Test:')
    mapped_elevations_thread = mapper.map_elevations_v2(vehicle_coord,
                                                        use_process_map=False)
    updated_coords = []
    for old_coord, new_alt in zip(vehicle_coord, mapped_elevations_thread):
        updated_coords.append(
            Coordinate(
                point=Point(
                    lat=old_coord.point.lat,
                    lon=old_coord.point.lon
                ),
                altitude=new_alt
            )
        )
    updated_vehicle_data = update_dataset(vehicle_df, updated_coords, 
                                          'lat_deg', 'lon_deg', 'alt_m')
    # updated_vehicle_data = updated_vehicle_data.with_columns(
    #     pl.Series(name=alt_col, values=mapped_elevations_thread))
    updated_vehicle_data.write_csv(output_path_thread)
    print(f'Updated dataset saved with thread_map to {output_path_thread}')

if __name__ == '__main__':
    main()
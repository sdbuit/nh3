import polars as pl
from geospatial import load_json_dem, CoordinateMapper, HaversineDistance
from dataset import extract_coordinates, update_dataset, load_dataset


def main():
    vehicle_data_path =     '01_ECM.csv'
    dem_data_path =         './data/geo/drive_cycle_route.json'
    output_path_process =   'process-updated_v1_01_ECM.csv'
    output_path_thread =    'thread-updated_v1_01_ECM.csv'
    alt_col = 'alt_m'

    vehicle_df = load_dataset()  # print(vehicle_data.columns)
    altitude_data = load_json_dem(dem_data_path)  # Load DEM data
    coordinates = extract_coordinates(vehicle_df, 'lat_deg', 'lon_deg', alt_col)
    mapper = CoordinateMapper(altitude_data, HaversineDistance())

    print('Process Map Test:2')
    mapped_elevations_process = mapper.map_elevations_v2(coordinates, 
                                                         use_process_map=True)
    updated_vehicle_data = update_dataset(vehicle_df, coordinates, 
                                          'lat_deg', 'lon_deg', alt_col)
    updated_vehicle_data  = updated_vehicle_data.with_columns(
        pl.Series(name=alt_col, values=mapped_elevations_process)
    )
    vehicle_df.write_csv(output_path_process)
    print(f'Updated dataset saved with process_map to {output_path_process}')

    # #########################################################################

    # Option 2: Using `thread_map`
    print('Thread Map Test:')
    mapped_elevations_thread = mapper.map_elevations_v2(coordinates,
                                                        use_process_map=False)
    updated_vehicle_data = updated_vehicle_data.with_columns(
        pl.Series(name=alt_col, values=mapped_elevations_thread))
    updated_vehicle_data.write_csv(output_path_thread)
    print(f'Updated dataset saved with thread_map to {output_path_thread}')


if __name__ == "__main__":
    main()

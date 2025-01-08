import os
import json
from docx import Document
from typing import Optional, Dict


def gen_hierarch_vehicle_dict(base_dir: Optional[str] = None) -> Dict:
    """Generates a hierarchical dictionary for vehicles."""
    if base_dir is None:
        base_dir = os.path.join('data', 'vehicle_v2')
    vehicle_data = {}
    for year in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year)
        if os.path.isdir(year_path):
            vehicle_data[year] = {}
            for make in os.listdir(year_path):
                make_path = os.path.join(year_path, make)
                if os.path.isdir(make_path):
                    vehicle_data[year][make] = {}
                    for model in os.listdir(make_path):
                        model_path = os.path.join(make_path, model)
                        if os.path.isdir(model_path):
                            vehicle_data[year][make][model] = model_path
    return vehicle_data


def vehicle_test_doc_generate(doc_input_path: str = 'info_sheet_test.docx',
                              output_file: str = 'veh_test_doc.json') -> None:
    """Parses a Word document and generates a JSON file."""
    doc = Document(doc_input_path)
    data = {
        'General Information': {},
        'Test Information': {},
        'Vehicle Information': {},
        'RPM vs Exhaust Data': []
    }

    def process_row(row):
        return list(dict.fromkeys(cell.text.strip() for cell in row if cell.text.strip()))

    for table in doc.tables:
        for row in table.rows:
            row_data = process_row(row.cells)
            if len(row_data) >= 2:
                if 'Owner Name' in row_data:
                    data['General Information']['Owner Name'] = row_data[1]
                elif 'Owner Mobile' in row_data:
                    data['General Information']['Owner Mobile'] = row_data[1]
                elif 'Owner Email' in row_data:
                    data['General Information']['Owner Email'] = row_data[1]
                elif 'Atm. temperature' in row_data:
                    data['General Information']['Atm. Temperature'] = row_data[1]
                elif 'Barometric pressure' in row_data:
                    data['General Information']['Barometric Pressure (mmHg)'] = row_data[-1]
                elif 'Dynamometer' in row_data:
                    data['Test Information']['Dynamometer'] = row_data[-1]
                elif 'Route' in row_data:
                    data['Test Information']['Route'] = row_data[-1]
                elif 'ECM test #' in row_data:
                    data['Test Information']['ECM Test Numbers'] = row_data[1]
                elif '5-Gas test #' in row_data:
                    data['Test Information']['5-Gas Test Numbers'] = row_data[1]
                elif 'Type' in row_data:
                    data['Vehicle Information']['Type'] = row_data[1]
                elif 'Model' in row_data:
                    data['Vehicle Information']['Model'] = row_data[1]
                elif 'Year' in row_data:
                    data['Vehicle Information']['Year'] = str(row_data[1])
                elif 'Mileage' in row_data:
                    data['Vehicle Information']['Mileage'] = int(row_data[1])
                elif 'Technology' in row_data:
                    data['Vehicle Information']['Technology (US EPA)'] = row_data[1]
                elif 'Engine size' in row_data:
                    data['Vehicle Information']['Engine Size'] = float(row_data[1])
                elif '# Cylinders' in row_data:
                    data['Vehicle Information']['Number of Cylinders'] = int(row_data[1])
                elif 'Displacement volume' in row_data:
                    data['Vehicle Information']['Displacement Volume'] = float(row_data[1])
                elif 'RPM' in row_data and 'Exhaust Velocity' in row_data:
                    continue
                elif len(row_data) == 3:  # RPM data row
                    data['RPM vs Exhaust Data'].append({
                        'RPM': int(row_data[0]),
                        'Exhaust Velocity (m/s)': float(row_data[1]),
                        'Exhaust Temperature (C)': float(row_data[2])
                    })

    # Write data to JSON
    with open(output_file, 'w') as json_object:
        json.dump(data, json_object, indent=4)

    print(f'Data successfully written to {output_file}')


if __name__ == "__main__":
    VEHICLE_DATA = gen_hierarch_vehicle_dict()
    path_to_ram1500 = VEHICLE_DATA['2007']['Dodge']['RAM1500']
    
    # assert VEHICLE_DATA['2008']['Nissan']['Pathfinder'] == 'data/vehicle_v2/2008/Nissan/Pathfinder', "Path to Pathfinder is incorrect.'

    assert VEHICLE_DATA['2019']['Toyota']['Tacoma'] == 'data/vehicle_v2/2019/Toyota/Tacoma', 'Path to 2019 Tacoma is incorrect.'

    a = VEHICLE_DATA['2019']['Toyota']['Tacoma']  
    
    
    v = 'data/vehicle_v2/2019/Toyota/Tacoma'

    assert '2010' in VEHICLE_DATA, '2010 is missing from VEHICLE_DATA.'
    assert 'Chevrolet' in VEHICLE_DATA['2011'], 'Chevrolet is missing from VEHICLE_DATA for 2011.'

    try:
        VEHICLE_DATA['2020']['NonExistent']['Model']
    except KeyError:
        print('Test passed: Non-existent vehicle throws KeyError.')
        
        vehicle_test_doc_generate()

import os
import json
from typing import Dict, Any


BASEDIR = os.path.abspath(os.path.dirname(__file__))
VEHICLE_CONFIG_JSON = os.path.join(BASEDIR, 'veh_config_path.json')


def load_vehicle_config(json_file: str = VEHICLE_CONFIG_JSON) -> Dict[str, Any]:
    """
    Loads the vehicle configuration paths from a JSON file.
    
    Structure 
        { year: { make: { model: path_str } } }
        
    return: Dict     
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

CONFIG = load_vehicle_config()  # load config

if __name__ == '__main__':
    print(f'Base directory: {BASEDIR}')
    print(f'vehicle config keys: {list(CONFIG.keys())}')
    try:
        ram1500_path = CONFIG['2007']['Dodge']['RAM1500']
        print(f'Path for 2007 Dodge RAM1500: {ram1500_path}')
    except KeyError:
        print('2007 Dodge RAM1500 not found in the config file.')

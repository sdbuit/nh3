# basedir.py

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
DATA_DIR = os.path.join(BASE_DIR, 'data')
GEODETIC_DATA_DIR = os.path.join(DATA_DIR, 'geo')

if __name__ == '__main__':
    print(BASE_DIR)
    print(DATA_DIR)
    print(GEODETIC_DATA_DIR)
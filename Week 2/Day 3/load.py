from __future__ import print_function
from glob import glob

import os
import pandas as pd
import tarfile
import urllib.request
import zipfile


class Flights():
    def __init__(self, data_directory, url, number_of_rows):
        self.url = url
        self.data_directory = 'data'
        self.number_of_rows = int(number_of_rows)
        flights_raw = os.path.join(self.data_directory, 'nycflights.tar.gz')
        flightdir = os.path.join(self.data_directory, 'nycflights')
        jsondir = os.path.join(self.data_directory, 'flightjson')

        if not os.path.exists(self.data_directory):
            os.mkdir(self.data_directory)

        if not os.path.exists(flights_raw):
            print("- Downloading NYC Flights dataset... ", end='', flush=True)
            url = self.url
            urllib.request.urlretrieve(url, flights_raw)
            print("done", flush=True)

        if not os.path.exists(flightdir):
            print("- Extracting flight data... ", end='', flush=True)
            tar_path = os.path.join(self.data_directory, 'nycflights.tar.gz')
            with tarfile.open(tar_path, mode='r:gz') as flights:
                flights.extractall('data/')
            print("done", flush=True)

        if not os.path.exists(jsondir):
            print("- Creating json data... ", end='', flush=True)
            os.mkdir(jsondir)
            for path in glob(os.path.join(self.data_directory, 'nycflights', '*.csv')):
                prefix = os.path.splitext(os.path.basename(path))[0]
                df = pd.read_csv(path).iloc[:self.number_of_rows]
                df.to_json(os.path.join(self.data_directory, 'flightjson', prefix + '.json'),
                        orient='records', lines=True)
            print("done", flush=True)

        print("** Download is completed! **") 
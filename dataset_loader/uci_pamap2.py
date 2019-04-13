#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import csv
import math
from collections import OrderedDict
from decimal import Decimal
from enum import Enum

import dataset_loader.sqlite_util as sqlite_util

DATA_PATH = "PAMAP2_Dataset/PAMAP2_Dataset"

DATASET_NAME = "uci_pamap2"

class Location(Enum):
    CHEST = 1
    ANKLE = 2
    HAND = 3

sampling_freq = Decimal('100.0')
sampling_interval = Decimal('1') / sampling_freq

sampling_freq_heartrate = 9.0
sampling_interval_heartrate = 1 / sampling_freq_heartrate

samples_table = "{}__samples".format(DATASET_NAME)
samples_schema = {
    "sample_id"     : "INTEGER PRIMARY KEY",
    "timestamp"     : "DECIMAL",
    "subject_id"    : "INTEGER",
    "activity_id"   : "INTEGER",
    "heart_rate"    : "INTEGER",
}
samples_n_columns = len(samples_schema)

sensor_readings_table = "{}__sensor_readings".format(DATASET_NAME)
sensor_readings_schema = {
    "sample_id"     : "INTEGER NOT NULL",
    "location"      : "INTEGER NOT NULL",
    "temperature"   : "FLOAT",
    "acc_x"         : "FLOAT",
    "acc_y"         : "FLOAT",
    "acc_z"         : "FLOAT",
    "acc_x_2"       : "FLOAT",
    "acc_y_2"       : "FLOAT",
    "acc_z_2"       : "FLOAT",
    "gyro_x"        : "FLOAT",
    "gyro_y"        : "FLOAT",
    "gyro_z"        : "FLOAT",
    "magn_x"        : "FLOAT",
    "magn_y"        : "FLOAT",
    "magn_z"        : "FLOAT",
    "PRIMARY KEY"   : "(sample_id, location)",
    "FOREIGN KEY"   : "(sample_id) REFERENCES {}(sample_id)".format(samples_table),
}
sensor_readings_n_columns = len(sensor_readings_schema) - 2

# column_headings = [
#     "timestamp",
#     "activity_id",
#     "heart_rate",
# ]

def _load_dataset(path, loader_func):
    _data_path = os.path.join(path, DATA_PATH)
    tbls = []
    subdirectories = ["Protocol", "Optional"]
    for subdir in subdirectories:
        tbls.append(OrderedDict(
        ( int(filepath[7:-4]) , loader_func(os.path.join(_data_path, subdir, filepath)) )
        for filepath in
            os.listdir(os.path.join(_data_path, subdir))
            if filepath.endswith('.dat')
    ))
    return tbls

def load_dataset_to_mem(path):
    return _load_dataset(path, np.genfromtxt)

def int_or_none(s):
    try:
        return int(s)
    except:
        return None

def float_or_none(s):
    try:
        v = float(s)
        return v if not math.isnan(v) else None
    except:
        return None

def _csv_loader(path):
    dtypes = [
        Decimal,
        int_or_none,
        int_or_none,
    ] + ([float_or_none] * (17 * 3))
    with open(path, 'r') as f:
        for row in csv.reader(f, delimiter=' '):
            yield [ func(v) for (func, v) in zip(dtypes,row) ]


def load_dataset_to_sqlite(cur, path):
    tbls = _load_dataset(path, _csv_loader)
    store_dataset_to_sql(cur, tbls)

def row_generator(tbls_lst, index=1):
    for tbls in tbls_lst:
        for (subject_id, tbl) in tbls.items():
            for row in tbl:
                yield (index, subject_id, row)
                index = index + 1

def samples_from_row(index, subject_id, row):
    # row[0] , timestamp, is of type Decimal
    return (index, str(row[0]), subject_id, row[1], row[2])

def sensor_readings_from_row(index, subject_id, row):
    return [
        (index, Location.HAND.value, *row[3:16]),
        (index, Location.CHEST.value, *row[20:33]),
        (index, Location.ANKLE.value, *row[37:50]),
    ]

def check_sqlite_table_not_exists(cur):
    return ((not sqlite_util.check_sql_table_exists(cur, samples_table)) and
           (not sqlite_util.check_sql_table_exists(cur, sensor_readings_table)))

def store_dataset_to_sql(cur, tbls_lst):
    try:
        cur.execute("BEGIN TRANSACTION")
        sqlite_util.create_table_if_not_exists(cur, samples_table, samples_schema)
        sqlite_util.create_table_if_not_exists(cur, sensor_readings_table, sensor_readings_schema)
        _sql_stmt_samples = "INSERT INTO {} VALUES ({})".format(samples_table, ','.join(['?'] * samples_n_columns))
        _sql_stmt_sensors = "INSERT INTO {} VALUES ({})".format(sensor_readings_table, ','.join(['?'] * sensor_readings_n_columns))
    
        for row in row_generator(tbls_lst):
            cur.execute(_sql_stmt_samples, samples_from_row(*row))
            for val in sensor_readings_from_row(*row):
                cur.execute(_sql_stmt_sensors, val)
        cur.execute("COMMIT TRANSACTION")
    except:
        cur.execute("ROLLBACK TRANSACTION")
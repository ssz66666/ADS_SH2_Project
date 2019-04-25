#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import pandas as pd
import numpy as np
import os
import csv
from collections import OrderedDict
from decimal import Decimal
from enum import Enum

import dataset_loader.sqlite_util as sqlite_util
from misc import mul_str_arr

DATA_PATH = "Activity Recognition from Single Chest-Mounted Accelerometer/Activity Recognition from Single Chest-Mounted Accelerometer"

DATASET_NAME = "uci_chest_accelerometer"

sampling_freq = Decimal('52.0')
sampling_interval = Decimal('1.0') / sampling_freq
# column_headings = [
#     "sequential_number"
#     "chest_acc_x",
#     "chest_acc_y",
#     "chest_acc_z",
#     "label",
# ]

samples_table = "__".join([DATASET_NAME, "samples"])
samples_schema = OrderedDict([
    ("sample_id"     , "INTEGER PRIMARY KEY"),
    ("timestamp"     , "DECIMAL"),
    ("subject_id"    , "INTEGER"),
    ("activity_id"   , "INTEGER"),
])
samples_n_columns = len(samples_schema)

sensor_readings_table = "__".join([DATASET_NAME, "sensor_readings"])
sensor_readings_schema = OrderedDict(
    [
        ("sample_id", "INTEGER PRIMARY KEY"),
    ] +
    [
        (k, "FLOAT") for k in
        mul_str_arr(["chest_acc"], ["x","y","z"]) 
      
    ] + 
    [("FOREIGN KEY", "(sample_id) REFERENCES {}(sample_id)".format(samples_table))]
)
sensor_readings_n_columns = len(sensor_readings_schema) - 1

def _load_dataset(path, loader):
    return OrderedDict(
        (int(filepath[:-4]) , loader(os.path.join(path, DATA_PATH, filepath)))
        for filepath in os.listdir(os.path.join(path, DATA_PATH))
            if filepath.endswith('.csv')
    )

def load_dataset_to_mem(path):
    return _load_dataset(path, np.genfromtxt)

def _csv_loader(path):
    dtypes = [float] * 4 + [int]
    with open(path, 'r') as f:
        for row in csv.reader(f, delimiter=','):
            yield [ func(v) for (func, v) in zip(dtypes,row) ]

def load_dataset_to_sqlite(cur, path):
    tbls = _load_dataset(path, _csv_loader)
    store_dataset_to_sql(cur, tbls)

def row_generator(tbls, index=1):
    for (subject_id, tbl) in tbls.items():
        timestamp = Decimal('0')
        for row in tbl:
            yield (index, timestamp, subject_id, row)
            index = index + 1
            timestamp += sampling_interval

def samples_from_row(index, timestamp, subject_id, row):
    return (index, str(timestamp), subject_id, row[4])

def sensor_readings_from_row(index, timestamp, subject_id, row):
    return (index, *row[1:-1])

def check_sqlite_table_not_exists(cur):
    return ((not sqlite_util.check_sql_table_exists(cur, samples_table)) and
           (not sqlite_util.check_sql_table_exists(cur, sensor_readings_table)))

def store_dataset_to_sql(cur, tbls):
    try:
        cur.execute("BEGIN TRANSACTION")
        sqlite_util.create_table_if_not_exists(cur, samples_table, samples_schema)
        sqlite_util.create_table_if_not_exists(cur, sensor_readings_table, sensor_readings_schema)
        _sql_stmt_samples = "INSERT INTO {} VALUES ({})".format(samples_table, ','.join(['?'] * samples_n_columns))
        _sql_stmt_sensors = "INSERT INTO {} VALUES ({})".format(sensor_readings_table, ','.join(['?'] * sensor_readings_n_columns))
        
        for row in row_generator(tbls):
            cur.execute(_sql_stmt_samples, samples_from_row(*row))
            cur.execute(_sql_stmt_sensors, sensor_readings_from_row(*row))
        cur.execute("COMMIT TRANSACTION")
    except:
        cur.execute("ROLLBACK TRANSACTION")

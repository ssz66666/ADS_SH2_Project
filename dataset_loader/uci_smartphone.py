#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import csv
from collections import OrderedDict
from decimal import Decimal
from enum import Enum

import dataset_loader.sqlite_util as sqlite_util
from misc import mul_str_arr

DATASET_NAME = "uci_smartphone"

DATA_PATH = "UCI HAR Dataset/UCI HAR Dataset"

sampling_freq = Decimal('50.0')
sampling_interval = Decimal('1.0') / sampling_freq

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
        mul_str_arr(["body_acc", "body_gyro"], ["x","y","z"])
    ] + 
    [("FOREIGN KEY", "(sample_id) REFERENCES {}(sample_id)".format(samples_table))]
)
sensor_readings_n_columns = len(sensor_readings_schema) - 1

def _load_dataset(path, loaders):
    return [
        loaders(*[ os.path.join(path, DATA_PATH, subdirectory, filepath)
            for filepath in ["subject_{}.txt".format(subdirectory), "y_{}.txt".format(subdirectory)] + 
            ["Inertial Signals/{}_{}_{}.txt".format(s,a,subdirectory)
                for s in ["total_acc", "body_gyro"]
                for a in ["x","y","z"]] ])
        for subdirectory in ["test", "train"]
    ]

def _reader(path, type_conv):
    with open(path, 'r') as f:
        for row in csv.reader(f, delimiter=' ', skipinitialspace=True):
            yield np.array([ type_conv(item) for item in row ])

def _data_reader(*paths):
    rdrs = [ _reader(paths[0], int), _reader(paths[1], int), *[ _reader(p, float) for p in paths[2:] ] ]
    return zip(*rdrs)

def load_dataset_to_sqlite(cur, path):
    tbls = _load_dataset(path, _data_reader)
    store_dataset_to_sql(cur, tbls)

def row_generator(tbls, index=1):
    # TODO fix sliding window related problem
    raise NotImplementedError("FIX ME")
    for tbl in tbls:
        timestamp = Decimal('0')
        last_subject = None
        for row in tbl:
            if row[0][0] != last_subject:
                timestamp = Decimal('0')
                last_subject = row[0][0]
            for i in range(len(row[2])):
                yield (index, timestamp, i, row)
                index = index + 1
                timestamp += sampling_interval

def samples_from_row(index, timestamp, i, row):
    return (index, str(timestamp), str(row[0][0]), str(row[1][0]))

def sensor_readings_from_row(index, timestamp, i, row):
    return (index, *map(lambda wind: wind[i], row[2:]))

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
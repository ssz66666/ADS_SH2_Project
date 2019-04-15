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
from misc import mul_str_arr

DATA_PATH = "PAMAP2_Dataset/PAMAP2_Dataset"

DATASET_NAME = "uci_pamap2"

sampling_freq = Decimal('100.0')
sampling_interval = Decimal('1') / sampling_freq

sampling_freq_heartrate = Decimal('9.0')
sampling_interval_heartrate = Decimal('1') / sampling_freq_heartrate

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
        ("sample_id",   "INTEGER PRIMARY KEY"),
        ("heart_rate",  "FLOAT"),
    ] +
    [
        (k, "FLOAT") for k in 
        mul_str_arr(
            ["hand", "chest", "ankle"],
            ["temperature"] +
            mul_str_arr(["acc", "gyro", "magn"], ["x","y","z"])
        )
    ] + 
    [("FOREIGN KEY", "(sample_id) REFERENCES {}(sample_id)".format(samples_table))]
)
sensor_readings_n_columns = len(sensor_readings_schema) - 1

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
        float_or_none,
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
    return (index, str(row[0]), subject_id, row[1])

def sensor_readings_from_row(index, subject_id, row):
    return (index, *row[2:7], *row[10:16], *row[20:24], *row[27:33], *row[37:41], *row[44:50])

def check_sqlite_table_not_exists(cur):
    return ((not sqlite_util.check_sql_table_exists(cur, samples_table)) and
           (not sqlite_util.check_sql_table_exists(cur, sensor_readings_table)))

def store_dataset_to_sql(cur, tbls_lst, force_reload=False):
    _store_dataset_to_sql(cur, tbls_lst, row_generator, samples_from_row, sensor_readings_from_row, force_reload)

def _store_dataset_to_sql(cur, tbls_lst, row_gen, samples_trans, sensor_trans, force_reload=False):
    try:
        cur.execute("BEGIN TRANSACTION")
        if force_reload:
            # drop table first, use with caution!
            sqlite_util.drop_table_if_exists(cur, samples_table)
            sqlite_util.drop_table_if_exists(cur, sensor_readings_table)
        sqlite_util.create_table_if_not_exists(cur, samples_table, samples_schema)
        sqlite_util.create_table_if_not_exists(cur, sensor_readings_table, sensor_readings_schema)
        _sql_stmt_samples = "INSERT INTO {} VALUES ({})".format(samples_table, ','.join(['?'] * samples_n_columns))
        _sql_stmt_sensors = "INSERT INTO {} VALUES ({})".format(sensor_readings_table, ','.join(['?'] * sensor_readings_n_columns))
    
        for row in row_gen(tbls_lst):
            cur.execute(_sql_stmt_samples, samples_trans(*row))
            cur.execute(_sql_stmt_sensors, sensor_trans(*row))
        cur.execute("COMMIT TRANSACTION")
    except:
        cur.execute("ROLLBACK TRANSACTION")

def _HACK_csv_loader(p):
    dtypes = [
        Decimal,
        int_or_none,
        float_or_none,
    ] + ([float_or_none] * (10 * 3))
    with open(p, 'r') as f:
        # discard first row
        next(f)
        for row in csv.reader(f, delimiter=','):
            yield [ func(v) for (func, v) in zip(dtypes,row[9:]) ]

def _HACK_sensor_from_row(index, subject_id, row):
    return (index, *row[2:])

def _HACK_force_store_updated_dataset_to_sql(cur, path):
    tbls = [OrderedDict([
        (int(filepath[14:-4])  ,  _HACK_csv_loader(os.path.join(path, filepath)))
        for filepath in
            os.listdir(path)
            if filepath.endswith(".csv")
    ])]
    _store_dataset_to_sql(cur, tbls, row_generator, samples_from_row, _HACK_sensor_from_row, True)
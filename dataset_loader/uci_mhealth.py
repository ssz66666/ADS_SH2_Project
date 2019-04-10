#!/usr/bin/env python
# -*- coding: utf-8 -*-

DATASET_NAME = "uci_mhealth"

import pandas as pd
import numpy as np
import os
from enum import Enum

import dataset_loader.sqlite_util as sqlite_util

DATA_PATH = "MHEALTHDATASET/MHEALTHDATASET"

sampling_freq = 50.0
sampling_interval = 1.0 / sampling_freq
# column_headings = [
#     "timestamp",
#     "chest_acc_x",
#     "chest_acc_y",
#     "chest_acc_z",
#     "ecg_1",
#     "ecg_2",
#     "left_ankle_acc_x",
#     "left_ankle_acc_y",
#     "left_ankle_acc_z",
#     "left_ankle_gyro_x",
#     "left_ankle_gyro_y",
#     "left_ankle_gyro_z",
#     "left_ankle_magn_x",
#     "left_ankle_magn_y",
#     "left_ankle_magn_z",
#     "right_lower_arm_acc_x",
#     "right_lower_arm_acc_y",
#     "right_lower_arm_acc_z",
#     "right_lower_arm_gyro_x",
#     "right_lower_arm_gyro_y",
#     "right_lower_arm_gyro_z",
#     "right_lower_arm_magn_x",
#     "right_lower_arm_magn_y",
#     "right_lower_arm_magn_z",
#     "label",
#     "subject_id",
# ]

samples_table = "{}__samples".format(DATASET_NAME)
sensor_readings_table = "{}__sensor_readings".format(DATASET_NAME)

samples_schema = {
    "sample_id"     : "INTEGER PRIMARY KEY",
    "timestamp"     : "DECIMAL",
    "subject_id"    : "INTEGER",
    "activity_id"   : "INTEGER",
}

sensor_readings_schema = {
    "sample_id"     : "INTEGER NOT NULL",
    "location"      : "INTEGER NOT NULL",
    "acc_x"         : "FLOAT NOT NULL",
    "acc_y"         : "FLOAT NOT NULL",
    "acc_z"         : "FLOAT NOT NULL",
    "gyro_x"        : "FLOAT",
    "gyro_y"        : "FLOAT",
    "gyro_z"        : "FLOAT",
    "magn_x"        : "FLOAT",
    "magn_y"        : "FLOAT",
    "magn_z"        : "FLOAT",
    "ecg_1"         : "FLOAT",
    "ecg_2"         : "FLOAT",
    "PRIMARY KEY"   : "(sample_id, location)",
    "FOREIGN KEY"   : "(sample_id) REFERENCES {}(sample_id)".format(samples_table),
}

class Location(Enum):
    CHEST = 1
    LEFT_ANKLE = 2
    RIGHT_LOWER_ARM = 3

def load_dataset_to_mem(path):
    tbls = [
        np.genfromtxt(os.path.join(path, DATA_PATH, filepath))
        for filepath in os.listdir(os.path.join(path, DATA_PATH))
            if filepath.endswith('.log')
    ]
    # insert timestamp and subject id
    tbls_new = []
    for subject in range(len(tbls)):
        nrows, ncols = np.shape(tbls[subject])
        timestamp = np.linspace(0.0, (nrows-1) * sampling_interval, nrows)
        tbl_new = np.full((nrows, ncols + 2), float(subject + 1))
        tbl_new[:, 0] = timestamp
        tbl_new[:, 1:-1] = tbls[subject]
        tbls_new.append(tbl_new)
    return [ pd.DataFrame(t) for t in tbls_new ]

def samples_row_generator(dfs):
    index = 0
    for df in dfs:
        for _, row in df.iterrows():
            index = index + 1
            yield (index, row[0], row[25], row[24])

def sensor_readings_row_generator(dfs):
    index = 0
    for df in dfs:
        for _, row in df.iterrows():
            index = index + 1
            yield (index, Location.CHEST.value, *row[1:4], *([None] * 6), *row[4:6])
            yield (index, Location.LEFT_ANKLE.value, *row[6:15], *([None] * 2),)
            yield (index, Location.RIGHT_LOWER_ARM.value, *row[15:24], *([None] * 2),)

def store_dataset_to_sql(cur, dfs):
    sqlite_util.create_table_if_not_exists(cur, samples_table, samples_schema)
    sqlite_util.create_table_if_not_exists(cur, sensor_readings_table, sensor_readings_schema)
    cur.executemany("INSERT INTO {} VALUES ({})".format(samples_table, ','.join(['?'] * len(samples_schema))),
        samples_row_generator(dfs))
    cur.executemany("INSERT INTO {} VALUES ({})".format(sensor_readings_table, ','.join(['?'] * (len(sensor_readings_schema) - 2))),
        sensor_readings_row_generator(dfs))
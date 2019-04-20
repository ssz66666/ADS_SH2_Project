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
        loaders(os.path.join(path, DATA_PATH, subdirectory, filepath))
        for subdirectory in ["test", "train"]
        for filepath in ["subject_{}.txt", "y_{}.txt"] + 
            ["Inertial Signals/{}_{}_{}.txt".format(s,a,subdirectory)
                for s in ["total_acc", "body_gyro"]
                for a in ["x","y","z"]]
    ]
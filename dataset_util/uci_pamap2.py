#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

from dataset_loader.uci_pamap2 import samples_table, sensor_readings_table, DATASET_NAME
from dataset_util import preprocess
from dataset_util.preprocess import DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_OVERLAP
from misc import mul_str_arr

distinct_subject_query = """
SELECT DISTINCT subject_id FROM {};
""".format(samples_table)

distinct_activity_query = """
SELECT DISTINCT activity_id FROM {};
""".format(samples_table)

raw_table_query_with_subject_id = ("""
SELECT
    activity_id, timestamp, subject_id, heart_rate,
""" +
", ".join(mul_str_arr(
        ["hand", "chest", "ankle"],
        ["temperature"] +
        mul_str_arr(["acc", "gyro", "magn"], ["x","y","z"])
    )) + """
FROM
    {0}, {1}
WHERE {0}.sample_id = {1}.sample_id AND subject_id = ?;
""").format(samples_table, sensor_readings_table)

raw_table_query = ("""
SELECT
    activity_id, timestamp, subject_id, heart_rate,
""" +
", ".join(mul_str_arr(
        ["hand", "chest", "ankle"],
        ["temperature"] +
        mul_str_arr(["acc", "gyro", "magn"], ["x","y","z"])
    )) + """
FROM
    {0}, {1}
WHERE {0}.sample_id = {1}.sample_id;
""").format(samples_table, sensor_readings_table)

def to_classification(df):
    return df.iloc[:,1:], df.loc[:,"activity_id"]

def get_subject_ids(conn):
    return list(map(lambda x: int(x[0]), conn.execute(distinct_subject_query)))

def get_activity_ids(conn):
    return list(map(lambda x: int(x[0]), conn.execute(distinct_activity_query)))

def to_sliding_windows(conn, size=DEFAULT_WINDOW_SIZE, overlap=DEFAULT_WINDOW_OVERLAP):
    ids = conn.execute(distinct_subject_query)
    for subject_id in ids:
        yield preprocess.query_to_sliding_windows(conn.execute(
            raw_table_query_with_subject_id, (subject_id,)
        ), size=size, overlap=overlap)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset_loader.uci_mhealth import samples_table, sensor_readings_table, DATASET_NAME
from dataset_util import preprocess
from dataset_util.preprocess import DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_OVERLAP
from misc import mul_str_arr

distinct_subject_query = """
SELECT DISTINCT subject_id FROM {};
""".format(samples_table)

raw_table_query_with_subject_id = ("""
SELECT
    activity_id, timestamp, subject_id,
""" +
", ".join(mul_str_arr(["chest_acc"], ["x","y","z"]) +
    ["ecg_1", "ecg_2"] +
    mul_str_arr(["left_ankle", "right_lower_arm"],
                ["acc", "gyro", "magn"],
                ["x", "y", "z"])) + """
FROM
    {0}, {1}
WHERE {0}.sample_id = {1}.sample_id AND subject_id = ?;
""").format(samples_table, sensor_readings_table)

raw_table_query = ("""
SELECT
    activity_id, timestamp, subject_id,
""" +
", ".join(mul_str_arr(["chest_acc"], ["x","y","z"]) +
    ["ecg_1", "ecg_2"] +
    mul_str_arr(["left_ankle", "right_lower_arm"],
                ["acc", "gyro", "magn"],
                ["x", "y", "z"])) + """
FROM
    {0}, {1}
WHERE {0}.sample_id = {1}.sample_id;
""").format(samples_table, sensor_readings_table)

def to_classification(df):
    return df.loc[:,"timestamp":], df.loc[:,"activity_id"]

def to_sliding_windows(cur, size=DEFAULT_WINDOW_SIZE, overlap=DEFAULT_WINDOW_OVERLAP):
    cur.execute(distinct_subject_query)
    for subject_id in list(cur):
        yield preprocess.query_to_sliding_windows(cur.execute(
            raw_table_query_with_subject_id, (subject_id[0],)
        ), size)

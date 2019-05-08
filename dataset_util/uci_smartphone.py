#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

from dataset_loader.uci_smartphone import samples_table, sensor_readings_table, DATASET_NAME
from dataset_util import preprocess
from dataset_util.preprocess import DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_OVERLAP
from misc import mul_str_arr

# uci_smartphone
# acc unit: g
# gyro unit: rad/s

raw_table_query_with_subject_id = ("""
SELECT
    activity_id, subject_id,
""" +
", ".join(mul_str_arr(
        ["body"],
        mul_str_arr(["acc", "gyro"], ["x","y","z"])
    )) + """
FROM
    {0}, {1}
WHERE {0}.sample_id = {1}.sample_id AND subject_id = ?;
""").format(samples_table, sensor_readings_table)

raw_table_query_shared_data = ("""
SELECT
    activity_id, timestamp, subject_id,
""" +
", ".join(
        mul_str_arr(["body"], ["acc", "gyro"], ["x","y","z"])
    ) + """
FROM
    {0}, {1}
WHERE {0}.sample_id = {1}.sample_id AND activity_id != 0;
""").format(samples_table, sensor_readings_table)

def small_g_to_m_per_sec_sq(df):
    df.loc[:,['body_acc_x','body_acc_y','body_acc_z']] *= 9.81
    return df
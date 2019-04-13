#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset_loader.uci_mhealth import samples_table, sensor_readings_table, Location, DATASET_NAME
from dataset_util import preprocess

distinct_subject_query = """
SELECT DISTINCT subject_id FROM {};
""".format(samples_table)

raw_table_query_with_subject_id = """
SELECT
    activity_id, timestamp, subject_id,
    chest_acc_x, chest_acc_y, chest_acc_z, 
    left_ankle_acc_x, left_ankle_acc_y, left_ankle_acc_z, 
    left_ankle_gyro_x, left_ankle_gyro_y, left_ankle_gyro_z,
    left_ankle_magn_x, left_ankle_magn_y, left_ankle_magn_z,
    right_lower_arm_acc_x, right_lower_arm_acc_y, right_lower_arm_acc_z, 
    right_lower_arm_gyro_x, right_lower_arm_gyro_y, right_lower_arm_gyro_z,
    right_lower_arm_magn_x, right_lower_arm_magn_y, right_lower_arm_magn_z
FROM
    {0},
    (SELECT sample_id as sid_1, acc_x as chest_acc_x, acc_y as chest_acc_y, acc_z as chest_acc_z
        FROM {1} WHERE location = {2.CHEST.value}),
    (SELECT sample_id as sid_2, acc_x as left_ankle_acc_x, acc_y as left_ankle_acc_y, acc_z as left_ankle_acc_z,
            gyro_x as left_ankle_gyro_x, gyro_y as left_ankle_gyro_y, gyro_z as left_ankle_gyro_z,
            magn_x as left_ankle_magn_x, magn_y as left_ankle_magn_y, magn_z as left_ankle_magn_z
                FROM {1} WHERE location = {2.LEFT_ANKLE.value}),
    (SELECT sample_id as sid_3, acc_x as right_lower_arm_acc_x, acc_y as right_lower_arm_acc_y, acc_z as right_lower_arm_acc_z,
            gyro_x as right_lower_arm_gyro_x, gyro_y as right_lower_arm_gyro_y, gyro_z as right_lower_arm_gyro_z,
            magn_x as right_lower_arm_magn_x, magn_y as right_lower_arm_magn_y, magn_z as right_lower_arm_magn_z
                FROM {1} WHERE location = {2.RIGHT_LOWER_ARM.value})
WHERE subject_id = {3} AND {0}.sample_id = sid_1 AND {0}.sample_id = sid_2 AND {0}.sample_id = sid_3;
""".format(samples_table, sensor_readings_table, Location, "{}")

raw_table_query = """
SELECT
    activity_id, timestamp, subject_id,
    chest_acc_x, chest_acc_y, chest_acc_z, 
    left_ankle_acc_x, left_ankle_acc_y, left_ankle_acc_z, 
    left_ankle_gyro_x, left_ankle_gyro_y, left_ankle_gyro_z,
    left_ankle_magn_x, left_ankle_magn_y, left_ankle_magn_z,
    right_lower_arm_acc_x, right_lower_arm_acc_y, right_lower_arm_acc_z, 
    right_lower_arm_gyro_x, right_lower_arm_gyro_y, right_lower_arm_gyro_z,
    right_lower_arm_magn_x, right_lower_arm_magn_y, right_lower_arm_magn_z
FROM
    {0},
    (SELECT sample_id as sid_1, acc_x as chest_acc_x, acc_y as chest_acc_y, acc_z as chest_acc_z
        FROM {1} WHERE location = {2.CHEST.value}),
    (SELECT sample_id as sid_2, acc_x as left_ankle_acc_x, acc_y as left_ankle_acc_y, acc_z as left_ankle_acc_z,
            gyro_x as left_ankle_gyro_x, gyro_y as left_ankle_gyro_y, gyro_z as left_ankle_gyro_z,
            magn_x as left_ankle_magn_x, magn_y as left_ankle_magn_y, magn_z as left_ankle_magn_z
                FROM {1} WHERE location = {2.LEFT_ANKLE.value}),
    (SELECT sample_id as sid_3, acc_x as right_lower_arm_acc_x, acc_y as right_lower_arm_acc_y, acc_z as right_lower_arm_acc_z,
            gyro_x as right_lower_arm_gyro_x, gyro_y as right_lower_arm_gyro_y, gyro_z as right_lower_arm_gyro_z,
            magn_x as right_lower_arm_magn_x, magn_y as right_lower_arm_magn_y, magn_z as right_lower_arm_magn_z
                FROM {1} WHERE location = {2.RIGHT_LOWER_ARM.value})
WHERE {0}.sample_id = sid_1 AND {0}.sample_id = sid_2 AND {0}.sample_id = sid_3;
""".format(samples_table, sensor_readings_table, Location)

def to_classification(df):
    return df.loc[:,"timestamp":], df.loc[:,"activity_id"]

def to_sliding_windows(cur, size=preprocess.DEFAULT_WINDOW_SIZE, overlap=preprocess.DEFAULT_WINDOW_OVERLAP):
    cur.execute(distinct_subject_query)
    for subject_id in list(cur):
        yield preprocess.query_to_sliding_windows(cur.execute(
            raw_table_query_with_subject_id.format(subject_id[0])
        ), size)

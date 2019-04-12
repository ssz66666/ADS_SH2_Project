#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum
import pandas as pd

from dataset_loader.uci_pamap2 import samples_table, sensor_readings_table, DATASET_NAME, Location

raw_table_query = """
SELECT
    activity_id, timestamp, subject_id, heart_rate,
    chest_temperature, chest_acc_x, chest_acc_y, chest_acc_z, 
    chest_gyro_x, chest_gyro_y, chest_gyro_z,
    chest_magn_x, chest_magn_y, chest_magn_z,
    ankle_temperature, ankle_acc_x, ankle_acc_y, ankle_acc_z, 
    ankle_gyro_x, ankle_gyro_y, ankle_gyro_z,
    ankle_magn_x, ankle_magn_y, ankle_magn_z,
    hand_temperature, hand_acc_x, hand_acc_y, hand_acc_z, 
    hand_gyro_x, hand_gyro_y, hand_gyro_z,
    hand_magn_x, hand_magn_y, hand_magn_z
FROM
    {0},
    (SELECT sample_id as sid_1, temperature as chest_temperature,
            acc_x as chest_acc_x, acc_y as chest_acc_y, acc_z as chest_acc_z,
            gyro_x as chest_gyro_x, gyro_y as chest_gyro_y, gyro_z as chest_gyro_z,
            magn_x as chest_magn_x, magn_y as chest_magn_y, magn_z as chest_magn_z
                FROM {1} WHERE location = {2.CHEST.value}),
    (SELECT sample_id as sid_2, temperature as ankle_temperature,
            acc_x as ankle_acc_x, acc_y as ankle_acc_y, acc_z as ankle_acc_z,
            gyro_x as ankle_gyro_x, gyro_y as ankle_gyro_y, gyro_z as ankle_gyro_z,
            magn_x as ankle_magn_x, magn_y as ankle_magn_y, magn_z as ankle_magn_z
                FROM {1} WHERE location = {2.ANKLE.value}),
    (SELECT sample_id as sid_3, temperature as hand_temperature,
            acc_x as hand_acc_x, acc_y as hand_acc_y, acc_z as hand_acc_z,
            gyro_x as hand_gyro_x, gyro_y as hand_gyro_y, gyro_z as hand_gyro_z,
            magn_x as hand_magn_x, magn_y as hand_magn_y, magn_z as hand_magn_z
                FROM {1} WHERE location = {2.HAND.value})
WHERE {0}.sample_id = sid_1 AND {0}.sample_id = sid_2 AND {0}.sample_id = sid_3;
""".format(samples_table, sensor_readings_table, Location)

def to_classification(df):
    return df.loc[:,"timestamp":], df.loc[:,"activity_id"]

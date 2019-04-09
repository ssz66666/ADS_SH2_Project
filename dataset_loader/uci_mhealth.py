#!/usr/bin/env python
# -*- coding: utf-8 -*-

DATASET_NAME = "uci_mhealth"

import pandas as pd
import numpy as np
import os

DATA_PATH = "MHEALTHDATASET/MHEALTHDATASET"

sampling_freq = 50.0
sampling_interval = 1.0 / sampling_freq
column_headings = [
    "timestamp",
    "chest_acc_x",
    "chest_acc_y",
    "chest_acc_z",
    "ecg_1",
    "ecg_2",
    "left_ankle_acc_x",
    "left_ankle_acc_y",
    "left_ankle_acc_z",
    "left_ankle_gyro_x",
    "left_ankle_gyro_y",
    "left_ankle_gyro_z",
    "left_ankle_magn_x",
    "left_ankle_magn_y",
    "left_ankle_magn_z",
    "right_lower_arm_acc_x",
    "right_lower_arm_acc_y",
    "right_lower_arm_acc_z",
    "right_lower_arm_gyro_x",
    "right_lower_arm_gyro_y",
    "right_lower_arm_gyro_z",
    "right_lower_arm_magn_x",
    "right_lower_arm_magn_y",
    "right_lower_arm_magn_z",
    "label",
    "subject_id",
]

def load_dataset_to_mem(path):
    tbls = [
        np.genfromtxt(os.path.join(path, DATA_PATH, filepath))
        for filepath in os.listdir(DATA_PATH)
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
    return [ pd.DataFrame(t, columns=column_headings) for t in tbls_new ]
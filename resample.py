from dataset_loader.uci_mhealth import samples_table, sensor_readings_table
from dataset_util.uci_mhealth import get_subject_ids
from misc import mul_str_arr
from config import SQLITE_DATABASE_FILE
import sqlite3
import pandas as pd
import numpy as np


def resample_raw_data(required_freq, df, **kwargs):
    try:
        df = df.set_index('timestamp')
    except:
        print('Already time series')

    req_period = 1 / required_freq
    resampled = df.resample('{}S'.format(req_period)).asfreq()
    resampled.index = range(len(resampled))
    resampled_interpolated = resampled.interpolate(**kwargs)
    return resampled_interpolated

def resample(df,new_freq, **kwargs):
    '''
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

    conn = sqlite3.connect(SQLITE_DATABASE_FILE)

    df = pd.read_sql(raw_table_query, conn)
    '''
    df.loc[:, 'timestamp'] = pd.TimedeltaIndex(df.loc[:, 'timestamp'], unit="s")
    df = df.set_index('timestamp')
    
    subject_ids = np.unique(df.loc[:,"subject_id"])
    dataset = []
    for i in subject_ids:
        subject_id = df.loc[df['subject_id'] == i]
        activity_ids = np.unique(subject_id.loc[:,"activity_id"])
        for aid in activity_ids:
            data_to_resample = subject_id.loc[subject_id["activity_id"] == aid]
            resampled = resample_raw_data(new_freq, data_to_resample, **kwargs)
        dataset.append(resampled)

    return pd.concat(dataset)






















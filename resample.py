from dataset_loader.uci_mhealth import samples_table, sensor_readings_table, DATASET_NAME
from dataset_util import preprocess
from dataset_util.preprocess import DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_OVERLAP
from misc import mul_str_arr
from config import SQLITE_DATABASE_FILE
import sqlite3
import pandas.io.sql as psql
import pandas as pd

import datetime

def change_type(row):
    row['timestamp'] = datetime.datetime.utcfromtimestamp(float(row['timestamp'])).strftime("%H:%M:%S.%f")
    return row


def resample_raw_data(current_freq, required_freq, df):
    try:
        df = df.set_index('timestamp')
    except:
        print('Already time series')

    req_period = 1 / required_freq
    df = df.resample('{}S'.format(req_period)).interpolate()
    return df

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


subject_id_1 = df.loc[df['subject_id'] == 1]
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#print(subject_id_1)

subject_id_1 = subject_id_1.apply(change_type,axis = 1)

#print(subject_id_1)

subject_id_1 = subject_id_1.set_index('timestamp')
subject_id_1.index = pd.to_datetime(df.index,format = '%M:%S.%f')


resample_raw_data(100,subject_id_1)

print(subject_id_1)
exit(0)
















'''
print("READ FROM DB")

df['timestamp'].max()

print(df['timestamp'].max())

df = df.apply(change_type,axis = 1)

print("changed type",df)

df.index[df['timestamp'] == max(df['timestamp'])]

exit(0)

df[df.index.duplicated()]

df.index[df.index == max(df.index)]

exit(0)

df = df.set_index('timestamp')
df.index = pd.to_datetime(df.index,format = '%H:%M:%S.%f')

resample_raw_data(50,100,df)
'''
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from itertools import chain
from dataset_util import uci_mhealth
import sqlite3
from config import SQLITE_DATABASE_FILE

def extract_features(df_windows,all_feature):

    check = 0
    input_features = []
    no_cols = 0

    for list in df_windows:

        for df_window in list:
            features = []

            if (check==0):
                column_headers = df_window.columns.values
                no_cols = len(column_headers)
                column_headers = column_headers[3:len(column_headers)]
                check = 1

            #print(df_window['activity_id'].value_counts().idxmax()," ",df_window['subject_id'].iloc[0])

            features.append(df_window['activity_id'].value_counts().idxmax()) # to find the most frequent value in the activity_id col
            features.append(df_window['subject_id'].iloc[0])

            for j in range(3, no_cols):


                col = df_window.iloc[:,j]

                features.append(col.mean())

                quantiles = np.quantile(col, [0, 0.25, 0.5, 0.75, 1])
                features.append(quantiles[0])
                features.append(quantiles[1])
                features.append(quantiles[2])
                features.append(quantiles[3])
                features.append(quantiles[4])

                fft_results = fft(col)
                fft_results = np.sqrt(abs(fft_results))

                fft_quantiles = np.quantile(fft_results, [0, 0.25, 0.5, 0.75, 1])

                features.append(fft_quantiles[0])
                features.append(fft_quantiles[1])
                features.append(fft_quantiles[2])
                features.append(fft_quantiles[3])
                features.append(fft_quantiles[4])

                fft_max = np.max(fft_results)
                fft_mean = fft_results.mean()
                fft_SNR = fft_max/fft_mean
                features.append(fft_max)
                features.append(fft_mean)
                features.append(fft_SNR)

            input_features.append(features)

    sensors = []
    for i in column_headers:
        sensors.append(i)

    new_features = []
    for i in sensors:
        for j in all_feature:
            new_features.append('_'.join([i, j]))

    all_features = ['activity_id','subject_id',] + new_features
    df_features = pd.DataFrame(input_features, columns=all_features)

    return df_features


def main():
    # Connection to the DB and retrieve sliding windows
    conn = sqlite3.connect(SQLITE_DATABASE_FILE)

    sliding_windows_object = uci_mhealth.to_sliding_windows(conn) #.cursor(), size=100)

    # Create an array with the headers of the columns
    # sw_list =[] # sliding_windows list

    # sw_list = list(sliding_windows_object)

    ''' For debugging
    for i in sw_list:
        for j in i:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(j)
    exit(0)
    '''

    '''
    column_headers = []
    for i in sw_list:
        for j in i:
            column_headers = j.columns.values

    no_cols = len(column_headers)
    column_headers = column_headers[3:len(column_headers)]
    '''


    all_feature = ['mean','quantile_1','quantile_2','quantile_3', 'quantile_4','quantile_5','fft_quantile_1',
                    'fft_quantile_2','fft_quantile_3', 'fft_quantile_4','fft_quantile_5','fft_max','fft_avg','fft_SNR']
    features = extract_features(sliding_windows_object,all_feature)

    features.to_csv('input_features.csv', header=features.columns.values, index=False, sep='\t', mode='w')
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(features)

if __name__ == "__main__":
    main()
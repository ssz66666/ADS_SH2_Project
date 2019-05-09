from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection
import pandas as pd
import sqlite3
import math

from datetime import datetime as dt

from dataset_util import preprocess, uci_mhealth, uci_pamap2
from dataset_util.extract_input_features import all_feature, extract_features
# import matplotlib.pyplot as plt

import numpy as np
from config import SQLITE_DATABASE_FILE, TRAINING_SET_PROPORTION
import os

def axes_normaliser_1(data, regions, preferred, measurements, standing, walking, acc_maps):
    df = pd.DataFrame()
    acc_id = data['activity_id'].values
    subj_id = data['subject_id'].values
    df['subject_id'] = data['subject_id']
    s_index = data.index[data['activity_id'] == standing]
    w_index = data.index[data['activity_id'] == walking]

    if len(acc_maps) > 0:
        for x in acc_maps:
            for i in range(len(acc_id)):
                if acc_id[i] == x[0]:
                    acc_id[i] = x[1]

    # if len(subj_maps) > 0:
    #     for i in range(len(subj_id)):
    #         subj_id[i] = max(subj_maps)+subj_id[i]

    df['activity_id'] = acc_id
    # df['subject_id'] = subj_id/

    for r in regions:
        params = []; temps = []
        for i in range(len(measurements)):
            vars()['temp_' + str(i+1)] = []
            temps.append('temp_' + str(i+1))
        for i in data.columns:
            for v in range(len(measurements)):
                if r in i and measurements[v] in i:
                    # exec('[] = np.array()'.format(r + '_' + measurements[v]))
                    # vars()[r + '_' + measurements[v]] = np.array(data[i])
                    # eval(r + '_' + measurements[v]) = np.array(data[i])
                    eval('temp_' + str(v+1)).append(np.array(data[i]))
                    # print(eval(r + '_' + measurements[v]))
                    # if measurements[v] not in params:
                    params.append(measurements[v])

        q = 10; w = 10; z = 4; x = 4; y = 4
        for m in range(len(measurements)):
            if 'acc' in measurements[m]:
                for i in range(len(eval('temp_'+str(m+1)))):
                    neg = 1
                    if (np.mean(eval('temp_' + str(m+1))[i][s_index])) < 0:
                        neg = -1
                    # print(abs(np.mean(eval('temp_' + str(m+1))[i][s_index] - 9.81)))
                    if abs(np.mean(eval('temp_' + str(m+1))[i][s_index] - neg*9.81)) < q:
                        q = abs(np.mean(eval('temp_' + str(m+1))[i][s_index]) - neg*9.81)
                        z = i+1
                    if abs(np.mean(eval('temp_' + str(m+1))[i][w_index])) < w:
                        w = abs(np.mean(eval('temp_' + str(m+1))[i][w_index]))
                        y = i+1

        neg = 1
        for m in range(len(measurements)):
            if 'acc' in measurements[m]:
                if (np.mean(eval('temp_' + str(m + 1))[z-1])) < -5:
                    neg = -1

                # eval('temp_' + str(m + 1))[z-1] = eval('temp_' + str(m + 1))[z-1] * neg

        for i in [1, 2, 3]:
            if i != z and i != y:
                x = i
        c1=0;c2=0;c3=0
        if neg == -1:
            for i in ['x', 'y', 'z']:
                if i == 'x':
                    for j in range(len(params)):
                        df[preferred[regions.index(r)] + '_' + params[j] + '_' + i] = eval('temp_' + str(measurements.index(params[j])+1))[eval(i)-1]
                else:
                    for j in range(len(params)):
                        df[preferred[regions.index(r)] + '_' + params[j] + '_' + i] = -1*eval('temp_' + str(measurements.index(params[j])+1))[eval(i)-1]
        else:
            for i in ['x', 'y', 'z']:
                for j in range(len(params)):
                    df[preferred[regions.index(r)] + '_' + params[j] + '_' + i] = eval('temp_' + str(measurements.index(params[j])+1))[eval(i)-1]

    return df



# def rotation_matrix(axis, theta):
#     """
#     Return the rotation matrix associated with counterclockwise rotation about
#     the given axis by theta radians.
#     """
#     axis = np.asarray(axis)
#     axis = axis / math.sqrt(np.dot(axis, axis))
#     a = math.cos(theta / 2.0)
#     b, c, d = -axis * math.sin(theta / 2.0)
#     aa, bb, cc, dd = a * a, b * b, c * c, d * d
#     bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
#     return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
#                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
#                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
#
# v = [1, 1, 1]
# axis = [1, 0, 0]
# theta = math.pi
#
# print(np.dot(rotation_matrix(axis, theta), v))


# def cv_main():
#     with sqlite3.connect(SQLITE_DATABASE_FILE) as conn:
#         if os.path.exists('mhealth_features.pkl'):
#             features_mhealth = pd.read_pickle('mhealth_features.pkl')
#             features_mhealth = axes_normaliser(features_mhealth, ['chest', 'left_ankle', 'right_lower_arm'], ['acc', 'gyro'], 1, 4)
#             # print(features_mhealth)
#         else:
#             data = pd.read_sql_query(uci_mhealth.raw_table_query_shared_data, conn)
#             data = axes_normaliser(data, ['chest', 'left_ankle', 'right_lower_arm'], ['acc', 'gyro'], 1, 4)
#             # print(data)
#             data.to_pickle('mhealth_reformatted.pickle')
#             # print(data)
#             # data = drop_activities(bad_acc, data)
#             # data = resample(data, 100)
#             # data = deg2rad(data)
#             # data = data.drop(['timestamp'], axis = 1)
#             # sliding_windows_mhealth = preprocess.full_df_to_sliding_windows(data)
#             # sliding_windows_mhealth = uci_mhealth.to_sliding_windows_shared_data(conn)
#             # features_mhealth = extract_features(sliding_windows_mhealth, all_feature)
#             # features_mhealth.to_pickle('mhealth_features.pkl')
#         if os.path.exists('pamap_features.pkl'):
#             features_pamap = pd.read_pickle('pamap_features.pkl')
#         else:
#             data = pd.read_sql_query(uci_pamap2.raw_table_query_shared_data, conn)
#             data = axes_normaliser(data, ['chest', 'ankle', 'hand'], ['acc', 'gyro'], 3, 4)
#             print(data)
#             data.to_pickle('pamap_reformatted.pickle')
#
#             # # data = resample(data, 50)
#             # data = data.drop(['timestamp'], axis = 1)
#             # sliding_windows_pamap = preprocess.full_df_to_sliding_windows(data)
#             # # sliding_windows_pamap = uci_pamap2.to_sliding_windows_shared_data(conn)
#             # features_pamap = extract_features(sliding_windows_pamap, all_feature)
#             # features_pamap.to_pickle('pamap_features.pkl')
#
# if __name__ == "__main__":
#     # main()
#     cv_main()
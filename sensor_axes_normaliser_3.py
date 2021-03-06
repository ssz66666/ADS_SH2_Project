from sklearn.metrics import brier_score_loss

import pandas as pd
import sqlite3

from datetime import datetime as dt

from dataset_util import preprocess, uci_mhealth, uci_pamap2, uci_smartphone
from dataset_util.extract_input_features import all_feature, extract_features
# import matplotlib.pyplot as plt

import numpy as np
from config import SQLITE_DATABASE_FILE, TRAINING_SET_PROPORTION
import os
import random
from sensor_axes_normaliser import axes_normaliser_1
from resample import resample
from pyquaternion import Quaternion
from tqdm import tqdm


def axes_normaliser_2(data, regions, measurements):
    df = pd.DataFrame()
    df['activity_id'] = data['activity_id']
    df['subject_id'] = data['subject_id']
    subj_id = data['subject_id'].values
    subj_list = np.unique(subj_id)

    print(len(df))

    # orienatations = [['x', 'y', 'z'], ['x', 'z', 'y'], ['y', 'x', 'z'], ['y', 'z', 'x'], ['z', 'x', 'y'], ['z', 'y', 'x']]
    sin_quarter_pi = np.sin(np.pi/4.0)
    orientations = [
        np.array([1,0,0,0]),
        np.array([sin_quarter_pi,0,sin_quarter_pi,0]),
        np.array([0,0,1,0]),
        np.array([sin_quarter_pi,0,-sin_quarter_pi,0]),
        np.array([sin_quarter_pi,0,0,sin_quarter_pi]),
        np.array([0.5,0.5,0.5,0.5]),
        np.array([0,sin_quarter_pi,sin_quarter_pi,0]),
        np.array([0.5,-0.5,-0.5,0.5]),
        np.array([sin_quarter_pi,0,0,-sin_quarter_pi]),
        np.array([0.5,-0.5,0.5,-0.5]),
        np.array([0,-sin_quarter_pi,sin_quarter_pi,0]),
        np.array([0.5,0.5,-0.5,-0.5]),
        np.array([sin_quarter_pi,sin_quarter_pi,0,0]),
        np.array([0.5,0.5,0.5,-0.5]),
        np.array([0,0,sin_quarter_pi,-sin_quarter_pi]),
        np.array([0.5,0.5,-0.5,0.5]),
        np.array([0,1,0,0]),
        np.array([0,sin_quarter_pi,0,-sin_quarter_pi]),
        np.array([0,0,0,1]),
        np.array([0,sin_quarter_pi,0,sin_quarter_pi]),
        np.array([sin_quarter_pi,-sin_quarter_pi,0,0]),
        np.array([0.5,-0.5,0.5,0.5]),
        np.array([0,0,sin_quarter_pi,sin_quarter_pi]),
        np.array([0.5,-0.5,-0.5,-0.5]),
    ]
    rot_quaternions = [Quaternion(rot) for rot in orientations]

    for r in regions:
        list_x = []; list_y = []; list_z = []

        c = 0
        orientation_list = []
        params = []

        for subj in subj_list:
            temp = data.loc[data['subject_id'] == subj]
            # print(temp)
            for x in data.columns:
                if r in x and 'acc' in x:
                    # n = int(len(data)/10000)  # chunk row size
                    if 'x' in x:
                        list_x.append(temp[x].values)
                    if 'y' in x:
                        list_y.append(temp[x].values)
                    if 'z' in x:
                        list_z.append(data[x].values)

        rand = subj_list[random.randint(0, len(subj_list))]
        bins = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
        control_x = np.histogram(list_x[rand], bins=bins, density=True)
        control_x = [control_x[0][n]*(control_x[1][n+1]-control_x[1][n]) for n in range(len(control_x[0]))]

        control_y = np.histogram(list_y[rand], bins=bins, density=True)
        control_y = [control_y[0][n] * (control_y[1][n + 1] - control_y[1][n]) for n in range(len(control_y[0]))]

        control_z = np.histogram(list_z[rand], bins=bins, density=True)
        control_z = [control_z[0][n] * (control_z[1][n + 1] - control_z[1][n]) for n in range(len(control_z[0]))]

        # print(control_y)
        # print(control_z)
        for i in range(len(list_x)):
            if i != rand:
                c+=1

                test_x = np.histogram(list_x[i], bins=bins, density=True)
                test_x = np.array([test_x[0][n] * (test_x[1][n + 1] - test_x[1][n]) for n in range(len(test_x[0]))])

                test_y = np.histogram(list_y[i], bins=bins, density=True)
                test_y = np.array([test_y[0][n] * (test_y[1][n + 1] - test_y[1][n]) for n in range(len(test_y[0]))])

                test_z = np.histogram(list_z[i], bins=bins, density=True)
                test_z = np.array([test_z[0][n] * (test_z[1][n + 1] - test_z[1][n]) for n in range(len(test_z[0]))])

                # dist1 = best_orientation(control_x, control_y, control_z, test_x, test_y, test_z)
                # dist2 = best_orientation(control_x, control_y, control_z, test_x, test_z, test_y)
                # dist3 = best_orientation(control_x, control_y, control_z, test_y, test_x, test_z)
                # dist4 = best_orientation(control_x, control_y, control_z, test_y, test_z, test_x)
                # dist5 = best_orientation(control_x, control_y, control_z, test_z, test_x, test_y)
                # dist6 = best_orientation(control_x, control_y, control_z, test_z, test_y, test_x)
                test_vec = np.column_stack([test_x,test_y,test_z])
                distances = [best_orientation(control_x,control_y,control_z, *np.apply_along_axis(rot.rotate, 1, test_vec).T) for rot in rot_quaternions]
                # print(distances.index(min(np.array(distances))))
                orientation = rot_quaternions[distances.index(min(np.array(distances)))]
                # print(orientation)
                orientation_list.append(orientation)

            if i == rand:
                orientation_list.append(rot_quaternions[0])


        final_fixed_x = []; final_fixed_y = []; final_fixed_z = []

        print(len(orientation_list))
        print(orientation_list)
        for m in measurements:
            temp_x = []; temp_y = []; temp_z = []
            fixed_x = []; fixed_y = []; fixed_z = []
            for subj in subj_list:
                temp = data.loc[data['subject_id'] == subj]
                for x in data.columns:
                    if r in x and m in x:
                        n = int(len(data) / 10000)  # chunk row size
                        if m not in params:
                            params.append(m)
                        if '_x' in x:
                            temp_x.append(temp[x].values)
                        if '_y' in x:
                            temp_y.append(temp[x].values)
                        if '_z' in x:
                            temp_z.append(temp[x].values)

            if len(temp_x) > 0:
                print(len(temp_x), len(temp_y), len(temp_z))
                for q in tqdm(range(len(orientation_list))):
                    temp_vec = np.column_stack((temp_x[q],temp_y[q],temp_z[q]))
                    fixed_vec = np.apply_along_axis(orientation_list[q].rotate, 1, temp_vec)
                    fixed_x = np.concatenate((fixed_x,fixed_vec[:,0]))
                    fixed_y = np.concatenate((fixed_y,fixed_vec[:,1]))
                    fixed_z = np.concatenate((fixed_z,fixed_vec[:,2]))

                    # fixed_x = np.concatenate((fixed_x, eval('temp_' + orientation_list[q][0])[q]), axis=0)
                    # fixed_y = np.concatenate((fixed_y, eval('temp_' + orientation_list[q][1])[q]), axis=None)
                    # fixed_z = np.concatenate((fixed_z, eval('temp_' + orientation_list[q][2])[q]), axis=None)

                final_fixed_x.append(fixed_x); final_fixed_y.append(fixed_y); final_fixed_z.append(fixed_z)

        for i in ['x', 'y', 'z']:
            for j in range(len(params)):
                # print(len(eval('final_fixed_' + i)[j]))
                df[r + '_' + params[j] + '_' + i] = eval('final_fixed_' + i)[j]

    return df

def best_orientation(control_x, control_y, control_z, test_x, test_y, test_z):
    dist_x = brierScore(control_x, test_x)
    dist_y = brierScore(control_y, test_y)
    dist_z = brierScore(control_z, test_z)

    dist = dist_x+dist_y+dist_z

    return dist

def brierScore(preds, outcomes):
    n = float(len(preds))
    return 1 / n * np.sum((preds - outcomes)**2)

def deg2rad(data):
    for i in data.columns:
        if 'gyro' in i:
            temp = np.array(data[i].values)
            temp = [np.deg2rad(z) for z in temp]
            data[i] = temp

    return data

def cv_main():
    with sqlite3.connect(SQLITE_DATABASE_FILE) as conn:
        if os.path.exists('mhealth_features.pkl'):
            features_mhealth = pd.read_pickle('mhealth_features.pkl')
            raise NotImplementedError
            # features_mhealth = axes_normaliser_1(features_mhealth, ['chest', 'left_ankle', 'right_lower_arm'], ['acc', 'gyro'])
            # print(features_mhealth)
        else:
            data_mhealth = pd.read_sql_query(uci_mhealth.raw_table_query_shared_data, conn)
            # data_mhealth = resample(data_mhealth, 100)
            # data_mhealth.to_pickle('mhealth.pkl')
            data_mhealth = deg2rad(data_mhealth)
            data_mhealth = axes_normaliser_1(data_mhealth, ['chest'], ['chest'],  ['acc'], 1, 4, [])
            # data = axes_normaliser_2(data, ['chest', 'ankle', 'hand'], ['acc', 'gyro'])
            # print(data)
            # data_mhealth.to_pickle('mhealth_reformatted.pickle')
            # print(data)
            # data = drop_activities(bad_acc, data)
            # sliding_windows_mhealth = preprocess.full_df_to_sliding_windows(data)
            # sliding_windows_mhealth = uci_mhealth.to_sliding_windows_shared_data(conn)
            # features_mhealth = extract_features(sliding_windows_mhealth, all_feature)
            # features_mhealth.to_pickle('mhealth_features.pkl')
        if os.path.exists('pamap_features.pkl'):
            features_pamap = pd.read_pickle('pamap_features.pkl')
        else:
            data_pamap = pd.read_sql_query(uci_pamap2.raw_table_query_shared_data, conn)
            maps = [[1, 3], [3, 1], [5, 11], [6, 9]]
            data_pamap = axes_normaliser_1(data_pamap, ['chest'], ['chest'], ['acc'], 3, 4, maps)
            data_pamap.index = [x + len(data_mhealth) for x in data_pamap.index]
#             print(data)
#             data.to_pickle('pamap_reformatted.pickle')

        if os.path.exists('smartphone_features.pkl'):
            features_pamap = pd.read_pickle('smartphone_features.pkl')
        else:
            data_smartphone = pd.read_sql_query(uci_smartphone.raw_table_query_shared_data, conn)
            data_smartphone = uci_smartphone.small_g_to_m_per_sec_sq(data_smartphone)
            maps = [[1, 4], [2, 50], [3, 51], [4, 2], [5, 1], [6, 3]]
            subj_maps = np.unique(data_smartphone['subject_id'].values)
            data_smartphone = axes_normaliser_1(data_smartphone, ['body'], ['chest'], ['acc'], 5, 1, maps)
            print(data_smartphone.index)
            data_smartphone.index = [x + len(data_mhealth) + len(data_pamap) for x in data_smartphone.index]
#
#             # # data = resample(data, 50)
#             # data = data.drop(['timestamp'], axis = 1)
#             # sliding_windows_pamap = preprocess.full_df_to_sliding_windows(data)
#             # # sliding_windows_pamap = uci_pamap2.to_sliding_windows_shared_data(conn)
#             # features_pamap = extract_features(sliding_windows_pamap, all_feature)
#             # features_pamap.to_pickle('pamap_features.pkl')
    data = pd.concat(preprocess.remap_subject_ids([data_mhealth, data_pamap, data_smartphone]))
    data.to_pickle('data.pkl')
    data = axes_normaliser_2(data, ['chest'], ['acc'])
    data.to_pickle('fully_reformatted_3.pkl')
#
if __name__ == "__main__":
    # main()
    cv_main()
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:02:43 2019

@author: alexh
"""

#import tensorflow as tf 
#from tensorflow.contrib import rnn  
import pandas as pd  
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
#from imblearn.ensemble import BalancedRandomForestClassifier

# =============================================================================
# def func(dataset, epoch = 8, n_unit = 200, batch_size = 100):
#         n_class = len(train_y.value_counts(sort = False))
#         n_feature = len(train_X.columns)
#         
#         X_placeHolder = tf.placeholder("float", [None, n_feature])
#         y_placeHolder = tf.placeholder("float")
#         
#         layer ={'weights': tf.Variable(tf.random_normal([n_unit, n_class])), 
#                 'bias': tf.Variable(tf.random_normal([n_class]))}  
#         x = tf.split(X_placeHolder, n_feature, 1)
#         print(x)  
#         lstm_cell = rnn.BasicLSTMCell(n_unit)
#         outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#         output = tf.matmul(outputs[-1], layer['weights']) + layer['bias']
# =============================================================================


# =============================================================================
# dataset1  = pd.read_csv("imputedPAMAP2_5.csv")
# dataset2  = pd.read_csv("imputedPAMAP2_6.csv")
# dataset3  = pd.read_csv("imputedPAMAP2_7.csv")
# dataset4  = pd.read_csv("imputedPAMAP2_8.csv")
# dataset5  = pd.read_csv("imputedPAMAP2_2.csv")
# dataset6  = pd.read_csv("imputedPAMAP2_4.csv")
# dataset = pd.concat([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6])
# 
# del dataset['Unnamed: 0']
# del dataset['sex']
# del dataset['weight']
# del dataset['height']
# del dataset['hand']
# del dataset['restingHR']
# del dataset['age']
# del dataset['maxHR']
# del dataset['timeStamp']
# =============================================================================



def transformActivityID(inputDataset, activity_col = 0):
        y_label_set = list(set(inputDataset.iloc[:,activity_col]))
        for i in range(0, len(y_label_set)):
                idx_flag = inputDataset.iloc[:,activity_col] == y_label_set[i]
                inputDataset.iloc[:,activity_col][idx_flag] = i
        return inputDataset
        

def formDataset(inputDataset, activity_col = 0, timeSteps = 100, overlap = 50):
        activityID = inputDataset.iloc[:,activity_col]
        nextState = activityID.shift(-1)
        nextState.iloc[-1] = activityID.iloc[-1]
        shiftVal = activityID - nextState
        changePos = shiftVal.nonzero()
        changePos = changePos[0].tolist()
        changePos_list = [-1]
        changePos_list.extend(changePos)
        changePos_list.append(len(nextState))
        inputDataset = inputDataset.values
        
        allOutputData = []
        for index in range(0, (len(changePos_list) - 1)):
                start_idx = changePos_list[index] + 1
                stop_idx = changePos_list[index + 1]             
                
                currentOutputData = []
                for timeIdx in range(start_idx, (stop_idx - timeSteps), (timeSteps - overlap)):
                                tmpData = inputDataset[timeIdx:(timeIdx + timeSteps),:]
                                currentOutputData.append(tmpData)
                               
                allOutputData.extend(currentOutputData)
        
        allOutputData = np.dstack(allOutputData)
        allOutputData = np.swapaxes(allOutputData, 0, 2)
        allOutputData = np.swapaxes(allOutputData, 1, 2)
        colIdx = [True] * np.shape(allOutputData)[2]
        colIdx[activity_col] = False
        X_dataset = allOutputData[:,:,colIdx]
        
        y_label = allOutputData[:,0,activity_col]        
        y_label = to_categorical(y_label)
        
        return X_dataset, y_label
               
def eval_lstm(trainSet, testSet, activity_col = 0, timeSteps = 100, overlap = 50, epoch = 10, batch_size = 32):    
        scaler = MinMaxScaler()
        scaleCol = trainSet.columns.drop(trainSet.columns[activity_col]) 
        trainSet[scaleCol] = scaler.fit_transform(trainSet[scaleCol])
        testSet[scaleCol]  = scaler.transform(testSet[scaleCol])        
        
        train_X, train_y = formDataset(trainSet, activity_col = activity_col, timeSteps = timeSteps, overlap = overlap)
        test_X,  test_y  = formDataset(testSet,  activity_col = activity_col, timeSteps = timeSteps, overlap = overlap)
        
        n_timestep, n_feature, n_class = train_X.shape[1], train_X.shape[2], train_y.shape[1]
        
        model = Sequential()
#        model.add(LSTM(64, input_shape = (n_timestep, n_feature)))
        model.add(LSTM(64, input_shape = (n_timestep, n_feature), return_sequences = True))
        model.add(Dropout(0.5))
        model.add(LSTM(64))
#        model.add(Dropout(0.3))  
#        model.add(Dense(32, activation='relu'))
        model.add(Dense(n_class, activation = "softmax"))        
        model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["acc"])
        
#        history = model.fit(train_X, train_y, epochs = epoch, batch_size = batch_size, validation_data = (test_X, test_y))
        history = model.fit(train_X, train_y, epochs = epoch, batch_size = batch_size)
        predict_y = model.predict(test_X, batch_size = batch_size)
        
        output = np.argmax(predict_y, axis = 1)
        label = np.argmax(test_y, axis = 1)
        summary = pd.DataFrame({"label": label, "predict": output})
        
        perf = classification_report(label, output, output_dict = True)
        acc  = accuracy_score(label, output)
        
        return [acc, pref, summary]


def eval_rf(trainSet, testSet, activity_col, ntree = 500):
        train_y = trainSet.iloc[:,activity_col]
        test_y  = testSet.iloc[:,activity_col]
        
        featureCol = trainSet.columns.drop(trainSet.columns[activity_col]) 
        train_X = trainSet[featureCol]
        test_X  = testSet[featureCol] 
        
        clsf = RandomForestClassifier(n_estimators = ntree, class_weight="balanced", n_jobs = -1)
        clsf.fit(train_X, train_y)
        rf_pred = clsf.predict(test_X)
        
        perf = classification_report(test_y, rf_pred, output_dict = True)
        acc  = accuracy_score(test_y, rf_pred)
        summary = pd.DataFrame({"label": test_y, "predict": rf_pred})
        
        return [acc, perf, summary]

def transformPerf(perf_dict):    
        acc = 0
        for i in range(len(perf_dict)):
                subperf = perf_dict[i][1]
                perf_activity = pd.DataFrame()
                for j in list(subperf.keys()):
                        perf_activity_tmp = pd.DataFrame(subperf[j].items()).iloc[:,1]
                        perf_activity.loc[:,j] = perf_activity_tmp
                if i == 0:
                        out_perf = perf_activity
                else:
                        out_perf = out_perf.append(perf_activity)
                acc = acc + perf_dict[i][0]
        acc_ave = acc / (len(perf_dict))
        f1_score_ave = out_perf.loc[0].mean(axis = 0)
        precision_ave = out_perf.loc[1].mean(axis = 0)
        recall_ave = out_perf.loc[2].mean(axis = 0)
        output_ave = pd.DataFrame({"f1_score_ave": f1_score_ave, "precision_ave": precision_ave, "recall_ave": recall_ave})
        return [acc_ave, out_perf]

def eval_cv(inputDataset, subject_col = 0, activity_col = 1, method = "lstm", ntree = 500,
            timeSteps = 100, overlap = 50, epoch = 15, batch_size = 32, group = False,
            group_subj = None):
        
        activity_col_name = inputDataset.columns[activity_col]
        subject_col_name = inputDataset.columns[subject_col]
        print("subject col:", subject_col_name, ";", "activity col:", activity_col_name)              
        relabel = transformActivityID(inputDataset, activity_col = activity_col)
#        relabel = inputDataset
        if group:
                subj_len = max(list(map(len, group_subj)))
                group_subj = np.array(group_subj)
        else:
                group_subj = list(set(relabel.iloc[:, subject_col]))
                subj_len = len(group_subj)
        
        counter_cv = 1
        for SubjIdx_Idx in range(subj_len):
                print(counter_cv, "/", subj_len, ":")
                
                if group:
                        SubjIdx = group_subj[:, SubjIdx_Idx]
                        testSetIdx = relabel.iloc[:, subject_col].isin(SubjIdx)
                else:
                        SubjIdx = group_subj[SubjIdx_Idx]
                        testSetIdx = relabel.iloc[:, subject_col] == SubjIdx                
                
                testSetRaw = relabel.loc[testSetIdx,:]
                trainSetRaw = relabel.loc[-testSetIdx,:]
                print("Subjects in Validation/Test Set:", list(SubjIdx))
                newColIDX = trainSetRaw.columns.drop(trainSetRaw.columns[subject_col])
                trainSet = trainSetRaw.loc[:,newColIDX]
                testSet  = testSetRaw.loc[:,newColIDX]
                
                activityIDX = trainSet.columns.get_loc(activity_col_name)
                
                if method == "lstm":
                        result_fold = eval_lstm(trainSet = trainSet, testSet = testSet, 
                                                activity_col = activityIDX, timeSteps = timeSteps, 
                                                overlap = overlap, epoch = epoch, batch_size = batch_size)
                if method == "rf" :
                        result_fold = eval_rf(trainSet = trainSet, testSet = testSet, 
                                              activity_col = activityIDX, ntree = ntree)
                        
                if SubjIdx_Idx == 0:
                        out_perf = [result_fold]
                else:
                        out_perf.append(result_fold)
                counter_cv = counter_cv + 1
                     
        return out_perf

#lstm_cv_res = eval_cv(inputDataset = dataset, subject_col = 0, activity_col = 1, method = "lstm", 
#                      timeSteps = 100, overlap = 50, epoch = 10, batch_size = 32)
#
#
#rf_cv_res = eval_cv(inputDataset = dataset, subject_col = 0, activity_col = 1, 
#                    method = "rf", ntree = 500)


#### Combined Dataset ####

#inputDataset = pd.read_pickle("fully_reformatted_10000_split.pkl")

#inputDataset.columns.values
#list(set(inputDataset.iloc[:, 1]))
#list(set(testSetRaw.iloc[:, 1]))
#(combinedDataset.iloc[:, 0]).value_counts(sort = False)
#dataset_refined = inputDataset.loc[inputDataset.iloc[:,0].isin([1, 2, 3, 4, 9, 11]), :]
#
#def evaluateDataset(inputDataset, subject_col = 1, activity_col = 0, method = "lstm", ntree = 500,
#                    timeSteps = 100, overlap = 50, epoch = 15, batch_size = 32, groupCV = True,
#                    group_subj = [[1, 2, 4, 6, 7, 8, 9], [12, 13, 14, 15, 16, 17, 18]], 
#                    testSet_subj = [3, 5, 10, 11]):
#
#        testSet_idx = inputDataset.iloc[:,subject_col].isin(testSet_subj)
#        testSet = inputDataset.loc[testSet_idx,:]
#        dataset = inputDataset.loc[-testSet_idx,:]
#        
#        cv_res = eval_cv(inputDataset = dataset, subject_col = subject_col, 
#                         activity_col = activity_col, method = method, ntree = ntree,
#                         timeSteps = timeSteps, overlap = overlap, epoch = epoch, 
#                         batch_size = batch_size, group = group, group_subj = group_subj)
#
#        return 0
#
#inputDataset = pd.read_pickle("fully_reformatted_10000_split.pkl")
#dataset_refined = inputDataset.loc[inputDataset.iloc[:,0].isin([1, 2, 3, 4, 9, 11]), :]
#testSet_idx = dataset_refined.iloc[:,1].isin([3, 5, 10, 11])
#testSet = dataset_refined.loc[testSet_idx,:]
#dataset = dataset_refined.loc[-testSet_idx,:]
#lstm_cv_res_10000 = eval_cv(inputDataset = dataset, subject_col = 1, 
#                            activity_col = 0, method = "lstm",
#                            timeSteps = 100, overlap = 50, epoch = 15, 
#                            batch_size = 32, group = True, 
#                            group_subj = [[1, 2, 4, 6, 7, 8, 9], [12, 13, 14, 15, 16, 17, 18]],)
#
#inputDataset = pd.read_pickle("fully_reformatted_subject_split.pkl")
#dataset_refined = inputDataset.loc[inputDataset.iloc[:,0].isin([1, 2, 3, 4, 9, 11]), :]
#testSet_idx = dataset_refined.iloc[:,1].isin([3, 5, 10, 11])
#testSet = dataset_refined.loc[testSet_idx,:]
#dataset = dataset_refined.loc[-testSet_idx,:]
#lstm_cv_res_subject = eval_cv(inputDataset = dataset, subject_col = 1, 
#                              activity_col = 0, method = "lstm",
#                              timeSteps = 100, overlap = 50, epoch = 15, 
#                              batch_size = 32, group = True, 
#                              group_subj = [[1, 2, 4, 6, 7, 8, 9], [12, 13, 14, 15, 16, 17, 18]],)
#
#
#
#feature_dataset_10000 = pd.read_pickle("fully_reformatted_10000_split_features.pkl")
#list(set(feature_dataset_10000.iloc[:, 1]))
#
#testSet_idx = feature_dataset_10000.iloc[:,1].isin([3, 5, 10, 11])
#testSet = feature_dataset_10000.loc[testSet_idx,:]
#dataset = feature_dataset_10000.loc[-testSet_idx,:]
#
#rf_cv_res_10000 = eval_cv(inputDataset = dataset, subject_col = 1, 
#                              activity_col = 0, method = "rf",
#                              ntree = 500, group = True, 
#                              group_subj = [[1, 2, 4, 6, 7, 8, 9], [12, 13, 14, 15, 16, 17, 18]],)
#
#rf_cv_res_10000_ave = transformPerf(rf_cv_res_10000)


ntree_candidate = [100, 300, 1000, 1500]
for i in ntree_candidate:
        rf_cv_res_10000 = eval_cv(inputDataset = dataset, subject_col = 1, 
                              activity_col = 0, method = "rf",
                              ntree = i, group = True, 
                              group_subj = [[1, 2, 4, 6, 7, 8, 9], [12, 13, 14, 15, 16, 17, 18]],)
        
        res_ave = transformPerf(rf_cv_res_10000)
        
        print("subject ntree:", i, "acc: ", res_ave[0])
        



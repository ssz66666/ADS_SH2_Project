# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:33:46 2019

@author: alexh
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeavePGroupsOut
from sklearn import metrics
import gc
gc.collect()

subj1 = pd.read_csv("C:/Users/alexh/Desktop/PAMAP2_Dataset/Protocol/subject101.dat", sep = " ", header = None)
dataset = subj1.iloc[10200:11200,:]
del subj1
dataset["groups"] = np.repeat([1,2,3,4,5,6,7,8,9,10], 100)

RF_mod = RandomForestClassifier(n_estimators = 500, n_jobs = -1, class_weight = "balanced")
RF_mod.fit(X_train, y_train)

RF_pred = RF_mod.predict(X_test) 

def tmpFUN(dataset, group_label = "groups", n_groups = 2, y_label = "groups", rf_n_estimators = 2000, n_jobs = -1):
        lpgo = LeavePGroupsOut(n_groups = n_groups)

        for train_index, validate_index in lpgo.split(X = dataset, y = dataset.loc[:,y_label], groups = dataset.loc[:,group_label]):
                trainset = dataset.iloc[train_index,:]
                validateset  = dataset.iloc[validate_index,:]
                X_train = trainset.drop(y_label, axis = 1)
                y_train = trainset.loc[:,y_label]
                
                RF_mod = RandomForestClassifier(n_estimators = rf_n_estimators, n_jobs = n_jobs, class_weight = "balanced")
                RF_mod.fit(X_train, y_train)
                RF_pred = RF_mod.predict(X_test)
                
                
                
RF_mod = RandomForestClassifier(n_estimators = 1000, n_jobs = -1, class_weight = "balanced")
RF_mod.fit(X_train, y_train)
RF_pred = RF_mod.predict(X_test)         

                
                
                
                
                
                
                
                
                
                
                
                
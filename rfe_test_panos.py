from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime as dt

from dataset_util import uci_mhealth, preprocess
from dataset_util.extract_input_features import all_feature, extract_features

from sklearn.model_selection import train_test_split, GroupShuffleSplit
import matplotlib.pyplot as plt

from config import SQLITE_DATABASE_FILE
from evaluate_classification import evaluation_metrics
from scikitplot.metrics import plot_confusion_matrix, plot_roc

### ------------------------------------------------------------------------------- ###

# with sqlite3.connect(SQLITE_DATABASE_FILE) as conn:
# #     # features = pd.read_sql_query(uci_mhealth.raw_table_valid_data_query, conn)
# #     global sliding_windows, subject_ids, activity_ids
# #     sliding_windows = uci_mhealth.to_sliding_windows(conn, size=100, overlap=50)
# #     subject_ids = uci_mhealth.get_subject_ids(conn)
# #     activity_ids = uci_mhealth.get_activity_ids(conn)
# #     activity_ids.sort()

data = pd.read_pickle('fully_reformatted_3.pkl')
data = data.loc[data['activity_id'].isin([1, 2, 3, 4, 9, 11])]
sliding_windows = preprocess.full_df_to_sliding_windows(data, size = 100, overlap = 50)
subject_ids = np.unique(data['subject_id'].values)
# activity_ids = data['activity_id'].values
# activity_ids.sort()

### ------------------------------------------------------------------------------- ###
TRAINING_SET_PROPORTION = 0.7

features = extract_features(sliding_windows, all_feature)
print("features extracted")

X, y = uci_mhealth.to_classification(features)

# n_subs = len(subject_ids)
# n_training = round(n_subs * TRAINING_SET_PROPORTION)
# # n_test = n_subs - n_training
# idx = np.isin(features.loc[:,"subject_id"], subject_ids[:n_training])
# training_set = features[idx]
# test_set = features[np.logical_not(idx)]
# train_X, train_y = uci_mhealth.to_classification(training_set)
# test_X, test_y = uci_mhealth.to_classification(test_set)
# print("training set and test set ready")
# print("training set:", np.shape(train_X))
# print("test set:", np.shape(test_X))

clsf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1)
# clsf.fit(train_X,train_y)
# RF_pred = clsf.predict(test_X)
# pred_probability = clsf.predict_proba(test_X)

# evaluation_metrics(test_y,RF_pred, pred_probability)

# plot_confusion_matrix(test_y, RF_pred)
# plot_roc(test_y, pred_probability)
### ------------------------------------------------------------------------------- ###

# fts = zip(train_X.columns.values, clsf.feature_importances_)
# assert(len(train_X.columns.values) == len(clsf.feature_importances_))
# # list(train_X.columns.values).index('subject_id_mean')
# ft_list = list(fts)
# ft_list.sort(key = lambda x: x[1], reverse=True)
# print(np.array(ft_list))

from sklearn.feature_selection import RFE
selector = RFE(clsf, 10, step=1)
selector = selector.fit(X, y)

### ------------------------------------------------------------------------------- ###

print(selector.support_)
print(selector.ranking_)

### ------------------------------------------------------------------------------- ###

features.columns[2:][selector.support_]

### ------------------------------------------------------------------------------- ###

# n_subs = len(subject_ids)
# n_training = round(n_subs * TRAINING_SET_PROPORTION)
# # n_test = n_subs - n_training
# idx = np.isin(features.loc[:,"subject_id"], subject_ids[:n_training])
# training_set = features[idx]
# test_set = features[np.logical_not(idx)]
# train_X, train_y = uci_mhealth.to_classification(training_set)
# train_X_2 = train_X.iloc[:,selector.support_]
# test_X, test_y = uci_mhealth.to_classification(test_set)
# test_X_2 = test_X.iloc[:,selector.support_]
#
# print(np.size(train_X), np.size(train_y), np.size(train_X_2), np.size(test_X), np.size(test_y), np.size(test_X_2))

gss = GroupShuffleSplit(test_size=0.3, random_state=42)
X, y, groups = preprocess.to_classification(features)
y = pd.Series([int(n) for n in y.values])
# train_X, test_X, train_y, test_y = train_test_split(features, [int(n) for n in features['activity_id'].values], test_size=0.3, shuffle=False, random_state=42)
training_ind, test_ind = next(gss.split(X.values, groups=groups))
train_X = X.values[training_ind]
test_X = X.values[test_ind]
train_y = y.values[training_ind]
test_y = y.values[test_ind]

train_X_2 = train_X[:, selector.support_]
test_X_2 = test_X[:, selector.support_]



clsf_1 = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1)
clsf_1.fit(train_X,train_y)
RF_pred = clsf_1.predict(test_X)
pred_probability = clsf_1.predict_proba(test_X)

evaluation_metrics(test_y,RF_pred, pred_probability)

_now = dt.now().strftime("%Y%m%d-%H-%M-%S")
plot_confusion_matrix(test_y, RF_pred)
plt.savefig("{}_result_before_rfe.png".format(_now))
plot_roc(test_y, pred_probability)
plt.savefig("{}_roc_result_before_rfe.png".format(_now))

clsf_2 = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1)
clsf_2.fit(train_X_2,train_y)
RF_pred = clsf_2.predict(test_X_2)
pred_probability = clsf_2.predict_proba(test_X_2)

evaluation_metrics(test_y,RF_pred, pred_probability)

#plot_confusion_matrix(test_y, RF_pred)
#plot_roc(test_y, pred_probability)

# make plots
_now = dt.now().strftime("%Y%m%d-%H-%M-%S")
plot_confusion_matrix(test_y, RF_pred)
plt.savefig("{}_result_after_rfe.png".format(_now))
plot_roc(test_y, pred_probability)
plt.savefig("{}_roc_result_after_rfe.png".format(_now))
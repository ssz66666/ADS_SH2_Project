from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection
import pandas as pd
import sqlite3
import sys

from datetime import datetime as dt

from dataset_util import preprocess, uci_mhealth
from dataset_util.extract_input_features import all_feature, extract_features
import matplotlib.pyplot as plt

import numpy as np
from config import SQLITE_DATABASE_FILE, TRAINING_SET_PROPORTION
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from evaluate_classification import evaluation_metrics
from cross_validation import cross_validate_multiclass_group_kfold


def cv_main():
    with sqlite3.connect(SQLITE_DATABASE_FILE) as conn:
        # features = pd.read_sql_query(uci_mhealth.raw_table_valid_data_query, conn)
        sliding_windows = uci_mhealth.to_sliding_windows(conn)
    features = extract_features(sliding_windows, all_feature)
    clsf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1)

    result = cross_validate_multiclass_group_kfold(clsf, *preprocess.to_classification(features), n_splits=5)
    mean_result = {m: np.mean(v) for (m, v) in result.items()}
    for p in mean_result.items():
        print(p)


def main():
    # with sqlite3.connect(SQLITE_DATABASE_FILE) as conn:
    #     # features = pd.read_sql_query(uci_mhealth.raw_table_valid_data_query, conn)
    #     sliding_windows = uci_mhealth.to_sliding_windows(conn)
    #     subject_ids = uci_mhealth.get_subject_ids(conn)
    data = pd.read_pickle('fully_reformatted_3.pkl')
    data = data.loc[data['activity_id'].isin([1, 2, 3, 4, 9, 11])]
    sliding_windows = preprocess.full_df_to_sliding_windows(data, size=100, overlap=50)
    features = extract_features(sliding_windows, all_feature)

    from sklearn.model_selection import train_test_split, GroupShuffleSplit

    # train_X, train_y = uci_mhealth.to_classification(training_set)
    # test_X, test_y = uci_mhealth.to_classification(test_set)

    # train_y = [int(n) for n in train_y]
    # test_y = [int(n) for n in test_y]



    np.set_printoptions(threshold=sys.maxsize)
    # print(train_y)
    # print(features['activity_id'].values)
    gss = GroupShuffleSplit(test_size=0.3, random_state=42)
    X, y, groups = preprocess.to_classification(features)
    y = pd.Series([int(n) for n in y.values])
    # train_X, test_X, train_y, test_y = train_test_split(features, [int(n) for n in features['activity_id'].values], test_size=0.3, shuffle=False, random_state=42)
    training_ind, test_ind = next(gss.split(X.values, groups=groups))
    train_X = X.values[training_ind]
    test_X = X.values[test_ind]
    train_y = y.values[training_ind]
    test_y = y.values[test_ind]
    # print(train_y)

    # print('train x: ', train_X)
    # print('train y: ', train_y)
    # print('test X: ', test_X)
    # print('test y: ', test_y)

    # print("training set:", np.shape(train_X))
    # print("test set:", np.shape(test_X))

    clsf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1)
    clsf.fit(train_X, train_y)
    RF_pred = clsf.predict(test_X)
    pred_probability = clsf.predict_proba(test_X)

    evaluation_metrics(test_y, RF_pred, pred_probability)
    # np.savetxt("predicts.txt", RF_pred)
    # print(activity_ids)

    # make plots
    _now = dt.now().strftime("%Y%m%d-%H-%M-%S")
    plot_confusion_matrix(test_y, RF_pred, normalize=True)
    plt.savefig("{}_result.png".format(_now))
    plot_roc(test_y, pred_probability)
    plt.savefig("{}_roc_result.png".format(_now))


# testset = sklearn.model_selection.train_test_split(dataset, test_size = 0.2, random_state = 1, stratify = dataset.iloc[:,1])

if __name__ == "__main__":
    main()
    # cv_main()

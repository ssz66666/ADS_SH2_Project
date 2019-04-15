from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection
import pandas as pd
import sqlite3

from datetime import datetime as dt

from dataset_util import uci_mhealth
from dataset_util.extract_input_features import all_feature, extract_features
import matplotlib.pyplot as plt

import numpy as np
from config import SQLITE_DATABASE_FILE, TRAINING_SET_PROPORTION
from plots import plot_confusion_matrix
from evaluate_classification import evaluation_metrics


def main():
    with sqlite3.connect(SQLITE_DATABASE_FILE) as conn:
        # features = pd.read_sql_query(uci_mhealth.raw_table_valid_data_query, conn)
        sliding_windows = uci_mhealth.to_sliding_windows(conn)
        subject_ids = uci_mhealth.get_subject_ids(conn)
        activity_ids = uci_mhealth.get_activity_ids(conn)
    features = extract_features(sliding_windows, all_feature)
    n_subs = len(subject_ids)
    n_training = round(n_subs * TRAINING_SET_PROPORTION)
    # n_test = n_subs - n_training
    idx = np.isin(features.loc[:,"subject_id"], subject_ids[:n_training])

    training_set = features[idx]
    test_set = features[np.logical_not(idx)]
    train_X, train_y = uci_mhealth.to_classification(training_set)
    test_X, test_y = uci_mhealth.to_classification(test_set)
    #print("training set:", np.shape(train_X))
    #print("test set:", np.shape(test_X))

    clsf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1)
    clsf.fit(train_X,train_y)
    RF_pred = clsf.predict(test_X)
    test_y = list(map(lambda x: int(x), test_y))
    RF_pred = list(map(lambda x: int(x), RF_pred))

    evaluation_metrics(test_y,RF_pred)

    # np.savetxt("predicts.txt", RF_pred)
    #print(activity_ids)
    activity_ids.sort()
    activity_ids = list(map(lambda x: str(x), activity_ids))[1:]
    plot_confusion_matrix(test_y, RF_pred, activity_ids)
    plt.savefig("{}_result.png".format(dt.now().strftime("%Y%m%d-%H-%M-%S")))

# testset = sklearn.model_selection.train_test_split(dataset, test_size = 0.2, random_state = 1, stratify = dataset.iloc[:,1])
        
if __name__ == "__main__":
    main()

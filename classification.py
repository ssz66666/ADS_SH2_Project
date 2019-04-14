from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection
from sklearn.metrics import confusion_matrix
import sklearn.utils.multiclass
import pandas as pd
import sqlite3

from datetime import datetime as dt

from dataset_util import uci_mhealth
from dataset_util.extract_input_features import all_feature, extract_features
# import pickle
import matplotlib.pyplot as plt

import numpy as np
from config import SQLITE_DATABASE_FILE

TRAINING_SET_PROPORTION = 0.7

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[sklearn.utils.multiclass.unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def main():
    with sqlite3.connect(SQLITE_DATABASE_FILE) as conn:
        sliding_windows = uci_mhealth.to_sliding_windows(conn)
        subject_ids = uci_mhealth.get_subject_ids(conn)
        activity_ids = uci_mhealth.get_activity_ids(conn)
    if activity_ids is None:
        raise ValueError("WTF")
    features = extract_features(sliding_windows, all_feature)
    n_subs = len(subject_ids)
    n_training = round(n_subs * TRAINING_SET_PROPORTION)
    n_test = n_subs - n_training
    idx = np.isin(features.loc[:,"subject_id"], subject_ids[:n_training])
    print("Moved on")
    training_set = features[idx]
    test_set = features[np.logical_not(idx)]
    train_X, train_y = uci_mhealth.to_classification(training_set)
    test_X, test_y = uci_mhealth.to_classification(test_set)
    print("training set:", np.shape(train_X))
    print("test set:", np.shape(test_X))

    clsf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1)
    clsf.fit(train_X,train_y)
    RF_pred = clsf.predict(test_X)
    test_y = list(map(lambda x: int(x), test_y))
    RF_pred = list(map(lambda x: int(x), RF_pred))
    np.savetxt("predicts.txt", RF_pred)
    print(activity_ids)
    activity_ids.sort()
    activity_ids = list(map(lambda x: str(x), activity_ids))[1:]
    plot_confusion_matrix(test_y, RF_pred, activity_ids)
    plt.savefig("{}_result.png".format(dt.now().strftime("%Y%m%d-%H-%M-%S")))

# testset = sklearn.model_selection.train_test_split(dataset, test_size = 0.2, random_state = 1, stratify = dataset.iloc[:,1])
        
if __name__ == "__main__":
    main()

from evaluate_classification import get_cv_scorer
from sklearn.model_selection import cross_validate, GroupKFold
import dataset_util.preprocess as preprocess
import numpy as np

# we use ``Group K-Fold'', where data from one subject are considered from one group here
# K defaults to 5
# data:
# first column should be class label, second should be subject ID,
# remaining columns are features
def cross_validate_multiclass_group_kfold(clf, X, y, subject_id, n_splits=5, **kwargs):
    group_kfold = GroupKFold(n_splits)
    classes = np.unique(y)
    return cross_validate(clf, X, y, groups=subject_id, scoring=get_cv_scorer(classes), cv=group_kfold, n_jobs=-1, **kwargs)


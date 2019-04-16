from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc
import sklearn.utils.multiclass
import numpy as np
from scipy import interp

def calculate_accuracy(targets,predictions):

    return accuracy_score(targets,predictions)

def calculate_f1_score(targets,predictions):
    return f1_score(targets,predictions,average='weighted')

def calculate_sensitivity_specificity(targets,predictions,classes):
    targets_2D = label_binarize(targets,classes = classes)
    predictions_2D = label_binarize(predictions, classes=classes)
    sensitivity_dict = {}
    specificity_dict = {}

    for (idx, klas) in enumerate(classes):

        cm = confusion_matrix(targets_2D[:,idx],predictions_2D[:,idx])
        sensitivity = 100*cm[0,0]/(cm[0,0]+cm[0,1])
        specificity = 100*cm[1, 1] / (cm[1, 0] + cm[1, 1])

        sensitivity_dict[klas] = sensitivity
        specificity_dict[klas] = specificity

    return sensitivity_dict, specificity_dict

# taken from https://github.com/reiinakano/scikit-plot/blob/2dd3e6a76df77edcbd724c4db25575f70abb57cb/scikitplot/metrics.py#L332
#   
# MIT License

# Copyright (c) [2018] [Reiichiro Nakano]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
def get_roc_auc(y_true, pred_proba):
    classes = np.unique(y_true)
    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}
    for (i, klas) in enumerate(classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true, pred_proba[:, i],
            pos_label=klas)
        auc_dict[klas] = auc(fpr_dict[i], tpr_dict[i])

    # compute micro-average AUC
    binarized_y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        binarized_y_true = np.hstack(
            (1 - binarized_y_true, binarized_y_true))
    fpr, tpr, _ = roc_curve(binarized_y_true.ravel(), pred_proba.ravel())
    micro_auc = auc(fpr, tpr)

    # compute macro-average AUC
    all_fpr = np.unique(np.concatenate([fpr_dict[x] for x in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)
    macro_auc = auc(all_fpr, mean_tpr)
    
    return auc_dict, micro_auc, macro_auc

def evaluation_metrics(targets, predictions, pred_probability, print_result=True):

    accuracy = calculate_accuracy(targets,predictions)*100
    f1_score = calculate_f1_score(targets,predictions)*100

    classes = sklearn.utils.multiclass.unique_labels(targets, predictions)
    sensitivity, specificity = calculate_sensitivity_specificity(targets,predictions,classes)

    auc_dict, micro_auc, macro_auc = get_roc_auc(targets, pred_probability)

    if print_result:
        print()
        print("classifier result")
        print("-"*50)
        for klas in classes:
            print("activity ", klas)
            print("sensitivity:\t", sensitivity[klas])
            print("specificity:\t", specificity[klas])
            print("auc:\t", auc_dict[klas])
            print("-"*50)

        print("f1_score ",f1_score)
        print("accuracy ",accuracy)
        print("micro-average ROC AUC:\t", micro_auc)
        print("macro-average ROC AUC:\t", macro_auc)

    return {
        "accuracy" : accuracy,
        "f1_score" : f1_score,
        "sensitivity" : sensitivity,
        "specificity" : specificity,
        "auc" : auc_dict,
        "micro_avg_auc" : micro_auc,
        "macro_avg_auc" : macro_auc,
    }

    
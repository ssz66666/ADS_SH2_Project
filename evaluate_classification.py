from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

def calculate_accuracy(targets,predictions):

    return accuracy_score(targets,predictions)

def calculate_f1_score(targets,predictions):
    return f1_score(targets,predictions,average='weighted')

def calculate_sensitivity_specificity(targets,predictions):
    classes = [1,2,3,4,5,6,7,8,9,10,11,12]

    targets_2D = label_binarize(targets,classes = classes)
    predictions_2D = label_binarize(predictions, classes=classes)
    results = []

    for i in range(0,len(classes)):

        cm = confusion_matrix(targets_2D[:,i],predictions_2D[:,i])
        sensitivity = 100*cm[0,0]/(cm[0,0]+cm[0,1])
        specificity = 100*cm[1, 1] / (cm[1, 0] + cm[1, 1])

        results.append(sensitivity)
        results.append(specificity)

    return results

def evaluation_metrics(targets,predictions):

    accuracy = calculate_accuracy(targets,predictions)*100
    f1_score = calculate_f1_score(targets,predictions)*100

    sensitivity_specificity = calculate_sensitivity_specificity(targets,predictions)

    for i in range(0, len(sensitivity_specificity),2):
        print("sensitivity ",sensitivity_specificity[i])
        print("specificity ",sensitivity_specificity[i+1])
        print("-"*50)

    print("f1_score ",f1_score)
    print("accuracy ",accuracy)
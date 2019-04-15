from sklearn.metrics import accuracy_score, f1_score


def calculate_accuracy(targets,predictions):

    return accuracy_score(targets,predictions)

def calculate_f1_score(targets,predictions):
    return f1_score(targets,predictions,average='weighted')


def evaluation_metrics(targets,predictions):

    accuracy = calculate_accuracy(targets,predictions)*100
    f1_score = calculate_f1_score(targets,predictions)*100

    print("f1_score ",f1_score)
    print("accuracy ",accuracy)

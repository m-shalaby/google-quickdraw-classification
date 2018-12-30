import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.externals.joblib import parallel_backend

from neural_network_sk_comp import CustomNNClassifier

def baseline_SVC(train, train_class, test, original=False):
    """
    Baseline classifier using SVC
    """

    parameters = {'C':np.logspace(-3, -1, 5), 'gamma':[1, 0.1], 'kernel':['linear']}

    grid = GridSearchCV(SVC(), parameters, refit=True, cv=3, verbose=5, return_train_score=True)

    with parallel_backend('threading'):
        grid.fit(train, train_class)

    best_parameters = grid.best_params_

    pred = grid.predict(test)

    filename = "Results\\baseline_SVC_performance"
    if(original):
        filename+= "_original"

    write_baselines(filename, "SVC", grid)

    print("Best score for best parameters:")
    print(grid.best_score_)
    print(grid.best_params_)

    return pred
  
def baseline_logistic_regression(train, train_class, test, original=False):
    """
    Baseline classifier using logistic regression with gridsearch
    """

    parameters = {'penalty':['l2'], 'C': np.logspace(-3, 0, 20)} 

    grid = GridSearchCV(LogisticRegression(), parameters, refit=True, cv=3, verbose=5, return_train_score=True)

    with parallel_backend('threading'):
        grid.fit(train, train_class)

    best_parameters = grid.best_params_

    pred = grid.predict(test)

    filename = "Results\\baseline_logistic_regression_performance"
    if(original):
        filename+= "_original"

    write_baselines(filename, "Logistic Regression", grid)

    print("Best score for best parameters:")
    print(grid.best_score_)
    print(grid.best_params_)

    return pred

def baseline_custom_NN(train, train_class, test):

    parameters = {"batch_size": [10, 100],
                "node_per_layer": [1000, 500],
                "layer_count": [1, 2], 
                "learning_rate": [1000, 500],
                "epoch": [15]}

    

    grid = GridSearchCV(CustomNNClassifier(), parameters, refit=True, cv=3, verbose=5, return_train_score=True)

    with parallel_backend('threading'):
        grid.fit(train, train_class)

    best_parameters = grid.best_params_

    filename = "Results\\baseline_custom_NN_performance"
    write_baselines(filename, "Custom Neural Network", grid)

    print("Best score for best parameters:")
    print(grid.best_score_)
    print(grid.best_params_)

    pred = grid.predict(test)
    return pred

def write_baselines(filename, name_of_baseline, grid):
    """
    Create a report for the cross validation on the specified baseline
    """
    detailed_filename = filename + "_details.txt"
    filename = filename + ".txt"
    with open(filename, "w") as file:
        file.write("Baseline cross validation for %s \n" % name_of_baseline)
        file.write("Best parameters: " + str(grid.best_params_) + "\n")
        file.write("Best score: " + str(grid.best_score_) + "\n")

    with open(detailed_filename, "w") as file:
        file.write("Detailed performance of cross validation for %s \n" % name_of_baseline)
        pd.DataFrame(grid.cv_results_).to_csv(file)
    
   
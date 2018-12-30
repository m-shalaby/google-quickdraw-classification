from sklearn.model_selection import cross_validate

#   defines library references to the tested classifiers

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

xTrainData = [None] * 10000
yTrainData = [None] * 10000

with open('train_features.txt') as xData:

    pixelsVector = [None] * 10000

    for image in range(10000):

        values = xData.readline().split()
        for index in range(10000): pixelsVector[index] = float(values[index])
        xTrainData[image] = list(pixelsVector)

with open('train_labels.csv') as yData:
    
    headerLine = yData.readline()
    for index in range(10000): yTrainData[index] = yData.readline().split(',')[1].replace('\n', '')

#   constructs a tuple of classifier descriptions and their respective constructors

classifiers = [#('AdaBoost Classifier', AdaBoostClassifier()),
               #('Random Forest Classifier', RandomForestClassifier(n_jobs = -1)),
               #('QDA Classifier', QuadraticDiscriminantAnalysis()),
               #('Gaussian Naive Bayes Classifer', GaussianNB()),
               #('K Nearest Neighbours Classifier', KNeighborsClassifier(n_jobs = -1)),
               #('Multi Layer Perceptron Classifier', MLPClassifier()),
               #('Linear Support Vector Classification', LinearSVC()),
               #('Support Vector Classification', SVC()),
               #('Decision Tree Classifier', DecisionTreeClassifier()),
               #('Gaussian Process Classifier', GaussianProcessClassifier(n_jobs = -1)),
               ('Bernoulli Naive Bayes Classifier', BernoulliNB())]

for (classifierName, classifier) in classifiers:

    results = cross_validate(classifier, xTrainData, yTrainData, cv = 3, return_estimator = True)
    print('Classification Scores for ' + classifierName + ':')
    print(results['test_score'])
    maxScoreIndex = 0
    if results['test_score'][1] > results['test_score'][maxScoreIndex]: maxScoreIndex = 1
    if results['test_score'][2] > results['test_score'][maxScoreIndex]: maxScoreIndex = 2
    estimator = results['estimator'][maxScoreIndex]
    outputFileName = classifierName + '.txt'

    with open('test_features.txt') as testData:

        with open(outputFileName, 'a') as file: file.write('Id,Category\n')
        pixelsVector = [None] * 10000

        for image in range(10000):

            values = testData.readline().split()
            for index in range(10000): pixelsVector[index] = float(values[index])
            with open(outputFileName, 'a') as file: file.write(str(image) + ',' + str(estimator.predict([pixelsVector])[0]) +'\n')

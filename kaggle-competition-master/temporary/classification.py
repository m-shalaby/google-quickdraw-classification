from sklearn import model_selection
from sklearn.metrics import f1_score

#   defines library references to the tested classifiers

from sklearn.neural_network import MLPClassifier

xTrainData = [None] * 10000
yTrainData = [None] * 10000

with open('train_images_data_x.bin') as xData:

    pixelsVector = [None] * 10000

    for index in range(10000):

        for offset in range(1250):

            byteValue = ord(xData.read(1))
            for i in range(8): pixelsVector[8 * offset + i] = 1.0 if byteValue & (1 << (7 - i)) != 0 else 0.0

        xTrainData[index] = pixelsVector
        lineFeedCharacter = xData.read(1)

with open('train_labels.csv') as yData:
    
    headerLine = yData.readline()
    for index in range(10000): yTrainData[index] = yData.readline().split(',')[1].replace('\n', '')

(xTrain, xTest, yTrain, yTest) = model_selection.train_test_split(xTrainData, yTrainData, random_state = 0xFF)

#   constructs a tuple of classifier descriptions and their respective constructors

classifiers = [('Multiple Layer Perceptron', MLPClassifier())]

for (classifierName, classifier) in classifiers:

    classifier.fit(xTrain, yTrain)
    print('Classification Score for ' + classifierName + ': ' + str(f1_score(yTest, classifier.predict(xTest), average = 'micro')))

from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
import numpy as np

indexForLabelName = {'empty'       : 0x00, 'sink'       : 0x01, 'pear'      : 0x02, 'moustache' : 0x03,
                     'nose'        : 0x04, 'skateboard' : 0x05, 'penguin'   : 0x06, 'peanut'    : 0x07,
                     'skull'       : 0x08, 'panda'      : 0x09, 'paintbrush': 0x0A, 'nail'      : 0x0B,
                     'apple'       : 0x0C, 'rifle'      : 0x0D, 'mug'       : 0x0E, 'sailboat'  : 0x0F,
                     'pineapple'   : 0x10, 'spoon'      : 0x11, 'rabbit'    : 0x12, 'shovel'    : 0x13,
                     'rollerskates': 0x14, 'screwdriver': 0x15, 'scorpion'  : 0x16, 'rhinoceros': 0x17,
                     'pool'        : 0x18, 'octagon'    : 0x19, 'pillow'    : 0x1A, 'parrot'    : 0x1B,
                     'squiggle'    : 0x1C, 'mouth'      : 0x1D, 'pencil'    : 0x1E}

labelNameForIndex = ['empty'       , 'sink'       , 'pear'      , 'moustache' ,
                     'nose'        , 'skateboard' , 'penguin'   , 'peanut'    ,
                     'skull'       , 'panda'      , 'paintbrush', 'nail'      ,
                     'apple'       , 'rifle'      , 'mug'       , 'sailboat'  ,
                     'pineapple'   , 'spoon'      , 'rabbit'    , 'shovel'    ,
                     'rollerskates', 'screwdriver', 'scorpion'  , 'rhinoceros',
                     'pool'        , 'octagon'    , 'pillow'    , 'parrot'    ,
                     'squiggle'    , 'mouth'      , 'pencil']

def generateAutoEncoder(cleanImages, noisyImages):

    inputLayer = Input(shape = cleanImages[0].shape)
    autoEncoder = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputLayer)
    autoEncoder = MaxPooling2D(padding = 'same')(autoEncoder)
    autoEncoder = Conv2D(64, 3, activation = 'relu', padding = 'same')(autoEncoder)
    autoEncoder = MaxPooling2D(padding = 'same')(autoEncoder)
    autoEncoder = Conv2D(64, 3, activation = 'relu', padding = 'same')(autoEncoder)
    autoEncoder = UpSampling2D()(autoEncoder)
    autoEncoder = Conv2D(64, 3, activation = 'relu', padding = 'same')(autoEncoder)
    autoEncoder = UpSampling2D()(autoEncoder)
    autoEncoder = Conv2D(1, 3, activation = 'sigmoid', padding = 'same')(autoEncoder)
    autoEncoder = Model(inputLayer, autoEncoder)
    autoEncoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
    autoEncoder.fit(noisyImages, cleanImages,
                    epochs = 0x0400, batch_size = 0x80,
                    validation_split = 0.2,
                    callbacks = [EarlyStopping(patience = 0x08)])
    return autoEncoder

def generateLabels(filePath, classesCount, labels):

    with open(filePath, 'w', newline = '\n') as output:

        index = 0
        output.write('Id,Category\n')

        for label in labels:

            column = 0
            for i in range(len(label)): column = i if label[i] > label[column] else column
            output.write(str(index) + ',' + labelNameForIndex[column] + '\n')
            index += 1

def generateNetworkModel(inputShape, classesCount):

    networkModel = Sequential()
    networkModel.add(Conv2D(64, 3, activation = 'relu', padding = 'same', input_shape = inputShape))
    networkModel.add(Conv2D(64, 3, activation = 'relu', padding = 'same'))
    networkModel.add(MaxPooling2D(padding = 'same'))
    networkModel.add(Conv2D(128, 3, activation = 'relu', padding = 'same'))
    networkModel.add(Conv2D(128, 3, activation = 'relu', padding = 'same'))
    networkModel.add(MaxPooling2D(padding = 'same'))
    networkModel.add(Conv2D(256, 3, activation = 'relu', padding = 'same'))
    networkModel.add(Conv2D(256, 3, activation = 'relu', padding = 'same'))
    networkModel.add(Conv2D(256, 3, activation = 'relu', padding = 'same'))
    networkModel.add(Conv2D(256, 3, activation = 'relu', padding = 'same'))
    networkModel.add(MaxPooling2D(padding = 'same'))
    networkModel.add(Conv2D(512, 3, activation = 'relu', padding = 'same'))
    networkModel.add(Conv2D(512, 3, activation = 'relu', padding = 'same'))
    networkModel.add(Conv2D(512, 3, activation = 'relu', padding = 'same'))
    networkModel.add(Conv2D(512, 3, activation = 'relu', padding = 'same'))
    networkModel.add(MaxPooling2D(padding = 'same'))
    networkModel.add(Conv2D(512, 3, activation = 'relu', padding = 'same'))
    networkModel.add(Conv2D(512, 3, activation = 'relu', padding = 'same'))
    networkModel.add(Conv2D(512, 3, activation = 'relu', padding = 'same'))
    networkModel.add(Conv2D(512, 3, activation = 'relu', padding = 'same'))
    networkModel.add(MaxPooling2D(padding = 'same'))
    networkModel.add(Flatten())
    networkModel.add(Dense(0x1000, activation = 'relu'))
    networkModel.add(Dropout(0.5))
    networkModel.add(Dense(0x1000, activation = 'relu'))
    networkModel.add(Dropout(0.5))
    networkModel.add(Dense(classesCount, activation = 'softmax'))
    networkModel.compile(loss = keras.losses.categorical_crossentropy,
                         optimizer = keras.optimizers.SGD(nesterov = True),
                         metrics = ['accuracy'])
    return networkModel
    
def loadFeatures(filePath, featuresShape):

    data = np.load(filePath, encoding = 'latin1')
    features = [None] * len(data)
    for i in range(len(data)): features[i] = data[i][1]
    features = np.array(features).clip(min = 0x00, max = 0xFF).astype('float16') / 0xFF
    return features.reshape((features.shape[0], ) + featuresShape)

def loadLabels(filePath, classesCount):

    with open(filePath) as data:

        _ = data.readline()
        labels = []
        for line in data: labels.append(indexForLabelName[line.strip().split(',')[1]])
        return to_categorical(labels, classesCount)

#
#
#

#   performs training of denoising auto encoder on prepared clean and noisy images

cleanImages = loadFeatures('clean_images.npy', (100, 100, 1))
noisyImages = loadFeatures('noisy_images.npy', (100, 100, 1))
autoEncoder = generateAutoEncoder(cleanImages, noisyImages)

#   loads training data transformed using denoising auto encoder

trainingFeatures = autoEncoder.predict(loadFeatures('train_images.npy', (100, 100, 1)))
trainingLabels = loadLabels('train_labels.csv', 0x1F)

#   splits loaded data into separate training and testing sets

trainingFeatures, testingFeatures, trainingLabels, testingLabels = train_test_split(trainingFeatures,
                                                                                    trainingLabels,
                                                                                    random_state = 0xFF)

#   compiles the VGG19 model (may require resizing input shapes to (224, 224, 3))

model = generateNetworkModel((100, 100, 1), 0x1F)

#   constructs a batch image feeder that randomly applies horizontal flips to training features

dataGenerator = ImageDataGenerator(horizontal_flip=True)
dataGenerator.fit(trainingFeatures)

#   trains VGG19 model using processed training data

model.fit_generator(dataGenerator.flow(trainingFeatures, trainingLabels, batch_size = 0x80),
                    epochs = 0x0400, validation_data = (testingFeatures, testingLabels),
                    steps_per_epoch = 0x80, callbacks = [EarlyStopping(patience = 0x08)])

#   performs predictions on test images and outputs results

testingFeatures = autoEncoder.predict(loadFeatures('test_images.npy', (100, 100, 1)))
testingLabels = model.predict(testingFeatures)
generateLabels('test_labels.csv', 0x1F, testingLabels)

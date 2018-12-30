import numpy as np
import pandas as pd

import data
import baseline
import preprocessing
from neural_network import NeuralNetwork

from category_enum import CategoryEnum, get_category_index

def main():
    # Convert the txt from Yi Feng to npy to speed up loading of datasets 

    # train_images = data.load_reduced_features('Datasets\\train_features_arranged.txt', 3025)
    # test_images =  data.load_reduced_features('Datasets\\test_features_arranged.txt', 3025)
    # np.save('Datasets\\train_features_55x55', train_images)
    # np.save('Datasets\\test_features_55x55', test_images)

    test_images = np.load('Datasets\\test_features_55x55.npy')
    train_images = np.load('Datasets\\train_features_55x55.npy')

    train_labels = data.load_categories('Datasets\\train_labels.csv')

    #
    # Perform baseline evaluation on reduced images
    #

    baseline_evaluation(train_images, train_labels, test_images, original=False)

    #
    # Perform baseline evaluation on original images
    #

    train_images_original = data.load_image_array('Datasets\\train_images.npy')
    test_images_original = data.load_image_array('Datasets\\test_images.npy')

    baseline_evaluation(train_images_original, train_labels, test_images_original, original=True)

    #
    # Perform basic neural network hyperparameter optimization and obtain prediction from best model
    #

    test_images = np.load('Datasets\\test_images_28x28.npy')
    train_images = np.load('Datasets\\train_images_28x28.npy')


    baseline_NN(train_images, train_labels, test_images)

    #
    # Run best CNN model on reduced images for the best performance
    #


def baseline_evaluation(train_images, train_labels, test_images, original=False):
    
    pred = baseline.baseline_SVC(train_images, train_labels, test_images, original=original)
    if original:
        data.create_prediction('Datasets\\baseline_SVC_original_test_labels.csv', pred)
    else:
        data.create_prediction('Datasets\\baseline_SVC_test_labels.csv', pred)

    test_pred = baseline.baseline_logistic_regression(train_images, train_labels, test_images, original=original)
    if original:
        data.create_prediction('Datasets\\baseline_LR_original_test_labels.csv', test_pred)
    else:
        data.create_prediction('Datasets\\baseline_LR_test_labels.csv', test_pred)

def baseline_NN(train_images, train_labels, test_images):

    pred = baseline.baseline_custom_NN(train_images, train_labels, test_images)
    data.create_prediction('Datasets\\baseline_NN_custom_test_labels.csv', pred)

def denoise_image(images, index):
    image = images[index]
    data.create_bmp(image, 'Images\\train_noisy_'+str(index))
    denoised_image = preprocessing.denoise_image(image)
    data.create_bmp(denoised_image, 'Images\\train_clean_'+str(index))

def make_output(image_index):
    output = 31*[0]
    output[image_index] = 1
    return np.array(output)

def get_image_index_from_output(output):
    return np.argmax(output)

if __name__ == '__main__':
    main()
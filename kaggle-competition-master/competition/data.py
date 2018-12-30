import numpy as np
import pandas as pd
import binascii
from category_enum import CategoryEnum, get_category_index

"""
    Image format:
    array of arrays (of length 2) containing the index and another numpy array of length 10000 for the image data.
    [index, image_data_array]

    Categories csv format:
    index, string_format
    Indices appear to be in order so ignore index column.
"""


def load_image_array(file:str, encoding='latin1'):
    """
    Load images from pickled numpy array into numpy array.
    """
    data = np.load(file, encoding=encoding)
    return remove_index(data)

def load_categories(file:str):
    """
    Load categories from csv file and output numpy array of strings.
    """
    categories = pd.read_csv(file)
    return convert_to_class_index(categories['Category'].values)

def load_reduced_features(file:str, image_length=10000):

    xTrainData = [None] * 10000

    with open(file) as xData:

        pixelsVector = [0] * image_length

        for image in range(10000):

            values = xData.readline().split()
            for index in range(image_length): pixelsVector[index] = float(values[index])
            xTrainData[image] = list(pixelsVector)
    
    return np.array(xTrainData).astype(np.float)

def remove_index(image_array):
    """
    Remove the index entry in the image array from the pickled file and return an array of
    arrays where the image index is the array index.
    """
    result = []
    for entry in image_array:
        image = entry[1]
        result.append(image)

    result = np.array(result)
    return result

def convert_to_class_index(categories):
    """
    Convert class names to class indices using the CategoryEnum
    """
    result = []
    for i in range(0, len(categories)):
        category_string = categories[i]
        result.append(get_category_index(category_string))

    result = np.array(result)
    return result

def convert_to_class_name(categories):
    """
    Convert class index to string using CategoryEnum
    """
    result = []
    for i in range(0, len(categories)):
        category = CategoryEnum(categories[i]).name
        result.append(category)
    return result

def create_prediction(filename, prediction):
    """
    Assume that prediction is a numpy array of integer values for the class indices.
    Converts prediction to the required format for the competition
    """
    result = pd.DataFrame()
    result['Id'] = range(0, len(prediction))
    result['Category'] = convert_to_class_name(prediction)
    
    result.to_csv(filename, sep=',', index=False)

def create_binary_representation(filename, image_array):
    """
    Given an array of images, write to file in a binary format for preprocessing purpose.
    """
    with open(filename, 'wb') as f:

        image_count = len(image_array)
        image_size = len(image_array[0])

        for image in range(0, image_count):
            for i in range(0, image_size):
                pixel = image_array[image][i]
                if pixel > 255.0:
                    pixel = 255
                if pixel < 0.0:
                    pixel = 0.0
                f.write(bytes([int(pixel)]))

def read_binary_representation(filename):
    """
    Read an array of images from a binary file.
    """
    result = []
    with open(filename, 'rb') as f:
        image_count = 10000
        image_size = 10000
        for i in range(0, image_count):
            image = []
            for j in range(0, image_size):
                image.append(float(ord(f.read(1))))
            result.append(np.array(image))
    return np.array(result)

def create_bmp(image, filename):
    """
    Create .bmp file from the array
    """
    # This is taken from a 100x100 image. Since the format is uncompressed it is enough to simply copy paste the header from
    # a valid bmp file
    header = '424D889C000000000000460000003800000064000000640000000100200003000000429C0000120B0000120B00000000000000000000000000FF0000FF0000FF00000000000000'
    header = binascii.unhexlify(header)
    empty = binascii.unhexlify('00')
    file = filename + '.bmp'
    with open(file, 'wb') as f:
        
        f.write(header)

        for i in range(len(image)-1, -1, -1):
            pixel = image[i]
            if pixel > 255.0:
                pixel = 255
            if pixel < 0.0:
                pixel = 0.0

            color = int(pixel)

            #RGB
            # 0 is black and 255 is white so inverse the data
            color = bytes([255 - color])
            f.write(color)
            f.write(color)
            f.write(color)
            #A
            f.write(empty)

        f.write(binascii.unhexlify('0000'))

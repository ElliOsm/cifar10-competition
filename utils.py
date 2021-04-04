import tensorflow as tf
from tensorflow import keras as k
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml

with open('./config.yaml', 'r') as f:
    doc = yaml.safe_load(f)

def one_hot_encoding(x):
    return k.utils.to_categorical(x, doc['num_class'])

def load_dataset():
    #data_loader
    (x_train, y_train), (x_test, y_test) = k.datasets.cifar10.load_data()
    return x_train, y_train, y_test, y_test

def normalize(x):
    x = x.astype('float32')
    # normalize to range 0-1
    x = x / 255.0
    # return normalized images
    return x


def datagen(x):
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    datagen.fit(x)
    return datagen

def preprocess_data(x, y):
    x = k.applications.inception_v3.preprocess_input(x)
    y = one_hot_encoding(y, 10)
    return x, y

from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Lambda, Flatten, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy

import yaml

with open('./config.yaml', 'r') as f:
    doc = yaml.safe_load(f)


def denseNet201_compile():
    pretrained_model = DenseNet201(include_top=False,
                                   weights='imagenet',
                                   pooling=max)

    pretrained_model.trainable = False

    resize_layer = Lambda(resize_image)

    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))
    model.add(resize_layer)
    model.add(pretrained_model)

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=doc['learning_rate'])

    model.compile(optimizer=opt,
                  loss=CategoricalCrossentropy(),
                  metrics=CategoricalAccuracy())

    model.summary()

    return model


def resize_image(image):
    return tf.image.resize(image, (224, 224))


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

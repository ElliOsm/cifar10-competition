from utils import one_hot_encoding,normalize,load_dataset
import tensorflow.keras as k
from competition.models.DenseNet201 import denseNet201_compile

import yaml

with open('./config.yaml', 'r') as f:
    param = yaml.safe_load(f)


x_train, y_train, x_test, y_test = load_dataset()

x_train = k.applications.densenet.preprocess_input(x_train)

y_train = one_hot_encoding(y_train)

model = denseNet201_compile()


model.fit(x_train, y_train,
          epochs=param['epochs'],
          verbose=1,
          batch_size=param['batch_size'],
          validation_split=param['validation_split'])

model.save_weights("./weights/DenseNet201_weights.hdf5")


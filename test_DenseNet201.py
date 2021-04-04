from competition.models.DenseNet201 import denseNet201_compile
from tensorflow import keras as k
from tensorflow.keras.applications import inception_v3
from utils import one_hot_encoding,normalize,load_dataset
import sklearn.metrics as sklm
import pandas as pd

import yaml

with open('./config.yaml', 'r') as f:
    doc = yaml.safe_load(f)


x_train, y_train, x_test, y_test = load_dataset()

print(x_test)
model = denseNet201_compile()

model.load_weights("./weights/DenseNet201_weights.hdf5")

scores = model.predict(x_test,
                        batch_size=doc['batch_size'])


submission = pd.DataFrame(scores)



filename = 'cifar10-DenseNet201.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)



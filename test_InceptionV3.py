from competition.models.InceptionV3 import inceptionV3_compile,scheduler
from tensorflow import keras as k
from tensorflow.keras.applications import inception_v3
from utils import one_hot_encoding,normalize,load_dataset
import sklearn.metrics as sklm
import pandas as pd


x_train, y_train, x_test, y_test = load_dataset()

model = inceptionV3_compile()

x_test = normalize(x_test)

model.load_weights("./weights/InceptionV3_weights.hdf5")

scores = model.predict(x_test,
                        batch_size=32)


submission = pd.DataFrame(scores)



filename = 'cifar10-inceptionv3.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)
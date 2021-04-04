from competition.models.InceptionV3 import inceptionV3_compile
from tensorflow import keras as k
from utils import one_hot_encoding,normalize,load_dataset



x_train, y_train, x_test, y_test = load_dataset()

x_train = normalize(x_train)

y_train = one_hot_encoding(y_train)

model = inceptionV3_compile()


model.fit(x_train,y_train,
          epochs=10,
          verbose=1,
          batch_size=32,
          validation_split=0.15)

model.save_weights("./weights/InceptionV3_weights.hdf5")


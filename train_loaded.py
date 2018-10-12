import pickle
import time

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D, ZeroPadding2D)
from keras.models import Sequential, model_from_json
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

name = 'flower-cnn-{}'.format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
Y = pickle.load(pickle_in)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

epochs = 20
num_classes = 5
img_shape = X.shape[1:]

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Defining the model
model = Sequential()


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# fitting the model and predicting
model.fit(X_train, y_train, epochs=epochs, callbacks=[tensorboard])
y_pred = model.predict(X_test)

y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

import os
import pickle
import time

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D, ZeroPadding2D)
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

name = 'flower-cnn-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

pickle_data = open('data.pickle', 'rb')
data = pickle.load(pickle_data)
x, y = data[0], data[1]

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=42)

epochs = 20
num_classes = 5
img_shape = x.shape[1:]

y = to_categorical(y, num_classes)
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)

# Defining the model
model = Sequential()

# Layer 1
model.add(Conv2D(24, (11, 11), input_shape=img_shape,
                 padding='same', kernel_regularizer=l2(0.)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Conv2D(64, (5, 5), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 4
model.add(Conv2D(256, (3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 5
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten
model.add(Flatten())

# Layer 6
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Layer 7
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Layer 8
model.add(Dense(num_classes, activation='softmax'))
model.add(BatchNormalization())

# Flatten
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# fitting the model
model.fit(x, y, epochs=epochs, validation_split=0.3, callbacks=[tensorboard])

# y_pred = model.predict(x_test)

# y_test_class = np.argmax(y_test, axis=1)
# y_pred_class = np.argmax(y_pred, axis=1)

# print(classification_report(y_true=y_test_class, y_pred=y_pred_class))
# print(confusion_matrix(y_true=y_test_class, y_pred=y_pred_class))

model.save('model.model')
print("Saved model to disk")

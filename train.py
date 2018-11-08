import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D)
from keras.models import Sequential, load_model
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from dataset_generator import DatasetGenerator
from plot_confusion_matrix import plot_confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Log name and Tensorboard settings
name = 'aphid-cnn-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

# Load Data
dataset = DatasetGenerator()
data = dataset.load()
data /= 255
x, y = data[0], data[1]

# Split data in train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Params
train_loaded = False
epochs = 20
num_classes = 3
img_shape = x.shape[1:]
cnf_matrix = True
model_name = 'model'
save_model = True
display_summary = True

# Categorical Conversion
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

if train_loaded:
    # Load existent Model
    model = load_model(model_name + '.model')
    print("Loaded model from disk")
else:
    # Initialize model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=32, input_shape=img_shape, kernel_size=(3,3), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())

    # 1st Dense Layer
    model.add(Dense(32))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(256))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(128))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

if display_summary:
    # Summary
    model.summary()

from keras.optimizers import SGD
optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting the model
model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), callbacks=[tensorboard])

if cnf_matrix:
    classes = ['alado', 'aptero', 'ninfa']

    y_pred = model.predict(x_test)

    y_test_class = np.argmax(y_test, axis=1)
    y_pred_class = np.argmax(y_pred, axis=1)

    cnf_matrix = confusion_matrix(y_true=y_test_class, y_pred=y_pred_class)

    # Precision, Recall, F1-Score, Support
    # classification_report(y_true=y_test_class, y_pred=y_pred_class))

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes)
    plt.show()

if save_model:
    model.save(model_name + '.model')
    print("Saved model to disk")

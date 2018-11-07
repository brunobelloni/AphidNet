import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D, ZeroPadding2D)
from keras.models import Sequential, load_model
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from dataset_generator import DatasetGenerator
from plot_confusion_matrix import plot_confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Log name and Tensorboard settings
name = 'flower-cnn-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

# Load Data
dataset = DatasetGenerator()
data = dataset.load()
x, y = data[0], data[1]

# Split data in train and test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Params
epochs = 1
num_classes = 5
img_shape = x.shape[1:]
cnf_matrix = True
model_name = 'model'
train_loaded = False
save_model = True
display_summary = True

# Categorical Conversion
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

if not train_loaded:
    # Load existent Model
    model = load_model(model_name + '.model')
    print("Loaded model from disk")
else:
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

if display_summary:
    # Summary
    model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fitting the model
model.fit(x_train, y_train, epochs=epochs, validation_data=(
    x_test, y_test), callbacks=[tensorboard])

if cnf_matrix:
    classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

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

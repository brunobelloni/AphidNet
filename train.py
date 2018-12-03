#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix

from utils.aphidnet import AphidNet
from utils.cfmatrix import ConfusionMatrix
from utils.dataset import DatasetGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Log name and Tensorboard settings
name = 'aphid-cnn-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

# Load Data
categories = ['alados', 'apteros', 'ninfas']
data_dir = 'dataset/'
dataset = DatasetGenerator(data_dir, categories, img_size=128, test_size=0.1)
(x_train, x_test, y_train, y_test) = dataset.get_data()

# Params
epochs = 10
num_classes = 3
batch_size = 32
learning_rate = 0.0001
img_shape = x_train.shape[1:]
cnf_matrix = True

model = AphidNet.build(size=img_shape, classes=num_classes)

aug = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    )

# Fitting the model
opt = optimizers.Adam(lr=learning_rate, decay=learning_rate / epochs)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])
model_fit = model.fit_generator(aug.flow(x_train, y_train,
                                batch_size=batch_size),
                                validation_data=(x_test, y_test),
                                steps_per_epoch=len(x_train)
                                // batch_size, epochs=epochs,
                                callbacks=[tensorboard])

if cnf_matrix:
    y_pred = model.predict(x_test)
    y_test_class = np.argmax(y_test, axis=1)
    y_pred_class = np.argmax(y_pred, axis=1)

    cnf_matrix = confusion_matrix(y_true=y_test_class,
                                  y_pred=y_pred_class)

    # Precision, Recall, F1-Score, Support
    # classification_report(y_true=y_test_class, y_pred=y_pred_class))

    plt.figure()
    ConfusionMatrix(cnf_matrix, classes=categories)
    plt.show()

model.save('model.h5')
print 'Saved model to disk'

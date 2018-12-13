import os
import pickle
import time

from keras import optimizers
from keras.callbacks import TensorBoard
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from utils.dataset import DatasetGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load Data
categories = ['alados', 'apteros', 'ninfas']
data_dir = 'dataset/'
dataset = DatasetGenerator(data_dir, categories, img_size=128, grayscale=True)
x_train, x_test, y_train, y_test = dataset.get_data()

num_classes = 3
epochs = 30
learning_rate = 0.001
batch_size = 32

aug = ImageDataGenerator(rotation_range=25,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')

dense_layers = [1, 2]
layer_sizes = [32, 64, 128, 256]
conv_layers = [4]

print(x_train.shape[1:])

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(
                conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (2, 2),
                             input_shape=x_train.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (2, 2)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(num_classes))
            model.add(Activation('softmax'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
            opt = optimizers.Adam(
                lr=learning_rate, decay=learning_rate / epochs)
            model.compile(loss='categorical_crossentropy',
                          optimizer=opt, metrics=['accuracy'])
            model_fit = model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                                            validation_data=(x_test, y_test),
                                            steps_per_epoch=len(
                                                x_train) // batch_size,
                                            epochs=epochs, verbose=1, callbacks=[tensorboard])

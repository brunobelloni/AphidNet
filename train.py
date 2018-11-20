# USAGE
# python train.py --dataset datasets/aphids --model model.model --labelbin lb.pickle

import argparse
import itertools
import os
import pickle
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from model.aphidnet import AphidNet
from model.cf import plot_confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to input dataset (i.e., directory of images)')
ap.add_argument('-m', '--model', required=True, help='path to output model')
ap.add_argument('-l', '--labelbin', required=True,
                help='path to output label binarizer')
ap.add_argument('-p', '--plot', type=str, default='plot.png',
                help='path to output accuracy/loss plot')
args = vars(ap.parse_args())

# Log name and Tensorboard settings
name = 'aphid-cnn-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
epochs = 100
learning_rate = 0.0001
batch_size = 32
image_dims = [96, 96, 3]
tic = time.clock()
classes = []

# initialize the data and labels
data = []
labels = []

# grab the image paths and randomly shuffle them
print('[INFO] loading images...')
imagePaths = sorted(list(paths.list_images(args['dataset'])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (image_dims[1], image_dims[0]))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

    if label not in classes:
        classes.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)
print('[INFO] data matrix: {:.2f}MB'.format(data.nbytes / (1024 * 1000.0)))

# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')

# initialize the model
print('[INFO] compiling model...')
model = AphidNet.build(
    width=image_dims[1], height=image_dims[0], depth=image_dims[2], classes=len(lb.classes_))

opt = optimizers.Adam(lr=learning_rate, decay=learning_rate / epochs)

model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])

# train the network
print('[INFO] training network...')
model_fit = model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                                validation_data=(x_test, y_test),
                                steps_per_epoch=len(x_train) // batch_size,
                                epochs=epochs, verbose=1, callbacks=[tensorboard])

# Training time
toc = time.clock()
print('[INFO] Training time:', round(toc - tic), 'seconds')

# save the model to disk
print('[INFO] serializing network...')
model.save(args['model'])

# save the label binarizer to disk
print('[INFO] serializing label binarizer...')
f = open(args['labelbin'], 'wb')
f.write(pickle.dumps(lb))
f.close()


# Confusion Matrix
y_pred = model.predict(x_test)

y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

cnf_matrix = confusion_matrix(y_true=y_test_class, y_pred=y_pred_class)

# Precision, Recall, F1-Score, Support
# classification_report(y_true=y_test_class, y_pred=y_pred_class))

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes)
plt.show()

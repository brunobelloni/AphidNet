import os

import cv2
import numpy as np
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

categories = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


def prepare(filepath):
    IMG_SIZE = 64
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


model = load_model('model.model')

dir = 'flower_photos/dandelion/7355522_b66e5d3078_m.jpg'
prediction = model.predict([prepare(dir)])
prediction = np.argmax(prediction)

print(categories[prediction])

import os
import pickle
import random

import cv2
import numpy as np

data_dir = 'flower_photos/'
categories = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

img_rows, img_cols = 64, 64

training_data = []
x = []
y = []


def create_training_data():
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_rows, img_cols))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()

random.shuffle(training_data)


for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, img_rows, img_cols, 3)
y = np.array(y)

data = [x, y]

pickle_data = open('data.pickle', 'wb')
pickle.dump(data, pickle_data)
pickle_data.close()
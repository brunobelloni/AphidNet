import os
import pickle
import random

import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from .progressbar import ProgressBar


class DatasetGenerator():
    """
    DataSetsFolder
    |--- Class-1
    |        .   |--- Image-1
    |        .   |--- Image-N
    |--- Class-N

    """

    def __init__(self, path=None, categories=None, img_size=64, grayscale=False, test_size=0.2):
        self.path = path
        self.categories = categories
        self.img_size = img_size
        self.training_data = []
        self.grayscale = grayscale
        self.test_size = test_size
        self.load_dataset()

    def load_dataset(self, shuffle=True):
        try:
            for category in self.categories:
                _path = os.path.join(self.path, category)
                class_num = self.categories.index(category)

                img_dir = os.listdir(_path)
                img_dir_len = len(img_dir)

                for img in img_dir:
                    try:
                        ProgressBar(category, img_dir.index(img), img_dir_len)
                        img_array = cv2.imread(os.path.join(_path, img))
                        img_array = cv2.resize(
                            img_array, (self.img_size, self.img_size))

                        if self.grayscale:
                            img_array = cv2.cvtColor(
                                img_array, cv2.COLOR_BGR2GRAY)

                        img_array = img_to_array(img_array)
                        self.training_data.append([img_array, class_num])
                    except Exception as e:
                        print('Error', e)
                print(' ')
        except FileNotFoundError as f:
            print('The system can not find the path specified: {}!'.format(_path))

        self.shuffle()

    def shuffle(self):
        random.shuffle(self.training_data)

    def get_data(self):
        x, y = [], []
        for features, label in self.training_data:
            x.append(features)
            y.append(label)

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(x), np.array(y), test_size=self.test_size)
        y_train = to_categorical(y_train, len(self.categories))
        y_test = to_categorical(y_test, len(self.categories))
        return x_train, x_test, y_train, y_test


def main():
    categories = ['alados', 'apteros', 'ninfas']
    data_dir = 'dataset/'

    dataset = DatasetGenerator(data_dir, categories, img_size=128)
    data = dataset.get_data()


if __name__ == '__main__':
    main()

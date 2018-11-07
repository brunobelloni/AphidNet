import os
import pickle
import random
import sys

import cv2
import numpy as np


def progressBar(name, value, endvalue, bar_length=50, width=10):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{0: <{1}} : [{2}] {3}%".format(name, width,
                                                       arrow + spaces,
                                                       int(round(percent * 100))))

    sys.stdout.flush()

    if value == endvalue:
        sys.stdout.write('\n\n ')


class DatasetGenerator():
    def __init__(self, path=None, categories=None, img_size=64, channels=3):
        self.path = path
        self.categories = categories
        self.img_size = img_size
        self.training_data = []
        self.channels = channels

    def load_dataset(self, img_size=64, shuffle=True):
        """
        DataSetsFolder
        |--- Class-1
        |        .   |--- Image-1
        |        .   |--- Image-N
        |--- Class-N

        """
        try:
            for category in categories:
                path = os.path.join(data_dir, category)
                class_num = categories.index(category)

                img_dir = os.listdir(path)
                img_dir_len = len(img_dir)

                for img in img_dir:
                    try:
                        progressBar(category, img_dir.index(img), img_dir_len)
                        img_array = cv2.imread(os.path.join(path, img))
                        new_array = cv2.resize(img_array, (img_size, img_size))
                        self.training_data.append([new_array, class_num])
                    except Exception as e:
                        print('Error', e)
                print(' ')
        except FileNotFoundError as f:
            print('The system can not find the path specified: {}!'.format(path))

        if shuffle:
            self.shuffle()

    def shuffle(self):
        random.shuffle(self.training_data)

    def save(self, name='data'):
        if not self.training_data:
            print('Run the "load_dataset" method before saving!')
        else:
            x, y = [], []
            for features, label in self.training_data:
                x.append(features)
                y.append(label)

            x = np.array(x).reshape(-1, self.img_size,
                                    self.img_size, self.channels)
            y = np.array(y)
            data = [x, y]

            pickle_data = open(name + '.pickle', 'wb')
            pickle.dump(data, pickle_data)
            pickle_data.close()

            print('Data saved as "{}.pickle"!'.format(name))

    def load(self, name='data'):
        try:
            pickle_data = open(name + '.pickle', 'rb')
            print('Data was loaded!')
            return pickle.load(pickle_data)
        except FileNotFoundError as f:
            print('No such file or directory: "{}.pickle"!'.format(name))


def main():
    categories = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    data_dir = 'flower_photos/'

    dataset = DatasetGenerator(data_dir, categories)
    dataset.load_dataset()
    dataset.save()
    data = dataset.load()


if __name__ == '__main__':
    main()

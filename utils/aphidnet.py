from keras import backend as K
from keras import regularizers
from keras.layers.convolutional import AveragePooling2D, Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


class AphidNet:
    @staticmethod
    def build(size, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (size[0], size[1], size[2])
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (size[2], size[1], size[0])
            chanDim = 1

        # 1st Convolutional Layer
        model.add(Conv2D(filters=32, input_shape=inputShape,
                         kernel_size=(3, 3), padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(BatchNormalization(axis=chanDim))

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(BatchNormalization(axis=chanDim))

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))

        # Passing it to a dense layer
        model.add(Flatten())

        # 1st Dense Layer
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization(axis=chanDim))

        # 2nd Dense Layer
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization(axis=chanDim))

        # 3rd Dense Layer
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization(axis=chanDim))

        # Output Layer
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        # return the constructed network architecture
        return model

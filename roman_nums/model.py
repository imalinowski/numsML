import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Activation, BatchNormalization
from tensorflow.python.keras import activations


def create_roman_model():
    model = tf.keras.models.Sequential([])

    # convolutional
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

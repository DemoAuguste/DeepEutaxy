import keras
# from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential


def lenet5():
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # sgd = keras.optimizers.SGD(learning_rate=0.1)
    ada = keras.optimizers.Adagrad(lr=0.1, epsilon=1e-6, decay=0.0005)
    model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=ada, metrics=['accuracy'])
    return model


def model1():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))
    # sgd = keras.optimizers.SGD(learning_rate=0.1)
    ada = keras.optimizers.Adagrad(lr=0.1, epsilon=1e-6, decay=0.0005)
    model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=ada, metrics=['accuracy'])
    return model


def lenet5_uncompiled():
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # sgd = keras.optimizers.SGD(learning_rate=0.1)
    ada = keras.optimizers.Adagrad(lr=0.1, epsilon=1e-6, decay=0.0005)
    return model, ada


def model1_uncompiled():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))
    # sgd = keras.optimizers.SGD(learning_rate=0.1)
    ada = keras.optimizers.Adagrad(lr=0.1, epsilon=1e-6, decay=0.0005)
    return model, ada


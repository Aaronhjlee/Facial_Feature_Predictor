import pandas as pd 
import numpy as np
import tensorflow
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Deconvolution2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam
import tensorflow as tf
from tensorflow.keras import layers
keras = tf.keras
AUTOTUNE = tf.data.experimental.AUTOTUNE
from keras.callbacks import TensorBoard
from livelossplot.keras import PlotLossesCallback
import warnings
warnings.filterwarnings("ignore")

from preprocessing import get_images

def train_gender_model(X_train, X_test, y_train, y_test):
    print('Training gender model! (might take a while)')
    cnn_g = Sequential()
    input_img = (218, 178, 3)
    batch_size=32
    epochs=15

    # layer 1
    cnn_g.add(Conv2D(2, (3, 3), activation='relu', padding='same', input_shape=input_img))
    cnn_g.add(Conv2D(2, (3, 3), activation='relu', padding='same'))
    cnn_g.add(MaxPooling2D((2, 2), strides=(2,2)))
    # layer 2
    cnn_g.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
    cnn_g.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
    cnn_g.add(MaxPooling2D((2, 2), strides=(2,2)))
    # layer 3
    cnn_g.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    cnn_g.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    cnn_g.add(MaxPooling2D((2, 2), strides=(2,2)))
    # flatten and add 3 FC layers
    cnn_g.add(Flatten())
    cnn_g.add(Dense(64, activation='relu'))
    cnn_g.add(Dropout(0.5))
    cnn_g.add(Dense(1, activation='sigmoid'))

    cnn_g.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_g.fit(X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            # callbacks=[PlotLossesCallback()],
            verbose=0)
    print('Finished training and saved gender model!')
    saved_model_path = "../saved_models/gender_11_10.h5"
    cnn_g.save(saved_model_path)


def train_male_attraction_model(Xm_train, Xm_test, ym_train, ym_test):
    print('Training gender model! (might take a while)')
    cnn_m = Sequential()
    input_img = (218, 178, 3)
    batch_size=32
    epochs=25

    # layer 1
    cnn_m.add(Conv2D(2, (3, 3), activation='relu', padding='same', input_shape=input_img))
    cnn_m.add(Conv2D(2, (3, 3), activation='relu', padding='same'))
    cnn_m.add(MaxPooling2D((2, 2), strides=(2,2)))
    # layer 2
    cnn_m.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
    cnn_m.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
    cnn_m.add(MaxPooling2D((2, 2), strides=(2,2)))
    # flatten and add 3 FC layers
    cnn_m.add(Flatten())
    cnn_m.add(Dense(64, activation='relu'))
    cnn_m.add(Dropout(0.5))
    cnn_m.add(Dense(32, activation='relu'))
    cnn_m.add(Dropout(0.5))
    cnn_m.add(Dense(1, activation='sigmoid'))

    cnn_m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    cnn_m.fit(Xm_train, ym_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(Xm_test, ym_test),
            # callbacks=[PlotLossesCallback()],
            verbose=0)
    print('Finished training and saved male attraction model!')
    saved_model_path = "../saved_models/male_att_11_10.h5"
    cnn_m.save(saved_model_path)
    

def train_female_attraction_model(Xf_train, Xf_test, yf_train, yf_test):
    print('Training gender model! (might take a while)')
    cnn_f = Sequential()
    input_img = (218, 178, 3)
    batch_size=32
    epochs=25

    # layer 1
    cnn_f.add(Conv2D(2, (3, 3), activation='relu', padding='same', input_shape=input_img))
    cnn_f.add(Conv2D(2, (3, 3), activation='relu', padding='same'))
    cnn_f.add(MaxPooling2D((2, 2), strides=(2,2)))
    # layer 2
    cnn_f.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
    cnn_f.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
    cnn_f.add(MaxPooling2D((2, 2), strides=(2,2)))
    # layer 3
    cnn_f.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    cnn_f.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    cnn_f.add(MaxPooling2D((2, 2), strides=(2,2)))
    # flatten and add 3 FC layers
    cnn_f.add(Flatten())
    cnn_f.add(Dense(64, activation='relu'))
    cnn_f.add(Dropout(0.5))
    cnn_f.add(Dense(1, activation='sigmoid'))

    cnn_f.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    cnn_f.fit(Xm_train, ym_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(Xm_test, ym_test),
            # callbacks=[PlotLossesCallback()],
            verbose=0)
    print('Finished training and saved female attraction model!')
    saved_model_path = "../saved_models/female_att_11_10.h5"
    cnn_f.save(saved_model_path)

if __name__ == "__main__":
    start = 0
    n = 5000
    print ('running {} data points.'.format(n))
    X_train, X_test, y_train, y_test = get_images(start,n)
    Xm_train, Xm_test, ym_train, ym_test, Xf_train, Xf_test, yf_train, yf_test = get_images(start,n, split=True)
    print('Loaded both datasets for training')
    train_gender_model(X_train, X_test, y_train, y_test)
    train_male_attraction_model(Xm_train, Xm_test, ym_train, ym_test)
    train_female_attraction_model(Xf_train, Xf_test, yf_train, yf_test)
    print('D O N E')
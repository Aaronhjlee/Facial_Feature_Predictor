import pandas as pd 
import numpy as np
import tensorflow
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
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

    print('Finished training and saved gender model!')
    pass

def train_male_attraction_model(Xm_train, Xm_test, ym_train, ym_test):
    print('Training male attraction model! (might take a while)')
    input_img = Input(shape=(200, 200, 3)) 
    batch_size=32
    epochs=25
    x = Conv2D(2, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(2, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2,2))(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2,2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2,2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    cnn_m = Model(input_img, output)
    cnn_m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    cnn_m.fit(Xm_train, ym_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(Xm_test, ym_test),
            callbacks=[PlotLossesCallback()],
            verbose=0)
    print('Finished training and saved male attraction model!')
    

def train_female_attraction_model(Xf_train, Xf_test, yf_train, yf_test):
    print('Training female attraction model! (might take a while)')
    input_img = Input(shape=(200, 200, 3)) 
    batch_size=32
    epochs=25
    x = Conv2D(2, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(2, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2,2))(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2,2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2,2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    cnn1 = Model(input_img, output)
    cnn1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    cnn1.fit(Xf_train, yf_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(Xf_test, yf_test),
            callbacks=[PlotLossesCallback()],
            verbose=0)
    print('Finished training and saved female attraction model!')

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_images(0,100)
    Xm_train, Xm_test, ym_train, ym_test, Xf_train, Xf_test, yf_train, yf_test = get_images(0,100, split=True)
    print('Loaded both datasets for training')
    train_gender_model(X_train, X_test, y_train, y_test)
    train_male_attraction_model(Xm_train, Xm_test, ym_train, ym_test)
    train_female_attraction_model(Xf_train, Xf_test, yf_train, yf_test)
    print('D O N E')
import pandas as pd 
import numpy as np
import tensorflow
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, MaxPooling2D
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
from load_and_predict import load_models

def gender_transfer(cnn_g, X_train, X_test, y_train, y_test, model_num):
    print('Training gender model! (might take a while)')
    batch_size=32
    epochs=25
    cnn_g.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_g.fit(X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            # callbacks=[PlotLossesCallback()],
            verbose=0)
    print('Finished training and saved gender model!')
    saved_model_path = "../saved_models/gender{}_11_10.h5".format(model_num+1)
    cnn_g.save(saved_model_path)

def male_transfer(cnn_m, Xm_train, Xm_test, ym_train, ym_test, model_num):
    print('Training male attraction model! (might take a while)')
    batch_size=32
    epochs=25
    cnn_m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_m.fit(X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            # callbacks=[PlotLossesCallback()],
            verbose=0)
    print('Finished training and saved male attraction model!')
    saved_model_path = "../saved_models/male{}_att_11_10.h5".format(model_num+1)
    cnn_m.save(saved_model_path)

def female_transfer(cnn_f, Xf_train, Xf_test, yf_train, yf_test, model_num):
    print('Training female attraction model! (might take a while)')
    batch_size=32
    epochs=25
    cnn_f.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_f.fit(X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            # callbacks=[PlotLossesCallback()],
            verbose=0)
    print('Finished training and saved female attraction model!')
    saved_model_path = "../saved_models/female{}_att_11_10.h5".format(model_num+1)
    cnn_f.save(saved_model_path)

if __name__ == "__main__":
    start = 4000
    n = 8000
    model_num=1
    print ('running {} data points.'.format(n-start))
    X_train, X_test, y_train, y_test = get_images(start,n)
    Xm_train, Xm_test, ym_train, ym_test, Xf_train, Xf_test, yf_train, yf_test = get_images(start,n, split=True)
    print('Loaded both datasets for training')
    cnn_g, cnn_m, cnn_f = load_models(model_num)
    gender_transfer(cnn_g, X_train, X_test, y_train, y_test, model_num)
    male_transfer(cnn_m, Xm_train, Xm_test, ym_train, ym_test, model_num)
    female_transfer(cnn_f, Xf_train, Xf_test, yf_train, yf_test, model_num)
    print('D O N E')
    
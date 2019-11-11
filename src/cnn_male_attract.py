from datetime import datetime
import cv2
import tensorflow
import keras
import pandas as pd
import numpy as np
from keras.datasets import mnist
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

from sklearn.model_selection import train_test_split, GridSearchCV

import datetime
from src.cnn_gender import get_and_clean_data

import warnings
warnings.filterwarnings("ignore")


def get_images():
    full_images, df = get_and_clean_data()
    img_list = np.asarray(full_images[:4000])
    img_list.shape
    temp=[]
    for i in img_list:
        temp.append(cv2.resize(i, (200,200)))
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(temp),
                                                       df.male[:4000] ,shuffle=False)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255    
    return X_train, X_test, y_train, y_test, np.asarray(temp), df[:4000]


if __name__ == "__main__":
    full_images, df_labels = get_and_clean_data()
    print ('Retrieved and cleaned_data!')
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(full_images), 
                            np.asarray(df_labels.male)[:100], shuffle=False)
    print ('Split the data!')
    model = MaleAttract(X_train, X_test, y_train, y_test).build_model()
    print ('Model made on {} samples'.format(X_train.shape[0]))
    X_train_images = X_train.reshape((X_train.shape[0], 218, 178, 3))
    # how many epochs?
    model.fit(X_train_images, y_train, epochs=10, verbose=2)
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', score[1]) # this is the one we care about

    saved_model_path = "male_attract_{}.h5".format(datetime.now().strftime("%Y%m%d")) # _%H%M%S
    # Save entire model to a HDF5 file
    model.save(saved_model_path)
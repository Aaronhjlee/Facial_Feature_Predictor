import pandas as pd 
import numpy as np
import tensorflow
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
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

from preprocessing import prep_size_new_data

def load_models(model_num):
    cnn_g = load_model('../saved_models/gender{}_11_10.h5'.format(model_num))
    cnn_m = load_model('../saved_models/male{}_att_11_10.h5'.format(model_num))
    cnn_f = load_model('../saved_models/female{}_att_11_10.h5'.format(model_num))
    return cnn_g, cnn_m, cnn_f

def gender_classifier(X, cnn_g):
    pred = cnn_g.predict_proba(X)
    if pred[0][0]*100 > 50:
        print('     MALE: {}%'.format(round(pred[0][0]*100,2)))
    else:
        print('     FEMALE: {}%'.format(round((1-pred[0][0])*100,2)))
    return cnn_g.predict_classes(X)[0][0]

def am_i_attractive(X, gender, cnn_m, cnn_f):
    if gender == 1:
        pred = cnn_m.predict_proba(X)
        print('     Photogenic MALE: {}%'.format(round(pred[0][0]*100,10)))
    else:
        pred = cnn_f.predict_proba(X)
        print('     Photogenic FEMALE: {}%'.format(round(pred[0][0]*100,10)))


if __name__ == "__main__":
    n=1
    model_num=1
    # Use prep_size_new_data_PB if faces are not aligned already
    X, names = prep_size_new_data(photobooth=True)
    cnn_g, cnn_m, cnn_f = load_models(model_num)
    print('-------------')
    for i in range(X.shape[0]):
        print ('{}'.format(names[i][14:]))
        gender = gender_classifier(X[i].reshape(1,218, 178,3), cnn_g)
        am_i_attractive(X[i].reshape(1,218, 178,3), gender, cnn_m, cnn_f)
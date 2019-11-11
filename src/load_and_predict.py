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
    print('By Western standards, the picture you submitted is {} a male. (scale: 0-1)'.format(pred[0][0]))
    return cnn_g.predict_classes(X)[0][0]

def am_i_attractive(X, gender, cnn_m, cnn_f):
    if gender == 1:
        pred = cnn_m.predict(X)
        print('By Western male standards, the picture you submitted is {} attractive (scale: 0-1)'.format(pred[0][0]))
    else:
        pred = cnn_f.predict(X)
        print('By Western female standards, the picture you submitted is {} attractive (scale: 0-1)'.format(pred[0][0]))


if __name__ == "__main__":
    n=18
    model_num=1
    print('Image: {}  |  Model Number: {}'.format(n, model_num))
    X = prep_size_new_data(n-1,n)
    cnn_g, cnn_m, cnn_f = load_models(model_num)
    gender = gender_classifier(X, cnn_g)
    am_i_attractive(X, gender, cnn_m, cnn_f)
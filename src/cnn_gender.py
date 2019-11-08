from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers
keras = tf.keras
AUTOTUNE = tf.data.experimental.AUTOTUNE ## tf.data transformation parameters

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


class GenderModel:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def build_model(self):
        # layer 1
        model = keras.models.Sequential([
        keras.layers.Conv2D(16, (5, 5), activation='relu',
                        input_shape=(218, 178, 3), padding = "same"),
        keras.layers.MaxPooling2D((2, 2))])
        # layer 2
        model.add(keras.layers.Conv2D(32, (5, 5), activation = 'relu', padding = 'same'))
        model.add(keras.layers.MaxPool2D(2,2))
        #layer 3
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation = "relu"))
        model.add(keras.layers.Dense(2,  activation = "softmax"))
        # compile
        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        return model


def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape 
    (width, height, channels)
    """
    with Image.open(image_path) as image:         
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
        return im_arr

def get_and_clean_data():
    # The number of celeb images: 202599
    n_celeb_images = 20000
    # The format specification here left pads zeros on the number: 000006.
    celeb_filenames = ['data/img_align_celeba/{:06d}.jpg'.format(i)
                        for i in range(1, n_celeb_images + 1)]
    full_images=[]
    for i in celeb_filenames:
        full_images.append(jpg_image_to_array(i))

    df_labels = pd.read_csv('data/list_attr_celeba.csv')
    df_labels.columns = map(str.lower, df_labels.columns)
    df_labels.replace([-1], 0, inplace=True)
    return full_images, df_labels


if __name__ == "__main__":
    full_images, df_labels = get_and_clean_data()
    print ('Retrieved and cleaned_data!')
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(full_images), 
                        np.asarray(df_labels.male)[:20000], shuffle=False)
    print ('Split the data!')
    model = GenderModel(X_train, X_test, y_train, y_test).build_model()
    print ('Model made on {} samples'.format(X_train.shape[0]))
    X_train_images = X_train.reshape((X_train.shape[0], 218, 178, 3))
    # how many epochs?
    model.fit(X_train_images, y_train, epochs=10, verbose=2)
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', score[1]) # this is the one we care about

    saved_model_path = "gender_{}.h5".format(datetime.now().strftime("%Y%m%d")) # _%H%M%S
    # Save entire model to a HDF5 file
    model.save(saved_model_path)
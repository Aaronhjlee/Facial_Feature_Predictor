import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape 
    (width, height, channels)
    """
    with Image.open(image_path) as image:         
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
        return im_arr

def get_and_clean_data(start, n):
    '''
    The number of celeb images: 202599
    The format specification here left pads zeros on the number: 000006
    '''
    celeb_filenames = ['../data/img_align_celeba/{:06d}.jpg'.format(i)
                        for i in range(start + 1, n + 1)]
    full_images=[]
    for i in celeb_filenames:
        full_images.append(jpg_image_to_array(i))

    df_labels = pd.read_csv('../data/list_attr_celeba.csv')
    df_labels.columns = map(str.lower, df_labels.columns)
    df_labels.replace([-1], 0, inplace=True)
    return full_images, df_labels[start:n]

def male_female_split(full_images, df):
    '''
    Splits images and df labels to train and testing set based on gender
    Input: 2 int
    Output: Train 2d array / Series
    '''
    full_males = full_images[df[df.male==1].index]
    y_male = df.attractive[df[df.male==1].index]
    full_females = full_images[df[df.male==0].index]
    y_female = df.attractive[df[df.male==0].index]
    Xm_train, Xm_test, ym_train, ym_test = train_test_split(full_males, y_male)
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(full_females, y_female)
    return Xm_train, Xm_test, ym_train, ym_test, Xf_train, Xf_test, yf_train, yf_test

def full_split(full_images, df):
    '''
    Splits images and df labels to train and testing set
    Input: 2 int
    Output: Train 2d array / Series
    '''
    X_train, X_test, y_train, y_test = train_test_split(full_images, df.male)
    return X_train, X_test, y_train, y_test

def get_images(start, n, split=False):
    full_images, df = get_and_clean_data(start,n)
    img_list = np.asarray(full_images[start:n])
    img_list.shape
    temp=[]
    for i in img_list:
        temp.append(cv2.resize(i, (218,178)))
    full_images = np.asarray(temp) / 255
    if split == False:
        return full_split(full_images, df)
    else:
        return male_female_split(full_images, df)
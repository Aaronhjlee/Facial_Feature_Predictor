import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join

def get_and_clean_data(start, n):
    '''
    The number of celeb images: 202599
    The format specification here left pads zeros on the number: 000006
    '''
    # celeb_filenames = ['../data/img_align_celeba/{:06d}.jpg'.format(i)
    #                     for i in range(start + 1, n + 1)]
    np.random.seed(123)
    idx = [i for i in np.random.choice(202599, n,  replace=False)]
    celeb_filenames = ['../data/img_align_celeba/{:06d}.jpg'.format(i) for i in idx]
    full_images = [cv2.imread(f)[...,::-1] for f in celeb_filenames]

    df_labels = pd.read_csv('../data/list_attr_celeba.csv')
    df_labels.columns = map(str.lower, df_labels.columns)
    df_labels.replace([-1], 0, inplace=True)
    return full_images, df_labels.iloc[np.asarray(idx)-1]

def male_female_split(full_images, df, start):
    '''
    Splits images and df labels to train and testing set based on gender
    Input: 2 int
    Output: Train 2d array / Series
    '''
    df = df.reset_index()
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
    full_images = np.asarray(full_images) / 255
    if split == False:
        return full_split(full_images, df)
    else:
        return male_female_split(full_images, df, start)

def prep_size_new_data(photobooth=False):
    '''
    Converts picture and sizes for CNN
    Input: multiple images
    Output: 3d array
    '''
    subject_filenames = ['../data/batch/{}'.format(f) for f in listdir('../data/batch') if f != '.DS_Store']
    full_images = [cv2.imread(f)[...,::-1] for f in subject_filenames]
    img_list = np.asarray(full_images) / 255
    if photobooth == True:
        pre = []
        for i in img_list:
            pre.append(i[:,250:830])
        X = []
        for i in pre:
            X.append(cv2.resize(i, dsize=(178, 218)))
    else:
        X = []
        for i in img_list:
            X.append(cv2.resize(i, dsize=(178, 218)))
    return np.asarray(X), subject_filenames

if __name__ == "__main__":
    # X_train, X_test, y_train, y_test = get_images(5000,5050)
    # Xm_train, Xm_test, ym_train, ym_test, Xf_train, Xf_test, yf_train, yf_test = get_images(5000, 5050, split=True)
    # X = prep_size_new_data(0,1)
    print ('it ran')
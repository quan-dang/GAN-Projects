"""
utility functions to preprocess the dataset
"""

from scipy.io import loadmat
from datetime import datetime

import os
import time

from keras import Input, Model
from keras.applications import InceptionResNetV2
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Reshape, concatenate, LeakyReLU, Lambda, \
    K, Activation, UpSampling2D, Dropout
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing import image

import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 
from tqdm import tqdm

def load_data(wiki_dir, dataset='wiki'):
    # load the wiki.mat in the meta
    meta = loadmat(os.path.join(wiki_dir, "{}.mat".format(dataset)))

    # load the list of all files
    full_path = meta[dataset][0, 0]["full_path"][0]

    # list of matlab serial date numbers for the corresponding photo in the list
    dob = meta[dataset][0, 0]["dob"][0]

    # list of years when photo was taken
    photo_taken = meta[dataset][0, 0]["photo_taken"][0] 

    # calculate age for all dobs
    age = [calculate_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    # create a list of tuples containing a pair of an image path and age
    imgs = []
    age_list = []

    for i, img_path in enumerate(full_path):
        imgs.append(img_path[0])
        age_list.append(age[i])

    # return a list of all images and their respective age
    return imgs, age_list

# calculate the age of the person from the serial date number
# and the year the photo was taken
def calculate_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


# convert age numerical value to the age categogy
def age_to_category(age_list):
    age_list1 = []

    for age in age_list:
        if 0 < age <= 18:
            age_category = 0
        elif 18 < age <= 29:
            age_category = 1
        elif 29 < age <= 39:
            age_category = 2
        elif 39 < age <= 49:
            age_category = 3
        elif 49 < age <= 59:
            age_category = 4
        elif age >= 60:
            age_category = 5

        age_list1.append(age_category)
    return age_list1


# load images from data directory
def load_images(data_dir, image_paths, image_shape):
    images = None
    # pbar = tqdm(total=len(image_paths))
    
    print("[INFO] processing images...")

    for i, image_path in enumerate(tqdm(image_paths)):
        # print("[INFO] processing image {}/{}".format(i+1, len(image_paths)))
        try:
            # Load image
            loaded_image = image.load_img(os.path.join(data_dir, image_path), target_size=image_shape)

            # Convert PIL image to numpy ndarray
            loaded_image = image.img_to_array(loaded_image)

            # Add another dimension (Add batch dimension)
            loaded_image = np.expand_dims(loaded_image, axis=0)

            # Concatenate all images into one tensor
            if images is None:
                images = loaded_image
            else:
                images = np.concatenate([images, loaded_image], axis=0)

        except Exception as e:
            print("Error:", i, e)

    return images

# calculate euclidean distance loss
def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


# write loss to tensorboard
def write_log(callback, name, value, batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


# save an rgb image
def save_rgb_image(img, path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Image")

    plt.savefig(path)
    plt.close()


# resize the image
def build_image_resizer():
    input_layer = Input(shape=(64, 64, 3))

    resized_images = Lambda(lambda x: K.resize_images(x, height_factor=3, width_factor=3,
                                                      data_format='channels_last'))(input_layer)

    model = Model(inputs=[input_layer], outputs=[resized_images])
    return model
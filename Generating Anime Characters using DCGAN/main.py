import glob
import io
import math
import time

import keras.backend as K
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from keras import Sequential, Input, Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.callbacks import TensorBoard
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image

from scipy.misc import imread, imsave
from scipy.stats import entropy

from utils import *

# setting for tensorflow backend
K.set_image_dim_ordering('tf')

# set random seed
seed = 42
np.random.seed(seed)

"""
create a generator network
"""
def build_generator():
    gen_model = Sequential()

    gen_model.add(Dense(2048, input_shape=(100,)))
    gen_model.add(ReLU())

    gen_model.add(Dense(256 * 8 * 8))
    gen_model.add(BatchNormalization())
    gen_model.add(ReLU())
    gen_model.add(Reshape((8, 8, 256), input_shape=(256 * 8 * 8,)))
    gen_model.add(UpSampling2D(size=(2, 2)))

    gen_model.add(Conv2D(128, (5, 5), padding='same'))
    gen_model.add(ReLU())

    gen_model.add(UpSampling2D(size=(2, 2)))

    gen_model.add(Conv2D(64, (5, 5), padding='same'))
    gen_model.add(ReLU())

    gen_model.add(UpSampling2D(size=(2, 2)))

    gen_model.add(Conv2D(3, (5, 5), padding='same'))
    gen_model.add(Activation('tanh'))
    
    return gen_model

"""
create a discriminator network
"""

def build_discriminator():
    dis_model = Sequential()
    dis_model.add(Conv2D(128, (5, 5),
                padding='same',
                input_shape=(64, 64, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    dis_model.add(Conv2D(256, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    dis_model.add(Conv2D(512, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    dis_model.add(Flatten())
    dis_model.add(Dense(1024))
    dis_model.add(LeakyReLU(alpha=0.2))

    dis_model.add(Dense(1))
    dis_model.add(Activation('sigmoid'))

    return dis_model


"""
create an adversarial model from generator and discriminator networks
"""

def build_adversarial_model(gen_model, dis_model):
    model = Sequential()
    model.add(gen_model)
    dis_model.trainable = False
    model.add(dis_model)
    return model


"""
ready to train the model
"""
def train():
    start = time.time()

    # param for dataset directory
    data_dir = "gallery-dll/danbooru/face"

    # specify hyperparams for training
    batch_size = 128
    z_shape = 100
    epochs = 10000
    dis_learning_rate = 0.005
    gen_learning_rate = 0.005
    dis_momentum = 0.5
    gen_momentum = 0.5
    dis_nesterov = True
    gen_nesterov = True

    # optimizers for generator and discriminator networks
    dis_optimizer = SGD(lr=dis_learning_rate, momentum=dis_momentum, nesterov=dis_nesterov)
    gen_optimizer = SGD(lr=gen_learning_rate, momentum=gen_momentum, nesterov=gen_nesterov)

    # load all images
    all_images = []
    for _, filename in enumerate(glob.glob(data_dir)):
        all_images.append(imread(filename, flatten=False, mode='RGB'))

    X = np.array(all_images)
    X = (X - 127.5) / 127.5
    X = X.astype(np.float32)

    # build and compile generator model
    gen_model = build_generator()
    gen_model.compile(loss='mse', optimizer=gen_optimizer)

    # build and compile discriminator
    dis_model = build_discriminator()
    dis_model.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

    # build and compile adversarial model
    adversarial_model = build_adversarial_model(gen_model, dis_model)
    adversarial_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    # add tensorboard to visualize losses
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), write_images=True, write_grads=True, write_graph=True)
    tensorboard.set_model(gen_model)
    tensorboard.set_model(dis_model)

    for epoch in range(epochs):
        print("-" * 50)
        print("[INFO] epoch: {}".format(epoch))

        dis_losses = []
        gen_losses = []

        num_batches = int(X.shape[0] / batch_size)
        print("[INFO] number of batches: {}".format(num_batches))

        for i in range(num_batches):
            print("-" * 25)
            print("[INFO] batch: {}".format(i))

            #sample a batch of noise vectors from a normal distribution
            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

            # enerate a batch of fake images using the generator networ
            generated_images = gen_model.predict_on_batch(z_noise)

            """
            train the discriminator network
            """

            # start to train dis_model
            dis_model.trainable = True

            # sample a batch of real images from the set of all images
            image_batch = X[i*batch_size: (i+1)*batch_size]

            # create real labels and fake labels
            y_real = np.ones((batch_size, )) * 0.9
            y_fake = np.zeros((batch_size, )) * 0.1

            # train the discriminator network on real images and real labels
            dis_loss_real = dis_model.train_on_batch(image_batch, y_real)

            # train the discriminator network on fake images and fake labels
            dis_loss_fake = dis_model.train_on_batch(generated_images, y_fake)

            # calculate the average loss
            d_loss = (dis_loss_real+dis_loss_fake)/2
            print("[INFO] d_loss: {}".format(d_loss))

            # stop training dis_model
            dis_model.trainable = False

            """
            train the generator model (adversarial model)
            """
            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

            g_loss = adversarial_model.train_on_batch(z_noise, y_real)
            print("[INFO] g_loss: {}".format(g_loss))

            dis_losses.append(d_loss)
            gen_losses.append(g_loss)

        """
        sample some images to check the performance
        """
        if epoch%100 == 0:
            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
            gen_images1 = gen_model.predict_on_batch(z_noise)

            for img in gen_images1[:2]:
                save_rgb_image(img, "results/one_{}.png".format(epoch))

        print("[INFO] epoch: {}, dis_loss: {}".format(epoch, np.mean(dis_losses)))
        print("[INFO] epoch: {}, gen_loss: {}".format(epoch, np.mean(gen_losses)))

        """
        save losses to tensorboard after each epoch
        """
        write_log(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
        write_log(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)

    """
    save our models
    """
    gen_model.save("generator_model.h5")
    dis_model.save("generator_model.h5")

    print("[INFO] time elapsed: {}s", (time.time() - start))


if __name__ == "__main__":
    train()
    


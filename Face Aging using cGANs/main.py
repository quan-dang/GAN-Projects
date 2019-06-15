import math
import time

import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 

from keras import Input, Model
from keras.applications import InceptionResNetV2
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Reshape, concatenate, LeakyReLU, Lambda, \
    K, Activation, UpSampling2D, Dropout
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing import image

from utils import *

# build encoder network
def build_encoder():
    # specify hyperparameters
    filters = [32, 64, 128, 256]
    kernel_sizes = [5, 5, 5, 5]
    strides = [2, 2, 2, 2]
    padding = ['same', 'same', 'same', 'same']
    conv_blocks = 4

    input_layer = Input(shape=(64, 64, 3))

    # 1st conv block
    enc = Conv2D(filters=filters[0], kernel_size=kernel_sizes[0], \
        strides=strides[0], padding=padding[0])(input_layer)
    enc = LeakyReLU(alpha=0.2)(enc)

    # add 3 more conv blocks
    for i in range(conv_blocks - 1):
        enc = Conv2D(filters=filters[i+1], kernel_size=kernel_sizes[i+1], \
            strides=strides[i+1], padding=padding[i+1])(enc)
        enc = BatchNormalization()(enc)
        enc = LeakyReLU(alpha=0.2)(enc)

    # flatten the output from the last conv block
    enc = Flatten()(enc)

    # 1st fc layer
    enc = Dense(4096)(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 2nd fc layer
    enc = Dense(100)(enc)

    # create a Keras model
    enc_model = Model(inputs=[input_layer], outputs=[enc])

    return enc_model

# build generator model
def build_generator():
    # specify hyperparameters
    latent_dims = 100
    num_classes = 6

    # input layer for vector z
    input_z_noise = Input(shape=(latent_dims,))

    # input layer for conditioning variable
    input_label = Input(shape=(num_classes,))

    # concatenate the inputs along the channel dimension
    x = concatenate([input_z_noise, input_label])

    # add 1st fc block
    x = Dense(2048, input_dim=latent_dims+num_classes)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    # add 2nd fc block
    x = Dense(256 * 8 * 8)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    # reshape the output from the last dense layer to a 3D tensor
    x = Reshape((8, 8, 256))(x) # generate a tensor of dim (batch_size, 8, 8, 256)

    # add an upsampling block 
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # add another upsampling block with filter=64
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # add the last upsampling block without Batch Normalization
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=3, kernel_size=5, padding='same')(x)
    x = Activation('tanh')(x)

    # create a Keras model
    model = Model(inputs=[input_z_noise, input_label], outputs=[x])

    return model


# function to expand label_input dim
def expand_label_input(x):
    x = K.expand_dims(x, axis=1)
    x = K.expand_dims(x, axis=1)
    x = K.tile(x, [1, 32, 32, 1])
    return x


# build the discriminator network
def build_discriminator():
    # input image shape
    input_shape = (64, 64, 3)

    # input conditioning variable shape
    label_shape = (6,)

    # two input layers
    image_input = Input(shape=input_shape)
    label_input = Input(shape=label_shape)

    # add a conv block
    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(image_input)
    x = LeakyReLU(alpha=0.2)(x)

    # expand label_input to have a shape of (32, 32, 6)
    # transform a tensor with a dim of (6,) to (32, 32, 6)
    label_input1 = Lambda(expand_label_input)(label_input)

    # concatenate the transformed label tensor
    # and the output of the last conv layer along the channel dim
    x = concatenate([x, label_input1], axis=3)

    # add three conv blocks
    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # add a flatten layer
    x = Flatten()(x)

    # add a dense layers to output a probability
    x = Dense(1, activation='sigmoid')(x)

    # create a Keras model
    model = Model(inputs=[image_input, label_input], outputs=[x])

    return model

# build the face recognition model
def build_fr_model(input_shape):
    resent_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    image_input = resent_model.input 

    # get the output from last layer
    x = resent_model.layers[-1].output

    # add a dense layer
    out = Dense(128)(x) 

    # build the embedding model
    embedder_model = Model(inputs=[image_input], outputs=[out])

    # input layer
    input_layer = Input(shape=input_shape)

    # embed the input
    x = embedder_model(input_layer)
    output = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)

    model = Model(inputs=[input_layer], outputs=[output])

    return model


if __name__ == '__main__':
    # specify hyperparameters for training
    wiki_dir = 'wiki_crop'
    epochs = 500
    batch_size = 2
    image_shape = (64, 64, 3)
    z_shape = 100

    TRAIN_GAN = True
    TRAIN_ENCODER = False
    TRAIN_GAN_WITH_FR = False 

    fr_image_shape = (192, 192, 3)

    # specify optimizers for training
    # optimizer for the discriminant network
    dis_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)

    # optimizer for the generator network
    gen_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)

    # optimizer for the adversarial network
    adversarial_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)

    # build and compile the discriminator network
    discriminator = build_discriminator()
    discriminator.compile(loss=['binary_crossentropy'], optimizer=dis_optimizer)
    
    # build and compile the generator network
    generator = build_generator()
    generator.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)

    """
    build and compile the adversarial model
    """

    discriminator.trainable = False 
    
    input_z_noise = Input(shape=(100,))
    input_label = Input(shape=(6,))
    
    # get the reconstructed image from the generator
    recons_images = generator([input_z_noise, input_label])
    valid = discriminator([recons_images, input_label])

    adversarial_model = Model(inputs=[input_z_noise, input_label], outputs=[valid])
    adversarial_model.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)

    # add tensorboard to store losses
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)

    """
    load the dataset
    """

    images, age_list = load_data(wiki_dir=wiki_dir, dataset='wiki')

    # convert age to category
    age_cat = age_to_category(age_list)

    # convert the age category to one-hot encoded vectors
    final_age_cat = np.reshape(np.array(age_cat), [len(age_cat), 1])
    num_classes = len(set(age_cat))
    y = to_categorical(final_age_cat, num_classes=num_classes)

    # load all images and create an ndarray containing all images
    loaded_images = load_images(wiki_dir, images, (image_shape[0], image_shape[1]))

    # label smoothing
    real_labels = np.ones((batch_size, 1), dtype=np.float32) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype=np.float32) * 0.1

    """
    train the generator and discriminator network
    """

    if TRAIN_GAN:
        for epoch in range(epochs):
            print("-" * 50)
            print("[INFO] epoch: {}".format(epoch))

            gen_losses = []
            dis_losses = []

            num_batches = int(len(loaded_images) / batch_size)
            print("[INFO] number of batches: {}".format(num_batches))

            for i in range(num_batches):
                print("-" * 25)
                print("[INFO] batch: {}".format(i + 1))

                images_batch = loaded_images[i*batch_size: (i+1)*batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[i*batch_size :(i+1)*batch_size]
                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
                
                """
                train the discriminator network
                """

                # generate fake images
                initial_recons_images = generator.predict_on_batch([z_noise, y_batch])

                d_loss_real = discriminator.train_on_batch([images_batch, y_batch], real_labels)
                d_loss_fake = discriminator.train_on_batch([initial_recons_images, y_batch], fake_labels)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                print("[INFO] d_loss: {}".format(d_loss))

                """
                train the generator network
                """

                z_noise2 = np.random.normal(0, 1, size=(batch_size, z_shape))
                random_labels = np.random.randint(0, 6, batch_size).reshape(-1, 1)
                random_labels = to_categorical(random_labels, 6)

                g_loss = adversarial_model.train_on_batch([z_noise2, random_labels], [1]*batch_size)
                print("[INFO] g_loss: {}".format(g_loss))

                gen_losses.append(g_loss)
                dis_losses.append(d_loss)

            # write lossess to Tensorboard
            write_log(tensorboard, 'g_loss', np.mean(gen_losses), epoch)
            write_log(tensorboard, 'd_loss', np.mean(dis_losses), epoch)

            """
            generate images after every 10th epoch
            """

            if epoch%10 == 0:
                images_batch = loaded_images[0:batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[0:batch_size]
                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                gen_images = generator.predict_on_batch([z_noise, y_batch])

                for i, image in enumerate(gen_images[:5]):
                    save_rgb_image(image, path="results/image_{}_{}.png".format(epoch, i))

        # save the networks
        try:
            generator.save_weights("generator.h5")
            discriminator.save_weights("discriminator.h5")
        except Exception as e:
            print("[INFO] error: {}".format(e))

    
    """
    train encoder network
    """

    if TRAIN_ENCODER:
        # build and compile encoder
        encoder = build_encoder()
        encoder.compile(loss=euclidean_distance_loss, optimizer='adam')

        # load the generator network's weights
        try:
            generator.load_weights("generator.h5")
        except Exception as e:
            print("[INFO] error: {}".format(e))

        z_i = np.random.normal(0, 1, size=(5000, z_shape))

        y = np.random.randint(low=0, high=6, size=(5000,), dtype=np.int64)
        num_classes = len(set(y))

        y = np.reshape(np.array(y), [len(y), 1])
        y = to_categorical(y, num_classes=num_classes)

        for epoch in range(epochs):
            print("-" * 50)
            print("[INFO] epoch: ", epoch)

            encoder_losses = []

            num_batches = int(z_i.shape[0] / batch_size)
            print("[INFO] number of batches: {}".format(num_batches))

            for i in range(num_batches):
                print("-" * 25)
                print("[INFO] batch: {}".format(i + 1))

                z_batch = z_i[i*batch_size : (i+1)*batch_size]
                y_batch = y[i*batch_size : (i+1)*batch_size]

                generated_images = generator.predict_on_batch([z_batch, y_batch])

                # train the encoder
                encoder_loss = encoder.train_on_batch(generated_images, z_batch)
                print("[INFO] encoder loss: {}".format(encoder_loss))

                encoder_losses.append(encoder_loss)

            # write the encoder loss to the Tensorboard
            write_log(tensorboard, "encoder_loss", np.mean(encoder_losses), epoch)

        # save the model
        encoder.save_weights("encoder.h5")

    """
    optimize the encoder and generator network
    """

    if TRAIN_GAN_WITH_FR:
        # load the encoder 
        encoder = build_encoder()
        encoder.load_weights("encoder.h5")

        # load the generator network
        generator.load_weights("generator.h5")

        image_resizer = build_image_resizer()
        image_resizer.compile(loss=['binary_crossentropy'], optimizer='adam')

        # face recognition model
        fr_model = build_fr_model(input_shape=fr_image_shape)
        fr_model.compile(loss=['binary_crossentropy'], optimizer='adam')

        # make the fr network non-trainable
        fr_model.trainable = False 

        # input layers
        input_image = Input(shape=(64, 64, 3))
        input_label = Input(shape=(6,))

        # use the encoder and generator network
        latent0 = encoder(input_image)
        gen_images = generator([latent0, input_label])

        # resize images to the desired shape
        resized_images = Lambda(lambda x: K.resize_images(gen_images, height_factor=3, width_factor=3,
                                                        data_format='channels_last'))(gen_images)
        embeddings = fr_model(resized_images)

        # create a Keras model
        fr_adversarial_model = Model(inputs=[input_image, input_label], outputs=[embeddings])

        # compile the model
        fr_adversarial_model.compile(loss=euclidean_distance_loss, optimizer=adversarial_optimizer)

        for epoch in range(epochs):
            print("-" * 50)
            print("[INFO] epoch: {}".format(epoch))
            recons_losses = []

            num_batches = int(len(loaded_images) / batch_size)

            print("[INFO] number of batches: {}".format(num_batches))

            for i in range(num_batches):
                print("-" * 25)
                print("[INFO] batch: {}".format(i + 1))

                images_batch = loaded_images[i * batch_size:(i + 1) * batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[i * batch_size:(i + 1) * batch_size]

                images_batch_resized = image_resizer.predict_on_batch(images_batch)

                real_embeddings = fr_model.predict_on_batch(images_batch_resized)

                recons_loss = fr_adversarial_model.train_on_batch([images_batch, y_batch], real_embeddings)

                print("Reconstruction loss:", recons_loss)

                recons_losses.append(recons_loss)

            # write the reconstruction loss to tensorboard
            write_log(tensorboard, "reconstruction_loss", np.mean(recons_losses), epoch)

            """
            generate images
            """

            if epoch%10 == 0:
                images_batch = loaded_images[0:batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[0:batch_size]
                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                gen_images = generator.predict_on_batch([z_noise, y_batch])

                for i, image in enumerate(gen_images[:5]):
                    save_rgb_image(image, path="results/image_opt_{}_{}".format(epoch, i))


        # save improved weights for both of the networks
        generator.save_weights("generator_optimized.h5")
        encoder.save_weights("encoder_optimized.h5")

                



from utils import *

"""
build a residual block
"""
def residual_block(x):
    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = "same"
    momentum = 0.8
    activation = "relu"

    res = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=strides, padding=padding)(x)
    res = Activation(activation=activation)(res)
    res = BatchNormalization(momentum=momentum)(res)

    res = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding)(res)
    res = BatchNormalization(momentum=momentum)(res)

    # Add res and x
    res = Add()([res, x])

    return res


"""
build a generator network
"""

def build_generator():
    # specify hyperparameters required for the generator network
    num_residual_blocks = 16
    momentum = 0.8
    input_shape = (64, 64, 3)

    # input layer of the generator network
    input_layer = Input(shape=input_shape)

    # add the pre-residual block
    gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu')(input_layer)

    # add 16 residual blocks
    res = residual_block(gen1)
    for i in range(num_residual_blocks):
        res = residual_block(res)

    # add the post-residual block
    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    gen2 = BatchNormalization(momentum=momentum)(gen2)

    # combine the output from pre-residual block (gen1)
    # and the post-residual block (gen2)
    gen3 = Add()([gen2, gen1])

    # add an upsampling block
    gen4 = UpSampling2D(size=2)(gen3)
    gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
    gen4 = Activation('relu')(gen4)

    # add another upsampling block
    gen5 = UpSampling2D(size=2)(gen4)
    gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
    gen5 = Activation('relu')(gen5)

    # output convolution layer
    gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)
    output = Activation('tanh')(gen6)

    # create a Keras model
    model = Model(inputs=[input_layer], outputs=[output], name='generator')
    
    return model

# build a discriminator network
def build_discriminator():
    leakyrelu_alpha = 0.2
    momentum = 0.8
    input_shape = (256, 256, 3)

    input_layer = Input(shape=input_shape)

    # add the first convolution block
    dis1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
    dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)

    # add the 2nd convolution block
    dis2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
    dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
    dis2 = BatchNormalization(momentum=momentum)(dis2)

    # add the third convolution block
    dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
    dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
    dis3 = BatchNormalization(momentum=momentum)(dis3)

    # add the fourth convolution block
    dis4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
    dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
    dis4 = BatchNormalization(momentum=0.8)(dis4)

    # add the fifth convolution block
    dis5 = Conv2D(256, kernel_size=3, strides=1, padding='same')(dis4)
    dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
    dis5 = BatchNormalization(momentum=momentum)(dis5)

    # add the sixth convolution block
    dis6 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
    dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
    dis6 = BatchNormalization(momentum=momentum)(dis6)

    # add the seventh convolution block
    dis7 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
    dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
    dis7 = BatchNormalization(momentum=momentum)(dis7)

    # add the eight convolution block
    dis8 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
    dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
    dis8 = BatchNormalization(momentum=momentum)(dis8)

    # add a dense layer
    dis9 = Dense(units=1024)(dis8)
    dis9 = LeakyReLU(alpha=0.2)(dis9)

    # last dense layer for classification
    output = Dense(units=1, activation='sigmoid')(dis9)

    # create a Keras model
    model = Model(inputs=[input_layer], outputs=[output], name='discriminator')
    
    return model

"""
build VGG network to extract image features
"""
def build_vgg():
    input_shape = (256, 256, 3)

    # load a pre-trained VGG19 model trained on 'Imagenet' dataset
    vgg = VGG19(weights="imagenet")
    vgg.outputs = [vgg.layers[9].output] # obtain output of layer[9]

    input_layer = Input(shape=input_shape)

    # extract features using VGG19
    features = vgg(input_layer)

    # create a Keras model
    model = Model(inputs=[input_layer], outputs=[features])
    
    return model

if __name__ == "__main__":
    data_dir = "img_align_celeba/*.*"
    epochs = 30000
    batch_size = 1
    mode = "train"

    # shape of low-resolution and high-resolution images
    low_resolution_shape = (64, 64, 3)
    high_resolution_shape = (256, 256, 3)

    # common optimizer for all networks
    common_optimizer = Adam(0.0002, 0.5)

    if mode == 'train':
        # build and compile VGG19 network to extract features
        vgg = build_vgg()
        vgg.trainable = False
        vgg.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

        # build and compile the discriminator network
        discriminator = build_discriminator()
        discriminator.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

        # build the generator network
        generator = build_generator()

        """
        build and compile the adversarial model
        """

        # input layers for high-resolution and low-resolution images
        input_high_resolution = Input(shape=high_resolution_shape)
        input_low_resolution = Input(shape=low_resolution_shape)

        # generate high-resolution images from low-resolution images
        generated_high_resolution_images = generator(input_low_resolution)

        # extract feature maps of the generated images
        features = vgg(generated_high_resolution_images)

        # make the discriminator network as non-trainable
        discriminator.trainable = False

        # get the probability of generated high-resolution fake images
        # probs represents the probability of the generated images belonging to a real dataset.
        probs = discriminator(generated_high_resolution_images)

        # create and compile an adversarial model
        adversarial_model = Model([input_low_resolution, input_high_resolution], [probs, features])
        adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=common_optimizer)

        # add Tensorboard
        tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), write_images=True, write_grads=True, write_graph=True)
        tensorboard.set_model(generator)
        tensorboard.set_model(discriminator)

        for epoch in range(epochs):
            print("-" * 50)
            print("[INFO] epoch: {}".format(epoch))

            """
            train the discriminator network
            """

            # sample a batch of images from dataset directory
            high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size, \
                                                                low_resolution_shape=low_resolution_shape, \
                                                                high_resolution_shape=high_resolution_shape)

            # normalize images 
            high_resolution_images = high_resolution_images / 127.5 - 1.
            low_resolution_images = low_resolution_images / 127.5 - 1.

            # generate high-resolution images from low-resolution images
            generated_high_resolution_images = generator.predict(low_resolution_images)

            # generate batch of real and fake labels
            real_labels = np.ones((batch_size, 16, 16, 1))
            fake_labels = np.zeros((batch_size, 16, 16, 1))

            # train the discriminator network on real and fake images
            d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)

            # calculate total discriminator loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            print("[INFO] d_loss: {}".format(d_loss))

            """
            train the generator network
            """

            # sample a batch of images from our dataset
            high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size, \
                                                                          low_resolution_shape=low_resolution_shape, \
                                                                          high_resolution_shape=high_resolution_shape)

            # normalize images
            high_resolution_images = high_resolution_images / 127.5 - 1.
            low_resolution_images = low_resolution_images / 127.5 - 1.

            # extract feature maps for real high-resolution images
            image_features = vgg.predict(high_resolution_images)

            # train the generator network
            g_loss = adversarial_model.train_on_batch([low_resolution_images, high_resolution_images], \
                                             [real_labels, image_features])

            print("[INFO] g_loss: {}".format(g_loss))

            # write the losses to Tensorboard
            write_log(tensorboard, 'g_loss', g_loss[0], epoch)
            write_log(tensorboard, 'd_loss', d_loss[0], epoch)

            # sample and save images after every 100 epochs
            if epoch%100 == 0:
                high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                low_resolution_shape=low_resolution_shape,
                                                                high_resolution_shape=high_resolution_shape)

                # normalize images
                high_resolution_images = high_resolution_images / 127.5 - 1.
                low_resolution_images = low_resolution_images / 127.5 - 1.

                generated_images = generator.predict_on_batch(low_resolution_images)

                for index, img in enumerate(generated_images):
                    save_images(low_resolution_images[index], high_resolution_images[index], img,
                                path="results/img_{}_{}".format(epoch, index))

        # save models
        generator.save_weights("generator.h5")
        discriminator.save_weights("discriminator.h5")

    if mode == 'predict':
        # build and compile the discriminator network
        discriminator = build_discriminator()

        # build the generator network
        generator = build_generator()

        # load models
        generator.load_weights("generator.h5")
        discriminator.load_weights("discriminator.h5")

        # sample 10 random images
        high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=10,
                                                                      low_resolution_shape=low_resolution_shape,
                                                                      high_resolution_shape=high_resolution_shape)

        # normalize images
        high_resolution_images = high_resolution_images / 127.5 - 1.
        low_resolution_images = low_resolution_images / 127.5 - 1.

        # generate high-resolution images from low-resolution images
        generated_images = generator.predict_on_batch(low_resolution_images)

        # save images
        for index, img in enumerate(generated_images):
            save_images(low_resolution_images[index], high_resolution_images[index], img,
                        path="results/gen_{}".format(index))

    
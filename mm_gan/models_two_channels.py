from tensorflow import keras
import tensorflow_addons as addons


SEED = 1337
DROP_OUT = 0.2
LEAKY_RELU_ALPHA = 0.2
OPTIMIZER = 'ADAM'
LEARNING_RATE = 0.0002
BETA_1 = 0.5
BETA_2 = 0.999
IMG_SHAPE = (4, 256, 256)

kernel_initializer = keras.initializers.he_normal(seed=SEED)
bias_initializer = keras.initializers.Zeros()


# ===================== U-Net =====================

def get_unet_down(out_size, normalize=True, drop_out=0.0) -> keras.Sequential:
    model = keras.Sequential()
    model.add(keras.layers.ZeroPadding2D(((1, 1), (1, 1)), data_format="channels_first"))
    model.add(keras.layers.Conv2D(out_size, (4, 4), data_format='channels_first', strides=(2, 2), padding='valid', use_bias=False))
    if normalize:
        model.add(addons.layers.InstanceNormalization(axis=1))
    model.add(keras.layers.LeakyReLU(LEAKY_RELU_ALPHA))
    if drop_out:
        model.add(keras.layers.Dropout(drop_out))
    return model


def get_unet_up(out_size, normalize=True, drop_out=0.0) -> keras.Model:
    model = keras.Sequential()
    model.add(keras.layers.Conv2DTranspose(out_size, (4, 4), (2, 2), data_format='channels_first', padding='same', use_bias=False))
    if normalize:
        model.add(addons.layers.InstanceNormalization(axis=1))
    model.add(keras.layers.ReLU())
    if drop_out:
        model.add(keras.layers.Dropout(drop_out))
    return model


def get_generator_unet() -> keras.Model:
    x = keras.layers.Input(shape=(2, 256, 256))
    down1 = get_unet_down(64, normalize=False)
    down2 = get_unet_down(128)
    down3 = get_unet_down(256)
    down4 = get_unet_down(512, drop_out=DROP_OUT)
    down5 = get_unet_down(512, drop_out=DROP_OUT)
    down6 = get_unet_down(512, drop_out=DROP_OUT)
    down7 = get_unet_down(512, drop_out=DROP_OUT)
    down8 = get_unet_down(512, normalize=False, drop_out=DROP_OUT)

    up1 = get_unet_up(512, drop_out=DROP_OUT)
    up2 = get_unet_up(512, drop_out=DROP_OUT)
    up3 = get_unet_up(512, drop_out=DROP_OUT)
    up4 = get_unet_up(512, drop_out=DROP_OUT)
    up5 = get_unet_up(256)
    up6 = get_unet_up(128)
    up7 = get_unet_up(64)

    # relu as final layer
    final = keras.Sequential([
        keras.layers.UpSampling2D(size=(2, 2), data_format='channels_first'),
        keras.layers.ZeroPadding2D(((2, 1), (2, 1)), data_format="channels_first"),
        keras.layers.Conv2D(2, (4, 4), data_format='channels_first', padding='valid'),
        keras.layers.ReLU()
    ])

    # link the unet
    d1 = down1(x)
    d2 = down2(d1)
    d3 = down3(d2)
    d4 = down4(d3)
    d5 = down5(d4)
    d6 = down6(d5)
    d7 = down7(d6)
    d8 = down8(d7)
    u1 = up1(d8)
    u1 = keras.layers.Concatenate(axis=1)([u1, d7])
    u2 = up2(u1)
    u2 = keras.layers.Concatenate(axis=1)([u2, d6])
    u3 = up3(u2)
    u3 = keras.layers.Concatenate(axis=1)([u3, d5])
    u4 = up4(u3)
    u4 = keras.layers.Concatenate(axis=1)([u4, d4])
    u5 = up5(u4)
    u5 = keras.layers.Concatenate(axis=1)([u5, d3])
    u6 = up6(u5)
    u6 = keras.layers.Concatenate(axis=1)([u6, d2])
    u7 = up7(u6)
    u7 = keras.layers.Concatenate(axis=1)([u7, d1])
    f = final(u7)
    model = keras.Model(inputs=[x], outputs=[f])
    return model


# ===================== Discriminator =====================

def get_discriminator_block(out_filters, normalization=True) -> keras.Sequential:
    model = keras.Sequential()
    model.add(keras.layers.ZeroPadding2D(((1, 1), (1, 1)), data_format="channels_first"))
    model.add(keras.layers.Conv2D(out_filters, (4, 4), data_format='channels_first', strides=(2, 2), padding='valid'))
    if normalization:
        model.add(addons.layers.InstanceNormalization(axis=1))
    model.add(keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA))
    return model


def get_discriminator() -> keras.Model:
    img_in = keras.layers.Input(shape=(1, 256, 256), name='img_in')
    x = get_discriminator_block(64, normalization=False)(img_in)
    x = get_discriminator_block(128)(x)
    x = get_discriminator_block(256)(x)
    x = get_discriminator_block(512)(x)
    x = keras.layers.ZeroPadding2D(((2, 1), (2, 1)), data_format='channels_first')(x)
    y = keras.layers.Conv2D(1, (4, 4), data_format='channels_first', padding='valid', use_bias=False)(x)
    model = keras.Model(inputs=[img_in], outputs=[y])
    return model


if __name__ == '__main__':
    generator = get_generator_unet()
    generator.compile(optimizer=OPTIMIZER, loss='mse', metrics=['accuracy'])
    generator.summary()
    discriminator = get_discriminator()
    discriminator.compile(optimizer=OPTIMIZER, loss='mse', metrics=['accuracy'])
    discriminator.summary()
    from tensorflow.keras.utils import plot_model
    plot_model(generator, to_file='../generator.png', show_shapes=True, rankdir='TB', expand_nested=True, dpi=256)
    plot_model(discriminator, to_file='../discriminator.png', show_shapes=True, rankdir='TB', expand_nested=True, dpi=256)


from tensorflow.keras import activations
from tensorflow.keras.initializers import he_normal, Zeros
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers \
    import Conv2D, Dropout, LeakyReLU, Conv2DTranspose, ReLU, Concatenate, UpSampling2D, ZeroPadding2D
from tensorflow.keras.activations import sigmoid
from tensorflow_addons.layers import InstanceNormalization

SEED = 1337
DROP_OUT = 0.2
LEAKY_RELU_ALPHA = 0.2
OPTIMIZER = 'ADAM'
LEARNING_RATE = 0.0002
BETA_1 = 0.5
BETA_2 = 0.999
IMG_SHAPE = (4, 256, 256)

kernel_initializer = he_normal(seed=SEED)
bias_initializer = Zeros()


# ===================== U-Net =====================

def get_unet_down(out_size, normalize=True, drop_out=0.0) -> Sequential:
    model = Sequential()
    model.add(ZeroPadding2D(((1, 1), (1, 1)), data_format="channels_first"))
    model.add(Conv2D(out_size, (4, 4), data_format='channels_first', strides=(2, 2), padding='valid', use_bias=False, kernel_initializer=kernel_initializer))
    if normalize:
        model.add(InstanceNormalization(axis=1))
    model.add(LeakyReLU(LEAKY_RELU_ALPHA))
    if drop_out:
        model.add(Dropout(drop_out))
    return model


def get_unet_up(out_size, normalize=True, drop_out=0.0) -> Model:
    '''
    direct_input = Input(shape=input_shape, name='direct_input')
    skip_input = Input(shape=input_shape, name='skip_input')
    x = Concatenate(axis=1)([direct_input, skip_input])
    x = Conv2DTranspose(out_size, (4, 4), (2, 2), data_format='channels_first', padding='same', use_bias=False)(x)
    if normalize:
        x = InstanceNormalization(axis=1)(x)
    x = ReLU()(x)
    if drop_out:
        x = Dropout(drop_out)(x)
    model = Model(inputs=[direct_input, skip_input], outputs=[x])
    '''
    model = Sequential()
    model.add(Conv2DTranspose(out_size, (4, 4), (2, 2), data_format='channels_first', padding='same', use_bias=False, kernel_initializer=kernel_initializer, bias_initializer = bias_initializer))
    if normalize:
        model.add(InstanceNormalization(axis=1))
    model.add(ReLU())
    if drop_out:
        model.add(Dropout(drop_out))
    return model


def get_generator_unet(input_shape, out_channels=4) -> Model:
    x = Input(shape=input_shape)
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
    final = Sequential([
        UpSampling2D(size=(2, 2), data_format='channels_first'),
        ZeroPadding2D(((2, 1), (2, 1)), data_format="channels_first"),
        Conv2D(out_channels, (4, 4), data_format='channels_first', padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer),
        ReLU()  # relu only provides positive output
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
    u1 = Concatenate(axis=1)([u1, d7])
    u2 = up2(u1)
    u2 = Concatenate(axis=1)([u2, d6])
    u3 = up3(u2)
    u3 = Concatenate(axis=1)([u3, d5])
    u4 = up4(u3)
    u4 = Concatenate(axis=1)([u4, d4])
    u5 = up5(u4)
    u5 = Concatenate(axis=1)([u5, d3])
    u6 = up6(u5)
    u6 = Concatenate(axis=1)([u6, d2])
    u7 = up7(u6)
    u7 = Concatenate(axis=1)([u7, d1])
    f = final(u7)
    model = Model(inputs=[x], outputs=[f])
    return model


# ===================== Discriminator =====================

def get_discriminator_block(out_filters, normalization=True) -> Sequential:
    model = Sequential()
    model.add(ZeroPadding2D(((1, 1), (1, 1)), data_format="channels_first"))
    model.add(Conv2D(out_filters, (4, 4), data_format='channels_first', strides=(2, 2), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer))
    if normalization:
        model.add(InstanceNormalization(axis=1))
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
    return model


def get_discriminator(input_shape, out_channels=4) -> Model:
    img_a = Input(shape=input_shape, name='img_A')
    img_b = Input(shape=input_shape, name='img_B')
    img_in = Concatenate(axis=1)([img_a, img_b])
    x = get_discriminator_block(64, normalization=False)(img_in)
    x = get_discriminator_block(128)(x)
    x = get_discriminator_block(256)(x)
    x = get_discriminator_block(512)(x)
    x = ZeroPadding2D(((2, 1), (2, 1)), data_format='channels_first')(x)
    y = Conv2D(out_channels, (4, 4), data_format='channels_first', padding='valid', use_bias=False, kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)
    y = sigmoid(y)
    model = Model(inputs=[img_a, img_b], outputs=[y])
    return model


if __name__ == '__main__':
    generator = get_generator_unet((4, 256, 256))
    generator.compile(optimizer=OPTIMIZER, loss='mse', metrics=['accuracy'])
    generator.summary()
    discriminator = get_discriminator((4, 256, 256))
    discriminator.compile(optimizer=OPTIMIZER, loss='mse', metrics=['accuracy'])
    discriminator.summary()
    from tensorflow.keras.utils import plot_model
    plot_model(generator, to_file='../generator.png', show_shapes=True, rankdir='TB', expand_nested=True, dpi=256)
    plot_model(discriminator, to_file='../discriminator.png', show_shapes=True, rankdir='TB', expand_nested=True, dpi=256)

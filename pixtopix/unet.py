import tensorflow as tf


def conv2d_block(input_tensor, n_filters, kernel_size=3):
    x = input_tensor
    for i in range(1):
        x = tf.keras.layers.Conv2D(filters=n_filters, strides=(2, 2), kernel_size=(kernel_size, kernel_size),
                                   kernel_initializer='he_normal', padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)

    return x


def encoder_block(inputs, n_filters=64):
    initializer = tf.random_normal_initializer(0., 0.02)
    f = tf.keras.layers.Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer)(
        inputs)
    f = tf.keras.layers.LeakyReLU(alpha=0.2)(f)
    return f, n_filters


def encoder(inputs):
    START_SIZE = 32
    e1 = encoder_block(inputs, n_filters=START_SIZE)
    e2 = encoder_block(e1[0], n_filters=START_SIZE * 2)
    e3 = encoder_block(e2[0], n_filters=START_SIZE * 4)
    e4 = encoder_block(e3[0], n_filters=START_SIZE * 8)
    e5 = encoder_block(e4[0], n_filters=START_SIZE * 8)
    e6 = encoder_block(e5[0], n_filters=START_SIZE * 8)
    e7 = encoder_block(e6[0], n_filters=START_SIZE * 8)

    return [e1, e2, e3, e4, e5, e6, e7]


def bottleneck(inputs):
    """
    This function defines the bottleneck convolutions to extract more features before the upsampling layers.
    """

    bottle_neck = conv2d_block(inputs, n_filters=512, kernel_size=4)

    return bottle_neck


def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, padding='same', dropout=0.3):
    """
    defines the one decoder block of the UNet

    Args:
      inputs (tensor) -- batch of input features
      conv_output (tensor) -- features from an encoder block
      n_filters (int) -- number of filters
      kernel_size (int) -- kernel size
      strides (int) -- strides for the deconvolution/upsampling
      padding (string) -- "same" or "valid", tells if shape will be preserved by zero padding

    Returns:
      c (tensor) -- output features of the decoder block
    """

    initializer = tf.random_normal_initializer(0., 0.02)
    u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides=strides, padding=padding,
                                        kernel_initializer=initializer,
                                        use_bias=True)(inputs)
    # u = tf.keras.layers.BatchNormalization()(u)
    if dropout != 0:
        u = tf.keras.layers.Dropout(dropout)(u)

    u_shape = u.shape[1:3]
    c_shape = conv_output.shape[1:3]
    offset = abs(c_shape[0] - u_shape[0]), abs(c_shape[1] - u_shape[1])

    # pad the smaller feature map
    if c_shape[0] > u_shape[0] or c_shape[1] - u_shape[1] > 0:
        u = tf.pad(u, [(0, 0), (0, offset[0]), (0, offset[1]), (0, 0)])
    else:
        conv_output = tf.pad(conv_output, [(0, 0), (0, offset[0]), (0, offset[1]), (0, 0)])
    c = tf.keras.layers.concatenate([u, conv_output])
    c = tf.keras.layers.Activation('relu')(c)

    return c


def decoder(inputs, convs, output_channels, greater_than_127):
    '''
    Defines the decoder of the UNet chaining together 4 decoder blocks.

    Args:
      inputs (tensor) -- batch of input features
      convs (tuple) -- features from the encoder blocks
      output_channels (int) -- number of classes in the label map

    Returns:
      outputs (tensor) -- the pixel wise label map of the image
    '''

    init = tf.random_normal_initializer(0., 0.02)
    if greater_than_127:
        S = (2, 2)
    else:
        S = (1, 1)

    dec_block = decoder_block(inputs, convs[-1][0], n_filters=convs[-1][1], kernel_size=(4, 4), strides=S, dropout=0.3)
    for i in range(2, len(convs) + 1):
        dropout = 0.3 if i < 5 else 0
        if i == 2:
            dec_block = decoder_block(dec_block, convs[-i][0], n_filters=convs[-i][1], kernel_size=(4, 4), strides=S,
                                      dropout=dropout)
        else:
            dec_block = decoder_block(dec_block, convs[-i][0], n_filters=convs[-i][1], kernel_size=(4, 4),
                                      strides=(2, 2), dropout=dropout)

    outputs = tf.keras.layers.Conv2DTranspose(output_channels, (4, 4), padding='same', strides=(2, 2),
                                              kernel_initializer=init, activation='tanh')(dec_block)

    return outputs


def unet(IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS=3):
    '''
    Defines the UNet by connecting the encoder, bottleneck and decoder.
    '''
    # specify the input shape
    inputs = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # feed the inputs to the encoder
    encoder_outputs = encoder(inputs)

    # feed the encoder output to the bottleneck
    bottle_neck = bottleneck(encoder_outputs[-1][0])

    # feed the bottleneck and encoder block outputs to the decoder
    # specify the number of classes via the `output_channels` argument
    outputs = decoder(bottle_neck, encoder_outputs, output_channels=OUTPUT_CHANNELS, greater_than_127=IMG_WIDTH>=127)

    # create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model



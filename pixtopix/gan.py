import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

from pixtopix.discriminator import discriminator
from pixtopix.unet import unet


def define_gan(generator_model, discriminator_model, image_shape):
    for layer in discriminator_model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    in_src = Input(shape=image_shape)
    gen_out = generator_model(in_src)
    dis_out = discriminator_model([in_src, gen_out])
    model = Model(in_src, [dis_out, gen_out])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt,
                  loss_weights=[1, 100])  # here 100 is the Lambda in the other notebook
    return model


# input images size
w, h = 256, 256
image_channels = 3
generator = unet(h, w)
discriminator_object = discriminator(w, h)
GAN = define_gan(generator, discriminator_object, (w, h, image_channels))

print("GAN init. done")
# You can now starting training the network

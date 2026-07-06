"""Encoder, decoder, and latent fluid-dynamics network definitions."""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, UpSampling2D


def build_encoder(input_shape=(128, 160, 1), latent_dim=4, show_summary=True):
    """Build the CNN encoder reported in the CAE table."""
    encoder_input = tf.keras.Input(shape=input_shape, name="1_flowimg")

    encoded = Conv2D(filters=32, kernel_size=(3, 3), activation="tanh", padding="same")(encoder_input)
    encoded = MaxPooling2D((2, 2), padding="same")(encoded)

    encoded = Conv2D(filters=16, kernel_size=(3, 3), activation="tanh", padding="same")(encoded)
    encoded = MaxPooling2D((2, 2), padding="same")(encoded)

    encoded = Conv2D(filters=8, kernel_size=(3, 3), activation="tanh", padding="same")(encoded)
    encoded = MaxPooling2D((2, 2), padding="same")(encoded)

    encoded = Conv2D(filters=8, kernel_size=(3, 3), activation="tanh", padding="same")(encoded)
    encoded = MaxPooling2D((2, 2), padding="same")(encoded)

    encoded = Conv2D(filters=4, kernel_size=(3, 3), activation="tanh", padding="same")(encoded)
    encoded = MaxPooling2D((2, 2), padding="same")(encoded)

    encoded = Conv2D(filters=4, kernel_size=(3, 3), activation="tanh", padding="same")(encoded)
    encoded = MaxPooling2D((2, 2), padding="same")(encoded)

    encoded = tf.keras.layers.Flatten()(encoded)
    encoder_output = tf.keras.layers.Dense(latent_dim, activation="linear")(encoded)
    encoder = tf.keras.Model(encoder_input, encoder_output, name="encoder")

    if show_summary:
        print("\n====== Encoder Model Summary ======")
        encoder.summary()
    return encoder


def build_decoder(latent_dim=4, show_summary=True):
    """Build the CNN decoder reported in the CAE table."""
    decoder_input = tf.keras.Input(shape=(latent_dim,), name="3_next_fluid")

    decoded = tf.keras.layers.Dense(2 * 3 * 4, activation="tanh")(decoder_input)
    decoded = Reshape((2, 3, 4))(decoded)

    decoded = UpSampling2D((2, 2))(decoded)
    decoded = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 1)))(decoded)
    decoded = Conv2D(filters=4, kernel_size=(3, 3), activation="tanh", padding="same")(decoded)

    decoded = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(filters=8, kernel_size=(3, 3), activation="tanh", padding="same")(decoded)

    decoded = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(filters=8, kernel_size=(3, 3), activation="tanh", padding="same")(decoded)

    decoded = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(filters=16, kernel_size=(3, 3), activation="tanh", padding="same")(decoded)

    decoded = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(filters=32, kernel_size=(3, 3), activation="tanh", padding="same")(decoded)

    decoded = UpSampling2D((2, 2))(decoded)
    decoder_output = Conv2D(1, kernel_size=(3, 3), padding="same", activation="linear")(decoded)
    decoder = tf.keras.Model(decoder_input, decoder_output, name="decoder")

    if show_summary:
        print("\n====== Decoder Model Summary ======")
        decoder.summary()
    return decoder


def build_fluid_dynamics(latent_dim=4, fluid_dynamic_dim=128, show_summary=True):
    """Build the latent-space fluid dynamics prediction module."""
    fluid_input = tf.keras.Input(shape=(latent_dim + 1,), name="2_fluid")

    fluid_layer = tf.keras.layers.Dense(fluid_dynamic_dim, activation="relu")(fluid_input)
    fluid_layer = tf.keras.layers.Dense(fluid_dynamic_dim, activation="relu")(fluid_layer)
    fluid_layer = tf.keras.layers.Dense(fluid_dynamic_dim, activation="relu")(fluid_layer)
    fluid_output = tf.keras.layers.Dense(latent_dim, activation="linear")(fluid_layer)

    fluid_dynamics = tf.keras.Model(fluid_input, fluid_output, name="fluid_dynamics")

    if show_summary:
        print("\n====== Fluid Dynamics Model Summary ======")
        fluid_dynamics.summary()
    return fluid_dynamics

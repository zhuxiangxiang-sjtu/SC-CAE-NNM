"""Pressure-lift structure update used in the strong-coupling loop."""

import numpy as np
import tensorflow as tf


def denormalize_channel_tensor(normalized_field, scaler):
    """Convert one normalized CNN output channel back to physical units."""
    height = normalized_field.shape[1]
    width = normalized_field.shape[2]
    data_min = tf.constant(np.reshape(scaler.data_min_, (1, height, width, 1)), dtype=tf.float64)
    data_max = tf.constant(np.reshape(scaler.data_max_, (1, height, width, 1)), dtype=tf.float64)
    feature_min = tf.constant(scaler.feature_range[0], dtype=tf.float64)
    feature_max = tf.constant(scaler.feature_range[1], dtype=tf.float64)
    scale = (data_max - data_min) / (feature_max - feature_min)
    offset = data_min - feature_min * scale
    return normalized_field * scale + offset


def structure_dynamics_from_pressure(
    x_grid,
    y_grid,
    y_feature,
    v_feature,
    dec_p_data,
    pressure_scaler,
    grid_height,
):
    """Update the cylinder response using pressure-induced lift only."""
    p_physical = denormalize_channel_tensor(dec_p_data, pressure_scaler)[:, :, :, 0]

    surface_pressure = p_physical[:, :, 0]
    x_circle = x_grid[:, :, 0]
    y_circle = y_grid[:, :, 0]

    row = grid_height
    d_t = 0.05
    m_body = 2.950307
    c_body = 0.074149
    k_body = 25.803348
    dtheta = 2 * np.pi / (row - 1)
    num_timesteps = y_feature.shape[0]

    lift_array = tf.TensorArray(dtype=tf.float64, size=num_timesteps)
    for time_index in range(num_timesteps):
        x_surface = x_circle[time_index, :]
        y_surface = y_circle[time_index, :]
        pressure = surface_pressure[time_index, :]

        x_origin = x_surface - (tf.reduce_max(x_surface) + tf.reduce_min(x_surface)) / 2
        y_origin = y_surface - (tf.reduce_max(y_surface) + tf.reduce_min(y_surface)) / 2
        diameter = tf.reduce_max(y_origin) - tf.reduce_min(y_origin)
        radius = diameter / 2
        ny = y_origin / radius

        pressure_lift = -pressure * ny * radius * dtheta
        lift_total = tf.reduce_sum(pressure_lift)
        lift_array = lift_array.write(time_index, lift_total)

    lift_total = tf.reshape(lift_array.stack(), (num_timesteps, 1))
    v_feature = tf.reshape(v_feature, (num_timesteps, 1))
    y_feature = tf.reshape(y_feature, (num_timesteps, 1))

    damping_force = v_feature * c_body
    spring_force = y_feature * k_body
    net_force = lift_total - damping_force - spring_force
    acceleration = net_force / m_body

    v_body = v_feature + acceleration * d_t
    y_body = y_feature + v_feature * d_t
    return y_body, v_body

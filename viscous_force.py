"""Velocity-gradient and viscous-force calculations for the structure update."""

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


def finite_difference_axis(field, axis):
    """Finite difference for tensors with shape (batch, row, col)."""
    if axis == 1:
        first = field[:, 1:2, :] - field[:, 0:1, :]
        middle = 0.5 * (field[:, 2:, :] - field[:, :-2, :])
        last = field[:, -1:, :] - field[:, -2:-1, :]
        return tf.concat([first, middle, last], axis=1)

    first = field[:, :, 1:2] - field[:, :, 0:1]
    middle = 0.5 * (field[:, :, 2:] - field[:, :, :-2])
    last = field[:, :, -1:] - field[:, :, -2:-1]
    return tf.concat([first, middle, last], axis=2)


def physical_gradients(x_grid, y_grid, field):
    """Compute d(field)/dX and d(field)/dY on a curvilinear XY grid."""
    dfield_di = finite_difference_axis(field, axis=1)
    dfield_dj = finite_difference_axis(field, axis=2)
    dX_di = finite_difference_axis(x_grid, axis=1)
    dX_dj = finite_difference_axis(x_grid, axis=2)
    dY_di = finite_difference_axis(y_grid, axis=1)
    dY_dj = finite_difference_axis(y_grid, axis=2)

    jacobian = dX_di * dY_dj - dX_dj * dY_di
    jacobian = tf.where(tf.abs(jacobian) < 1.0e-12, tf.ones_like(jacobian) * 1.0e-12, jacobian)

    dfield_dX = (dfield_di * dY_dj - dfield_dj * dY_di) / jacobian
    dfield_dY = (dX_di * dfield_dj - dX_dj * dfield_di) / jacobian
    return dfield_dX, dfield_dY


def structure_dynamics_from_velocity_gradients(
    x_grid,
    y_grid,
    y_feature,
    v_feature,
    dec_u_data,
    dec_v_data,
    dec_p_data,
    scalers,
    fluid_viscosity,
    grid_height,
):
    """Update structure response using pressure lift and viscous lift."""
    u_physical = denormalize_channel_tensor(dec_u_data, scalers[0])[:, :, :, 0]
    v_physical = denormalize_channel_tensor(dec_v_data, scalers[1])[:, :, :, 0]
    p_physical = denormalize_channel_tensor(dec_p_data, scalers[2])[:, :, :, 0]

    _, dUdY = physical_gradients(x_grid, y_grid, u_physical)
    dVdX, dVdY = physical_gradients(x_grid, y_grid, v_physical)

    surface_pressure = p_physical[:, :, 0]
    surface_dUdY = dUdY[:, :, 0]
    surface_dVdX = dVdX[:, :, 0]
    surface_dVdY = dVdY[:, :, 0]
    x_circle = x_grid[:, :, 0]
    y_circle = y_grid[:, :, 0]

    row = grid_height
    d_t = 0.05
    m_body = 2.950307
    c_body = 0.074149
    k_body = 25.803348
    dtheta = 2 * np.pi / (row - 1)
    num_timesteps = y_feature.shape[0]

    lift_total_array = tf.TensorArray(dtype=tf.float64, size=num_timesteps)
    for time_index in range(num_timesteps):
        x_surface = x_circle[time_index, :]
        y_surface = y_circle[time_index, :]
        pressure = surface_pressure[time_index, :]
        dUdY_wall = surface_dUdY[time_index, :]
        dVdX_wall = surface_dVdX[time_index, :]
        dVdY_wall = surface_dVdY[time_index, :]

        x_origin = x_surface - (tf.reduce_max(x_surface) + tf.reduce_min(x_surface)) / 2
        y_origin = y_surface - (tf.reduce_max(y_surface) + tf.reduce_min(y_surface)) / 2
        diameter = tf.reduce_max(y_origin) - tf.reduce_min(y_origin)
        radius = diameter / 2
        nx = x_origin / radius
        ny = y_origin / radius

        pressure_lift = -pressure * ny * radius * dtheta
        viscous_lift = (
            2 * fluid_viscosity * dVdY_wall * ny
            + fluid_viscosity * (dUdY_wall + dVdX_wall) * nx
        ) * radius * dtheta
        lift_total = tf.reduce_sum(pressure_lift + viscous_lift)
        lift_total_array = lift_total_array.write(time_index, lift_total)

    lift_total = tf.reshape(lift_total_array.stack(), (num_timesteps, 1))
    v_feature = tf.reshape(v_feature, (num_timesteps, 1))
    y_feature = tf.reshape(y_feature, (num_timesteps, 1))

    damping_force = v_feature * c_body
    spring_force = y_feature * k_body
    net_force = lift_total - damping_force - spring_force
    acceleration = net_force / m_body

    v_body = v_feature + acceleration * d_t
    y_body = y_feature + v_feature * d_t
    return y_body, v_body

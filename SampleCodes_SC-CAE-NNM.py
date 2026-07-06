# ****************************************************************************************#
# *      Strong-coupling Covolution Autoencoder Nonlinear Normal Mode (SC-CAE-NNM)       *#
# *          School Ocean and Civil Engineering, Shanghai Jiao Tong University           *#
# *                               Author by Xiangxiang Zhu                               *#
# *                             MODIFIED: 18 MAY 2025, 18:24                             *#
# ****************************************************************************************#


import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory to grow as needed
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Add, Reshape
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import h5py 

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

tf.keras.backend.set_floatx('float64')

myseed = 135

def reset_random_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    
reset_random_seeds(myseed)

# ****************************************************************************************#
#                               [1] importing data                                       #
# ****************************************************************************************#

# 读取结构数据
file_path1 = './Data/VIV_displacement_velocity_lift_drag_Re100_hrk2.mat'

with h5py.File(file_path1, 'r') as f:
    drag_structure = np.array(f['drag_structure']).T
    drag_viscous_structure = np.array(f['drag_viscous_structure']).T
    eta_structure = np.array(f['eta_structure']).T
    lift_structure = np.array(f['lift_structure']).T
    velocity_structure = np.array(f['velocity_structure']).T

print('velocity_structure维度:', velocity_structure.shape)

# 读取流场数据 - 调整为新的CNN格式数据
file_path2 = './Data/VIV_1dof_Re100_XY_UVP.mat'

with h5py.File(file_path2, 'r') as f:
    XY = np.array(f['XY']).T  # expected shape: (n_snapshots, 128, 160, 2)
    UVP = np.array(f['UVP']).T  # expected shape: (n_snapshots, 128, 160, 3)

print('XY维度:', XY.shape)
print('UVP维度:', UVP.shape)

# 分离XY和UVP
# XY = XYUVP[:, :, :, 0:2]  # (2000, 128, 160, 2)
# UVP = XYUVP[:, :, :, 2:5]  # (2000, 128, 160, 3)

print('XY维度:', XY.shape)
print('UVP维度:', UVP.shape)

file_path3 = './models/trial1/'

# ****************************************************************************************#
#                                 [2] 定义 nnmVIV类别                                     #
# ****************************************************************************************#

class nnmVIV:
    def __init__(self, input_shape, encode_dim, latent_dim, fluid_dynamic_dim, struc_output_dim):

        self.input_shape = input_shape  # (128, 160, 1) for the full-field input
        self.encode_dim = encode_dim
        self.latent_dim = latent_dim
        self.fluid_dynamic_dim = fluid_dynamic_dim
        self.struc_output_dim = struc_output_dim
        self.grid_height = input_shape[0]
        self.grid_width = input_shape[1]
        self.fluid_viscosity = tf.constant(0.01, dtype=tf.float64)  # Re=100: mu = 1 / Re

        # Section 2.3.3: Adam starts from 1e-3, decays exponentially,
        # and remains fixed at 1e-5 for the last 2000 epochs.
        self.initial_learning_rate = 1.0e-3
        self.final_learning_rate = 1.0e-5
        self.fixed_lr_epochs = 2000

        # optimizer
        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate)
        self.decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate)
        self.fluid_dynamics_optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate)

        self.encoder = self.build_encoder()  # 定义编码器
        self.decoder = self.build_decoder()  # 定义解码器
        self.fluid_dynamics = self.build_fluid_dynamics()  # 定义流体动力学演化块

        # 定义损失函数
        self.mse = tf.keras.losses.MeanSquaredError()

    def learning_rate_for_epoch(self, epoch, num_epochs):
        """Exponential decay followed by a fixed fine-tuning learning rate."""
        decay_epochs = max(num_epochs - self.fixed_lr_epochs, 1)
        if epoch >= decay_epochs:
            return self.final_learning_rate

        decay_ratio = self.final_learning_rate / self.initial_learning_rate
        return self.initial_learning_rate * (decay_ratio ** (epoch / decay_epochs))

    def set_optimizer_learning_rate(self, learning_rate):
        """Use the same learning rate for encoder, decoder, and fluid DP module."""
        self.encoder_optimizer.learning_rate.assign(learning_rate)
        self.decoder_optimizer.learning_rate.assign(learning_rate)
        self.fluid_dynamics_optimizer.learning_rate.assign(learning_rate)

    # ****************************************************************************************#
    #                                     [3] 归一化数据                                       #
    # ****************************************************************************************#
    def normalize_data(self, XY_initial, UVP_initial, velocity_structure_initial,
                       train_snapshots=1400, validation_snapshots=400, test_snapshots=500):
        """Normalize the flow field and split snapshots as reported in the paper."""
        first_dim = UVP_initial.shape[0]
        num_rows = UVP_initial.shape[1]
        num_cols = UVP_initial.shape[2]
        required_snapshots = train_snapshots + validation_snapshots + test_snapshots
        if first_dim < required_snapshots:
            raise ValueError(
                "The paper split requires at least "
                f"{required_snapshots} snapshots "
                f"({train_snapshots} train + {validation_snapshots} validation + "
                f"{test_snapshots} test), but UVP contains {first_dim} snapshots."
            )
        train_end = train_snapshots
        validation_end = train_snapshots + validation_snapshots
        test_end = required_snapshots

        # 重塑UVP用于归一化
        reshaped_u = UVP_initial[:, :, :, 0].reshape(first_dim, -1)
        reshaped_v = UVP_initial[:, :, :, 1].reshape(first_dim, -1)
        reshaped_p = UVP_initial[:, :, :, 2].reshape(first_dim, -1)

        # 1. 归一化UVP数据 - 存储scaler对象
        scaler_u = MinMaxScaler(feature_range=(-1, 1))
        scaler_v = MinMaxScaler(feature_range=(-1, 1))
        scaler_p = MinMaxScaler(feature_range=(-1, 1))

        # Fit scalers on training snapshots only to avoid validation/test leakage.
        scaler_u.fit(reshaped_u[:train_end])
        U_norm = scaler_u.transform(reshaped_u).reshape(first_dim, num_rows, num_cols, 1)
        scaler_v.fit(reshaped_v[:train_end])
        V_norm = scaler_v.transform(reshaped_v).reshape(first_dim, num_rows, num_cols, 1)
        scaler_p.fit(reshaped_p[:train_end])
        P_norm = scaler_p.transform(reshaped_p).reshape(first_dim, num_rows, num_cols, 1)

        # 2. 将归一化后的数据重新组合
        UVP_norm = np.concatenate((U_norm, V_norm, P_norm), axis=-1)  # (n_snapshots, 128, 160, 3)
        XYUVP_norm = np.concatenate((XY_initial, UVP_norm), axis=-1)  # (n_snapshots, 128, 160, 5)
         
        # 3. 流场训练和测试数据分割
        # Flow-field split: 1400 train, 400 validation, 500 test.
        XYUVP_train = XYUVP_norm[:train_end, :, :, :]
        XYUVP_valid = XYUVP_norm[train_end:validation_end, :, :, :]
        XYUVP_test = XYUVP_norm[validation_end:test_end, :, :, :]

        # 4. 结构训练和测试数据分割
        # Structure-velocity split aligned with the flow-field snapshots.
        velocity_train = velocity_structure_initial[:train_end, :]
        velocity_valid = velocity_structure_initial[train_end:validation_end, :]
        velocity_test = velocity_structure_initial[validation_end:test_end, :]

        # 将scaler对象存储在列表中
        scalers = [scaler_u, scaler_v, scaler_p]

        # 计算最小最大值
        minn_u = np.reshape(np.min(reshaped_u[validation_end:test_end], axis=0), (-1, 1))
        maxx_u = np.reshape(np.max(reshaped_u[validation_end:test_end], axis=0), (-1, 1))
        minn_v = np.reshape(np.min(reshaped_v[validation_end:test_end], axis=0), (-1, 1))
        maxx_v = np.reshape(np.max(reshaped_v[validation_end:test_end], axis=0), (-1, 1))
        minn_p = np.reshape(np.min(reshaped_p[validation_end:test_end], axis=0), (-1, 1))
        maxx_p = np.reshape(np.max(reshaped_p[validation_end:test_end], axis=0), (-1, 1))
        
        minn = np.concatenate((minn_u, minn_v, minn_p), axis=-1)
        maxx = np.concatenate((maxx_u, maxx_v, maxx_p), axis=-1)

        return (minn, maxx,
                XYUVP_train, XYUVP_valid, XYUVP_test,
                velocity_train, velocity_valid, velocity_test,
                scalers)

    # ****************************************************************************************#
    #                                          反归一化数据                                    #
    # ****************************************************************************************#
    def denormalize_uvp(self, UVP_normalized, scalers):
        # 需要先重塑数据以进行反归一化
        shape = UVP_normalized.shape
        UVP_original = np.zeros_like(UVP_normalized)
        
        # 分别反归一化每个分量
        for j in range(3):
            reshaped = UVP_normalized[:, :, :, j].reshape(shape[0], -1)
            unscaled = scalers[j].inverse_transform(reshaped)
            UVP_original[:, :, :, j] = unscaled.reshape(shape[0], shape[1], shape[2])
            
        return UVP_original

    def build_encoder(self):
        """构建CNN编码器模型"""
        encoder_input = tf.keras.Input(shape=self.input_shape, name='1_flowimg')
        # Encoder Model 1st: 128x160 -> 64x80
        encoded = Conv2D(filters=32, kernel_size=(3, 3), activation='tanh', padding='same')(encoder_input)
        encoded = MaxPooling2D((2, 2), padding='same')(encoded)
        
        # Encoder Model 2nd: 64x80 -> 32x40
        encoded = Conv2D(filters=16, kernel_size=(3, 3), activation='tanh', padding='same')(encoded)
        encoded = MaxPooling2D((2, 2), padding='same')(encoded)
        
        # Encoder Model 3rd: 32x40 -> 16x20
        encoded = Conv2D(filters=8, kernel_size=(3, 3), activation='tanh', padding='same')(encoded)
        encoded = MaxPooling2D((2, 2), padding='same')(encoded)
        
        # Encoder Model 4th: 16x20 -> 8x10
        encoded = Conv2D(filters=8, kernel_size=(3, 3), activation='tanh', padding='same')(encoded)
        encoded = MaxPooling2D((2, 2), padding='same')(encoded)
        
        # Encoder Model 5th: 8x10 -> 4x5
        encoded = Conv2D(filters=4, kernel_size=(3, 3), activation='tanh', padding='same')(encoded)
        encoded = MaxPooling2D((2, 2), padding='same')(encoded)
        
        # Encoder Model 6th: 4x5 -> 2x3
        encoded = Conv2D(filters=4, kernel_size=(3, 3), activation='tanh', padding='same')(encoded)
        encoded = MaxPooling2D((2, 2), padding='same')(encoded)
        # Flatten
        encoded = tf.keras.layers.Flatten()(encoded)
        encoder_output = tf.keras.layers.Dense(self.latent_dim, activation='linear')(encoded)
        encoder = tf.keras.Model(encoder_input, encoder_output, name='encoder')
        # 打印模型摘要
        print("\n====== Encoder Model Summary ======")
        encoder.summary()
        
        return encoder

    def build_fluid_dynamics(self):
        """构建流体动力学模型"""
        fluid_dynamic_input = tf.keras.Input(shape=(self.latent_dim + 1), name='2_fluid')

        fluid_dynamic_layer = tf.keras.layers.Dense(self.fluid_dynamic_dim, activation='relu')(fluid_dynamic_input)
        fluid_dynamic_layer = tf.keras.layers.Dense(self.fluid_dynamic_dim, activation='relu')(fluid_dynamic_layer)
        fluid_dynamic_layer = tf.keras.layers.Dense(self.fluid_dynamic_dim, activation='relu')(fluid_dynamic_layer)
        
        fluid_dynamic_output = tf.keras.layers.Dense(self.latent_dim, activation='linear')(fluid_dynamic_layer)
        fluid_dynamics = tf.keras.Model(fluid_dynamic_input, fluid_dynamic_output, name='fluid_dynamics')
        
        # 打印模型摘要
        print("\n====== Fluid Dynamics Model Summary ======")
        fluid_dynamics.summary()
    
        return fluid_dynamics

    def build_decoder(self):
        """构建CNN解码器模型"""
        decoder_input = tf.keras.Input(shape=self.latent_dim, name='3_next_fluid')
        # Decoder Model Full-Connected + Reshape
        decoded = tf.keras.layers.Dense(2 * 3 * 4, activation='tanh')(decoder_input)
        decoded = Reshape((2, 3, 4))(decoded)
        
        # Decoder Model 1st: 2x3 -> 4x5
        decoded = UpSampling2D((2, 2))(decoded)
        decoded = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 1)))(decoded)  # 4x6 -> 4x5
        decoded = Conv2D(filters=4, kernel_size=(3, 3), activation='tanh', padding='same')(decoded)
        
        # Decoder Model 2nd: 4x5 -> 8x10
        decoded = UpSampling2D((2, 2))(decoded)
        decoded = Conv2D(filters=8, kernel_size=(3, 3), activation='tanh', padding='same')(decoded)
        
        # Decoder Model 3rd: 8x10 -> 16x20
        decoded = UpSampling2D((2, 2))(decoded)
        decoded = Conv2D(filters=8, kernel_size=(3, 3), activation='tanh', padding='same')(decoded)
        
        # Decoder Model 4th: 16x20 -> 32x40
        decoded = UpSampling2D((2, 2))(decoded)
        decoded = Conv2D(filters=16, kernel_size=(3, 3), activation='tanh', padding='same')(decoded)
        
        # Decoder Model 5th: 32x40 -> 64x80
        decoded = UpSampling2D((2, 2))(decoded)
        decoded = Conv2D(filters=32, kernel_size=(3, 3), activation='tanh', padding='same')(decoded)

        # Decoder Model 6th: 64x80 -> 128x160
        decoded = UpSampling2D((2, 2))(decoded)
        decoder_output = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='linear')(decoded)
        decoder = tf.keras.Model(decoder_input, decoder_output, name='decoder')
        
        # 打印模型摘要
        print("\n====== Decoder Model Summary ======")
        decoder.summary()
        
        
        return decoder

    def denormalize_channel_tensor(self, normalized_field, scaler):
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

    def finite_difference_axis(self, field, axis):
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

    def physical_gradients(self, X_grid, Y_grid, field):
        """Compute d(field)/dX and d(field)/dY on a curvilinear XY grid."""
        dfield_di = self.finite_difference_axis(field, axis=1)
        dfield_dj = self.finite_difference_axis(field, axis=2)
        dX_di = self.finite_difference_axis(X_grid, axis=1)
        dX_dj = self.finite_difference_axis(X_grid, axis=2)
        dY_di = self.finite_difference_axis(Y_grid, axis=1)
        dY_dj = self.finite_difference_axis(Y_grid, axis=2)

        jacobian = dX_di * dY_dj - dX_dj * dY_di
        jacobian = tf.where(tf.abs(jacobian) < 1.0e-12, tf.ones_like(jacobian) * 1.0e-12, jacobian)

        dfield_dX = (dfield_di * dY_dj - dfield_dj * dY_di) / jacobian
        dfield_dY = (dX_di * dfield_dj - dX_dj * dfield_di) / jacobian
        return dfield_dX, dfield_dY

    def build_struc_dynamics_legacy_pressure_only(self, X_grid, Y_grid, y_feature, v_feature,
                                                  dec_u_data, dec_v_data, dec_p_data, scalers):
        print('X_grid: ', X_grid.shape)
        print('Y_grid: ', Y_grid.shape)
        print('y_feature: ', y_feature.shape)
        print('v_feature: ', v_feature.shape)
        print('dec_u_data: ', dec_u_data.shape)
        print('dec_v_data: ', dec_v_data.shape)
        print('dec_p_data: ', dec_p_data.shape)

        """计算圆柱体受到的阻力和升力，并模拟结构动力学响应
        X_circle: cylinder-surface x coordinates (batch, 128)
        Y_circle: cylinder-surface y coordinates (batch, 128)
        y_feature: 初始位移 (360, );
        v_feature: 初始速度 (360, );
        viscous lift: computed from predicted velocity gradients
        dec_p_data: predicted pressure field (batch, 128, 160, 1)
        返回:  位移数组, 速度数组"""
        
        # 对于CNN模型，dec_p_data的维度已经改变，需要提取圆柱表面的压力
        # 假设第一个列索引(列号为0)对应圆柱表面
        surface_pressure = dec_p_data[:, :, 0, 0]  # (batch, 128)
        u_physical = self.denormalize_channel_tensor(dec_u_data, scalers[0])[:, :, :, 0]
        v_physical = self.denormalize_channel_tensor(dec_v_data, scalers[1])[:, :, :, 0]
        p_physical = self.denormalize_channel_tensor(dec_p_data, scalers[2])[:, :, :, 0]

        dUdX, dUdY = self.physical_gradients(X_grid, Y_grid, u_physical)
        dVdX, dVdY = self.physical_gradients(X_grid, Y_grid, v_physical)

        surface_pressure = p_physical[:, :, 0]
        surface_dUdY = dUdY[:, :, 0]
        surface_dVdX = dVdX[:, :, 0]
        surface_dVdY = dVdY[:, :, 0]
        X_circle = X_grid[:, :, 0]
        Y_circle = Y_grid[:, :, 0]
        
        # 1. 反归一化压力
        # 1.1 提取scalers具体参数
        data_min = scalers[2].data_min_
        data_max = scalers[2].data_max_
        feature_min = scalers[2].feature_range[0]
        feature_max = scalers[2].feature_range[1]

        # 1.2 计算缩放因子
        scale = (data_max - data_min) / (feature_max - feature_min)

        # 1.3 计算偏移量
        offset = data_min - feature_min * scale

        # 使用沿表面的压力值
        # 注意：这里需要调整，因为scale和offset的形状变化了
        # 使用形状匹配的缩放和偏移计算
        scale_surface = scale[:self.grid_height]
        offset_surface = offset[:self.grid_height]
        
        # 反归一化表面压力
        surface_pressure_denorm = surface_pressure * scale_surface + offset_surface

        # 2. 参数设置
        row = self.grid_height
        d_t = 0.05
        m_body = 2.950307
        c_body = 0.074149
        k_body = 25.803348

        num_timesteps = y_feature.shape[0]  # 360

        # 3. 初始化力数组
        F_L_total_array = tf.TensorArray(dtype=tf.float64, size=num_timesteps)

        # 4. 角度定义
        dtheta = 2 * np.pi / (row - 1)
        
        # 5. 计算单个时刻圆柱所受到的升力, 并循环360次计算
        for time_struct_fce in range(num_timesteps):
            # 预测出来的
            X = X_circle[time_struct_fce, :]
            Y = Y_circle[time_struct_fce, :]
            P = surface_pressure_denorm[time_struct_fce, :]  # 使用反归一化后的表面压力
            F_L_2 = tf.constant(0.0, dtype=tf.float64)
            
            # 坐标点平移至以圆点为圆心的圆
            x_origin = X
            y_origin = Y - (tf.reduce_max(Y) + tf.reduce_min(Y))/2
            
            # 计算圆柱直径和半径
            D = tf.reduce_max(y_origin) - tf.reduce_min(y_origin)
            rds = D / 2

            # 计算法向量
            ny = y_origin / rds  # 外法向量y分量

            F_L = tf.zeros((row, 1), dtype=tf.float64)
            F_L_1 = -(P * ny * rds * dtheta)
            F_L_total = tf.reduce_sum(F_L_1) + F_L_2

            F_L_total_array = F_L_total_array.write(time_struct_fce, F_L_total)

        # 将 TensorArray 转换成 Tensor (360, 1)
        F_L_total = tf.reshape(F_L_total_array.stack(), (num_timesteps, 1))

        v_feature = tf.reshape(v_feature, (num_timesteps, 1))
        y_feature = tf.reshape(y_feature, (num_timesteps, 1))

        # 计算阻尼力、弹簧力和净力
        second_term = v_feature * c_body  # damping force
        third_term = y_feature * k_body  # spring force
        first_term = F_L_total - second_term - third_term  # net force
        a_body = first_term / m_body

        v_body = v_feature + a_body * d_t
        y_body = y_feature + v_feature * d_t

        return y_body, v_body

    def build_struc_dynamics_from_velocity_gradients(self, X_grid, Y_grid, y_feature, v_feature,
                                                     dec_u_data, dec_v_data, dec_p_data, scalers):
        """Compute structure response using pressure lift and viscous lift from velocity gradients."""
        u_physical = self.denormalize_channel_tensor(dec_u_data, scalers[0])[:, :, :, 0]
        v_physical = self.denormalize_channel_tensor(dec_v_data, scalers[1])[:, :, :, 0]
        p_physical = self.denormalize_channel_tensor(dec_p_data, scalers[2])[:, :, :, 0]

        dUdX, dUdY = self.physical_gradients(X_grid, Y_grid, u_physical)
        dVdX, dVdY = self.physical_gradients(X_grid, Y_grid, v_physical)

        surface_pressure = p_physical[:, :, 0]
        surface_dUdY = dUdY[:, :, 0]
        surface_dVdX = dVdX[:, :, 0]
        surface_dVdY = dVdY[:, :, 0]
        X_circle = X_grid[:, :, 0]
        Y_circle = Y_grid[:, :, 0]

        row = self.grid_height
        d_t = 0.05
        m_body = 2.950307
        c_body = 0.074149
        k_body = 25.803348
        dtheta = 2 * np.pi / (row - 1)
        num_timesteps = y_feature.shape[0]

        F_L_total_array = tf.TensorArray(dtype=tf.float64, size=num_timesteps)
        for time_struct_fce in range(num_timesteps):
            X = X_circle[time_struct_fce, :]
            Y = Y_circle[time_struct_fce, :]
            P = surface_pressure[time_struct_fce, :]
            dUdY_wall = surface_dUdY[time_struct_fce, :]
            dVdX_wall = surface_dVdX[time_struct_fce, :]
            dVdY_wall = surface_dVdY[time_struct_fce, :]

            x_origin = X - (tf.reduce_max(X) + tf.reduce_min(X)) / 2
            y_origin = Y - (tf.reduce_max(Y) + tf.reduce_min(Y)) / 2
            D = tf.reduce_max(y_origin) - tf.reduce_min(y_origin)
            rds = D / 2
            nx = x_origin / rds
            ny = y_origin / rds

            F_L_pre = -P * ny * rds * dtheta
            F_L_vis = (2 * self.fluid_viscosity * dVdY_wall * ny
                       + self.fluid_viscosity * (dUdY_wall + dVdX_wall) * nx) * rds * dtheta
            F_L_total = tf.reduce_sum(F_L_pre + F_L_vis)
            F_L_total_array = F_L_total_array.write(time_struct_fce, F_L_total)

        F_L_total = tf.reshape(F_L_total_array.stack(), (num_timesteps, 1))
        v_feature = tf.reshape(v_feature, (num_timesteps, 1))
        y_feature = tf.reshape(y_feature, (num_timesteps, 1))

        second_term = v_feature * c_body
        third_term = y_feature * k_body
        first_term = F_L_total - second_term - third_term
        a_body = first_term / m_body

        v_body = v_feature + a_body * d_t
        y_body = y_feature + v_feature * d_t
        return y_body, v_body

    def save_models(self, save_dir):
        """保存完整模型"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.encoder.save(os.path.join(save_dir, 'encoder'))
        self.decoder.save(os.path.join(save_dir, 'decoder'))
        self.fluid_dynamics.save(os.path.join(save_dir, 'fluid_dynamics'))

        print(f"模型已保存到 {save_dir}")

    @tf.function
    def train_step(self, viv_uvp, eta_structure, velocity_structure,
                   scalers, time_advance, coupling_sub_iterations):
        # 输入四个参数
        alfa_recons = 1
        alfa_latent = 10
        alfa_pred = 1
        alfa_corr = 1
        
        with tf.GradientTape() as tape_enc, tf.GradientTape() as tape_dec, tf.GradientTape() as tape_fsi_dyn:

            # ****************************************************************************************#
            #                                     Reconstruction loss                                 #
            # ****************************************************************************************#
            # 对CNN模型，我们需要分别处理每个通道
            batch_size = tf.shape(viv_uvp)[0]
            
            # high dim → encoder → low dim (u v p)
            enc_inp_u = viv_uvp[:, :, :, 2:3]  # (batch, 128, 160, 1)
            enc_out_u = self.encoder(enc_inp_u)

            enc_inp_v = viv_uvp[:, :, :, 3:4]
            enc_out_v = self.encoder(enc_inp_v)

            enc_inp_p = viv_uvp[:, :, :, 4:5]
            enc_out_p = self.encoder(enc_inp_p)

            # u v p → decoder → high dim
            dec_out_u = self.decoder(enc_out_u)  # (batch, 128, 160, 1)
            dec_out_v = self.decoder(enc_out_v)  # (batch, 128, 160, 1)
            dec_out_p = self.decoder(enc_out_p)  # (batch, 128, 160, 1)

            # high dim (u v p) → concatenate
            dec_out_uvp = tf.keras.backend.concatenate([dec_out_u, dec_out_v, dec_out_p], axis=-1)  # (batch, 128, 160, 3)
            print('dec_out_uvp: ',dec_out_uvp.shape)
            # 重建损失
            recons_loss = alfa_recons * self.mse(viv_uvp[:, :, :, 2:5], dec_out_uvp)

            # ****************************************************************************************#
            #                                         Latent loss                                     #
            # ****************************************************************************************#
            # 低维特征与结构速度连接
            enc_out_us = tf.keras.backend.concatenate([enc_out_u, velocity_structure], axis=-1)  # (batch, latent_dim+1)
            enc_out_vs = tf.keras.backend.concatenate([enc_out_v, velocity_structure], axis=-1)
            enc_out_ps = tf.keras.backend.concatenate([enc_out_p, velocity_structure], axis=-1)
            enc_out_uvps = tf.stack([enc_out_us, enc_out_vs, enc_out_ps], axis=-1)  # 堆叠成3D张量 (batch, latent_dim+1, 3)

            # 重塑为时间序列数据
            fluid_dyn_inp_us = tf.keras.backend.reshape(enc_out_us, (-1, time_advance, self.latent_dim + 1))  # (360, 5, 5)
            fluid_dyn_inp_vs = tf.keras.backend.reshape(enc_out_vs, (-1, time_advance, self.latent_dim + 1))
            fluid_dyn_inp_ps = tf.keras.backend.reshape(enc_out_ps, (-1, time_advance, self.latent_dim + 1))

            # 为结构动力学准备圆柱表面数据
            x_grid = tf.keras.backend.reshape(
                viv_uvp[:, :, :, 0], (-1, time_advance, self.grid_height, self.grid_width))
            y_grid = tf.keras.backend.reshape(
                viv_uvp[:, :, :, 1], (-1, time_advance, self.grid_height, self.grid_width))
            eta_structure = tf.keras.backend.reshape(eta_structure, (-1, time_advance))  # (360, 5)

            for i in range(time_advance):
                if i == 0:
                    fluid_dynamics_out_us = fluid_dyn_inp_us[:, 0, :]
                    fluid_dynamics_out_us = tf.keras.backend.reshape(fluid_dynamics_out_us, (-1, 1, self.latent_dim + 1))

                    fluid_dynamics_out_vs = fluid_dyn_inp_vs[:, 0, :]
                    fluid_dynamics_out_vs = tf.keras.backend.reshape(fluid_dynamics_out_vs, (-1, 1, self.latent_dim + 1))

                    fluid_dynamics_out_ps = fluid_dyn_inp_ps[:, 0, :]
                    fluid_dynamics_out_ps = tf.keras.backend.reshape(fluid_dynamics_out_ps, (-1, 1, self.latent_dim + 1))

                    eta_structure_inp = eta_structure[:, 0]
                    eta_structure_inp = tf.keras.backend.reshape(eta_structure_inp, (-1, 1))
                else:
                    # Multi-iteration strong coupling prediction for one time step.
                    # Strong-coupling corrector iterations, ell = 0 ... ell_max - 1.
                    # During training, Eq. (2.14) initializes phi_{t+1}^{0}
                    # from the encoded next state and then repeatedly corrects
                    # fluid modal coordinates and structural velocity.
                    base_phi_u = fluid_dynamics_out_us[:, -1, 0:self.latent_dim]
                    base_phi_v = fluid_dynamics_out_vs[:, -1, 0:self.latent_dim]
                    base_phi_p = fluid_dynamics_out_ps[:, -1, 0:self.latent_dim]
                    base_velocity = fluid_dynamics_out_us[:, -1, self.latent_dim:self.latent_dim + 1]

                    corrected_phi_u = fluid_dyn_inp_us[:, i, 0:self.latent_dim]
                    corrected_phi_v = fluid_dyn_inp_vs[:, i, 0:self.latent_dim]
                    corrected_phi_p = fluid_dyn_inp_ps[:, i, 0:self.latent_dim]
                    corrected_velocity = fluid_dyn_inp_us[:, i, self.latent_dim:self.latent_dim + 1]
                    corrected_y_body = eta_structure[:, i:i + 1]

                    for _ in range(coupling_sub_iterations):
                        coupled_velocity = 0.5 * (base_velocity + corrected_velocity)
                        fluid_dyn_inp_u = tf.keras.backend.concatenate(
                            [0.5 * (base_phi_u + corrected_phi_u), coupled_velocity], axis=-1)
                        fluid_dyn_inp_v = tf.keras.backend.concatenate(
                            [0.5 * (base_phi_v + corrected_phi_v), coupled_velocity], axis=-1)
                        fluid_dyn_inp_p = tf.keras.backend.concatenate(
                            [0.5 * (base_phi_p + corrected_phi_p), coupled_velocity], axis=-1)

                        fluid_dyn_delta_u = self.fluid_dynamics(fluid_dyn_inp_u)
                        fluid_dyn_delta_v = self.fluid_dynamics(fluid_dyn_inp_v)
                        fluid_dyn_delta_p = self.fluid_dynamics(fluid_dyn_inp_p)

                        corrected_phi_u = base_phi_u + fluid_dyn_delta_u
                        corrected_phi_v = base_phi_v + fluid_dyn_delta_v
                        corrected_phi_p = base_phi_p + fluid_dyn_delta_p

                        struc_inp_u = self.decoder(corrected_phi_u)
                        struc_inp_v = self.decoder(corrected_phi_v)
                        struc_inp_p = self.decoder(corrected_phi_p)
                        corrected_y_body, corrected_velocity = self.build_struc_dynamics_from_velocity_gradients(
                            x_grid[:, i, :, :], y_grid[:, i, :, :],
                            eta_structure_inp[:, -1],
                            base_velocity[:, 0],
                            struc_inp_u,
                            struc_inp_v,
                            struc_inp_p,
                            scalers)

                        corrected_y_body = tf.keras.backend.reshape(corrected_y_body, (-1, self.struc_output_dim))
                        corrected_velocity = tf.keras.backend.reshape(corrected_velocity, (-1, self.struc_output_dim))

                    fluid_dyn_out_u = tf.keras.backend.reshape(corrected_phi_u, (-1, 1, self.latent_dim))
                    fluid_dyn_out_v = tf.keras.backend.reshape(corrected_phi_v, (-1, 1, self.latent_dim))
                    fluid_dyn_out_p = tf.keras.backend.reshape(corrected_phi_p, (-1, 1, self.latent_dim))
                    struc_dyn_out_v_body = tf.keras.backend.reshape(corrected_velocity, (-1, 1, self.struc_output_dim))

                    # 组合流体和结构动力学输出
                    fsi_dyn_out_us = tf.keras.backend.concatenate([fluid_dyn_out_u, struc_dyn_out_v_body], axis=-1)
                    fsi_dyn_out_vs = tf.keras.backend.concatenate([fluid_dyn_out_v, struc_dyn_out_v_body], axis=-1)
                    fsi_dyn_out_ps = tf.keras.backend.concatenate([fluid_dyn_out_p, struc_dyn_out_v_body], axis=-1)

                    # 在时间维度上连接
                    fluid_dynamics_out_us = tf.keras.backend.concatenate([fluid_dynamics_out_us, fsi_dyn_out_us], axis=1)
                    fluid_dynamics_out_vs = tf.keras.backend.concatenate([fluid_dynamics_out_vs, fsi_dyn_out_vs], axis=1)
                    fluid_dynamics_out_ps = tf.keras.backend.concatenate([fluid_dynamics_out_ps, fsi_dyn_out_ps], axis=1)
                    
                    eta_structure_inp = tf.keras.backend.concatenate([eta_structure_inp, corrected_y_body], axis=-1)

            # 重塑流固耦合结构
            fsi_dynamics_output_us = tf.keras.backend.reshape(fluid_dynamics_out_us, (-1, self.latent_dim + 1, 1))
            fsi_dynamics_output_vs = tf.keras.backend.reshape(fluid_dynamics_out_vs, (-1, self.latent_dim + 1, 1))
            fsi_dynamics_output_ps = tf.keras.backend.reshape(fluid_dynamics_out_ps, (-1, self.latent_dim + 1, 1))
            
            fsi_dynamics_output_uvps = tf.keras.backend.concatenate([fsi_dynamics_output_us, fsi_dynamics_output_vs, fsi_dynamics_output_ps], axis=-1)
            
            latent_loss = alfa_latent * self.mse(fsi_dynamics_output_uvps, enc_out_uvps)

            # ****************************************************************************************#
            #                                       Prediction loss                                   #
            # ****************************************************************************************#
            # 动力学潜在特征
            dec_prd_inp = fsi_dynamics_output_uvps  # (batch, latent_dim+1, 3)
            dec_prd_inp_u = dec_prd_inp[:, 0:self.latent_dim, 0]
            dec_prd_inp_v = dec_prd_inp[:, 0:self.latent_dim, 1]
            dec_prd_inp_p = dec_prd_inp[:, 0:self.latent_dim, 2]

            # 解码器
            dec_prd_out_u = self.decoder(dec_prd_inp_u)  # (batch, 128, 160, 1)
            dec_prd_out_v = self.decoder(dec_prd_inp_v)
            dec_prd_out_p = self.decoder(dec_prd_inp_p)

            # 预测损失
            predict_u_loss = alfa_pred * self.mse(viv_uvp[:, :, :, 2:3], dec_prd_out_u)
            predict_v_loss = alfa_pred * self.mse(viv_uvp[:, :, :, 3:4], dec_prd_out_v)
            predict_p_loss = alfa_pred * self.mse(viv_uvp[:, :, :, 4:5], dec_prd_out_p)
            predict_loss = predict_u_loss + predict_v_loss + predict_p_loss

            # ****************************************************************************************#
            #                                      Correlation loss                                   #
            # ****************************************************************************************#
            # 模态坐标之间的相关性损失 - 确保模态之间的独立性
            u_modal = tf.keras.backend.reshape(enc_out_u, (-1, self.latent_dim))
            u_cor = tfp.stats.correlation(u_modal, sample_axis=0, event_axis=-1) - np.eye(self.latent_dim) #revise
            correlation_loss_u = self.mse(u_cor, tf.zeros_like(u_cor))

            v_modal = tf.keras.backend.reshape(enc_out_v, (-1, self.latent_dim))
            v_cor = tfp.stats.correlation(v_modal, sample_axis=0, event_axis=-1) - np.eye(self.latent_dim) #revise
            correlation_loss_v = self.mse(v_cor, tf.zeros_like(v_cor)) 

            p_modal = tf.keras.backend.reshape(enc_out_p, (-1, self.latent_dim))
            p_cor = tfp.stats.correlation(p_modal, sample_axis=0, event_axis=-1) - np.eye(self.latent_dim) #revise
            correlation_loss_p = self.mse(p_cor, tf.zeros_like(p_cor))

            correlation_loss = alfa_corr * (correlation_loss_u + correlation_loss_v + correlation_loss_p)

            # ****************************************************************************************#
            #                                     Three Model losses                                  #
            # ****************************************************************************************#
            enc_loss = recons_loss + predict_loss + latent_loss + correlation_loss
            dec_loss = recons_loss + predict_loss
            dyn_fsi_loss = predict_loss + latent_loss

        # 计算梯度并应用
        gradients_of_enc = tape_enc.gradient(enc_loss, self.encoder.trainable_variables)
        gradients_of_dec = tape_dec.gradient(dec_loss, self.decoder.trainable_variables)
        gradients_of_fsi_dyn = tape_fsi_dyn.gradient(dyn_fsi_loss, self.fluid_dynamics.trainable_variables)

        self.encoder_optimizer.apply_gradients(zip(gradients_of_enc, self.encoder.trainable_variables))
        self.decoder_optimizer.apply_gradients(zip(gradients_of_dec, self.decoder.trainable_variables))
        self.fluid_dynamics_optimizer.apply_gradients(zip(gradients_of_fsi_dyn, self.fluid_dynamics.trainable_variables))

        return recons_loss, latent_loss, predict_u_loss, predict_v_loss, predict_p_loss, correlation_loss

    # ****************************************************************************************#
    #                                     训练模型                                             #
    # ****************************************************************************************#
    def train(self, XYUVP_train, eta_structure_train, velocity_structure_train, 
              scalers, num_epochs, batch_size,
              coupling_sub_iterations=5):
    # ****************************************************************************************#
    #                          Train the model using batch processing                         #
    # ****************************************************************************************#
    
        current_lr_record = []
        reco_loss_record = []
        late_loss_record = []
        pred_loss_u_record = []
        pred_loss_v_record = []
        pred_loss_p_record = []
        corr_loss_record = []

        N = XYUVP_train.shape[0]
        
        for epoch in range(num_epochs):
            current_lr = self.learning_rate_for_epoch(epoch, num_epochs)
            self.set_optimizer_learning_rate(current_lr)
            for it in range(0, N, batch_size):  # 900,1800
                
                start_time = time.time()
                
                idx = np.arange(it, it + batch_size)
                XYUVP_train_batch                  = XYUVP_train[idx, :]
                eta_structure_train_batch          = eta_structure_train[idx, :]
                velocity_structure_train_batch     = velocity_structure_train[idx, :]
                
                reco_loss_value, late_loss_value, pred_u_loss_value, pred_v_loss_value, pred_p_loss_value, corr_loss_value = self.train_step(
                    XYUVP_train_batch,
                    eta_structure_train_batch,
                    velocity_structure_train_batch,
                    scalers,
                    5,
                    coupling_sub_iterations
                )
                if it % (batch_size) == 0:
                    elapsed = time.time() - start_time
                    print('Epoch: %d, It: %d, Time: %.2f, lr: %.3e, reco_loss: %.3e, late_loss: %.3e, '
                          'pred_loss_u: %.3e, pred_loss_v: %.3e, pred_loss_p: %.3e, corr_loss: %.3e'
                          % (epoch, it/batch_size, elapsed, current_lr, reco_loss_value, late_loss_value, pred_u_loss_value, pred_v_loss_value,
                             pred_p_loss_value, corr_loss_value))

                current_lr_record.append(current_lr)
                reco_loss_record.append(reco_loss_value)
                late_loss_record.append(late_loss_value)
                pred_loss_u_record.append(pred_u_loss_value)
                pred_loss_v_record.append(pred_v_loss_value)
                pred_loss_p_record.append(pred_p_loss_value)
                corr_loss_record.append(corr_loss_value)

#         if num_epochs % 1000 == 0:
#             plt.figure(figsize=(12, 10))
#             plt.subplot(321)
#             plt.plot(reco_loss_record)
#             plt.xlabel('epoch')
#             plt.ylabel('recons_loss')
#             plt.yscale('log')

#             plt.subplot(322)
#             plt.plot(late_loss_record)
#             plt.xlabel('epoch')
#             plt.ylabel('latent_loss')
#             plt.yscale('log')

#             plt.subplot(323)
#             plt.plot(pred_loss_u_record)
#             plt.xlabel('epoch')
#             plt.ylabel('pred_loss_u')
#             plt.yscale('log')

#             plt.subplot(324)
#             plt.plot(pred_loss_v_record)
#             plt.xlabel('epoch')
#             plt.ylabel('pred_loss_v')
#             plt.yscale('log')

#             plt.subplot(325)
#             plt.plot(pred_loss_p_record)
#             plt.xlabel('epoch')
#             plt.ylabel('pred_loss_p')
#             plt.yscale('log')

#             plt.subplot(326)
#             plt.plot(corr_loss_record)
#             plt.xlabel('epoch')
#             plt.ylabel('corr_loss')
#             plt.yscale('log')

#             plt.tight_layout()
#             plt.savefig(file_path3 + 'loss.pdf')
#             plt.clf()

        self.save_models(file_path3)

        scipy.io.savemat(file_path3 + 'loss.mat', {'reconstruction': reco_loss_record,
                                                   'latent': late_loss_record,
                                                   'prediction_u': pred_loss_u_record,
                                                   'prediction_v': pred_loss_v_record,
                                                   'prediction_p': pred_loss_p_record,
                                                   'correlation': corr_loss_record})

        return current_lr_record, reco_loss_record, late_loss_record, pred_loss_u_record, pred_loss_v_record, pred_loss_p_record, corr_loss_record


######################## Run Model ####################################################################################################
if __name__ == '__main__':
    model = nnmVIV(
        input_shape=(128, 160, 1),  # full-field CNN input shape
        encode_dim=64,
        latent_dim=4,
        fluid_dynamic_dim=128,
        struc_output_dim=1
    )
    
    train_snapshots = 1400
    validation_snapshots = 400
    test_snapshots = 500
    train_end = train_snapshots
    validation_end = train_snapshots + validation_snapshots
    test_end = train_snapshots + validation_snapshots + test_snapshots
    available_snapshots = min(
        XY.shape[0],
        UVP.shape[0],
        velocity_structure.shape[0],
        eta_structure.shape[0],
        lift_structure.shape[0],
        drag_structure.shape[0],
        drag_viscous_structure.shape[0]
    )
    if available_snapshots < test_end:
        raise ValueError(
            "The paper split requires 2300 aligned snapshots "
            "(1400 train + 400 validation + 500 test), "
            f"but the shortest loaded array contains {available_snapshots} snapshots."
        )
    
    # 结构数据划分：训练、测试
    drag_structure_train = drag_structure[:train_end, :]
    drag_structure_valid = drag_structure[train_end:validation_end, :]
    drag_structure_test = drag_structure[validation_end:test_end, :]

    drag_viscous_structure_train = drag_viscous_structure[:train_end, :]
    drag_viscous_structure_valid = drag_viscous_structure[train_end:validation_end, :]
    drag_viscous_structure_test = drag_viscous_structure[validation_end:test_end, :]

    eta_structure_train = eta_structure[:train_end, :]
    eta_structure_valid = eta_structure[train_end:validation_end, :]
    eta_structure_test = eta_structure[validation_end:test_end, :]

    lift_structure_train = lift_structure[:train_end, :]
    lift_structure_valid = lift_structure[train_end:validation_end, :]
    lift_structure_test = lift_structure[validation_end:test_end, :]

    print(f"drag_viscous_structure_train: {drag_viscous_structure_train.shape}")
    print(f"drag_viscous_structure_valid: {drag_viscous_structure_valid.shape}")
    print(f"drag_viscous_structure_test: {drag_viscous_structure_test.shape}")
    print(f"eta_structure_train: {eta_structure_train.shape}")
    print(f"eta_structure_valid: {eta_structure_valid.shape}")
    print(f"eta_structure_test: {eta_structure_test.shape}")
    print(f"lift_structure_train: {lift_structure_train.shape}")
    print(f"lift_structure_valid: {lift_structure_valid.shape}")
    print(f"lift_structure_test: {lift_structure_test.shape}")

    # 数据2: 进行归一化
    (minn, maxx,
     XYUVP_train, XYUVP_valid, XYUVP_test,
     velocity_structure_train, velocity_structure_valid, velocity_structure_test,
     scalers) = model.normalize_data(
        XY, UVP, velocity_structure,
        train_snapshots=train_snapshots,
        validation_snapshots=validation_snapshots,
        test_snapshots=test_snapshots)

    print(f"maxx: {maxx.shape}")
    print(f"XYUVP_train: {XYUVP_train.shape}")
    print(f"XYUVP_valid: {XYUVP_valid.shape}")
    print(f"XYUVP_test: {XYUVP_test.shape}")
    print(f"velocity_structure_train: {velocity_structure_train.shape}")
    print(f"velocity_structure_valid: {velocity_structure_valid.shape}")
    print(f"velocity_structure_test: {velocity_structure_test.shape}")

    # 训练模型
    current_lr_record, reco_loss_record, late_loss_record, pred_loss_u_record, pred_loss_v_record, pred_loss_p_record, corr_loss_record = model.train(
        XYUVP_train,
        eta_structure_train,
        velocity_structure_train,
        scalers,
        num_epochs=5000,
        batch_size=350,
        coupling_sub_iterations=5)

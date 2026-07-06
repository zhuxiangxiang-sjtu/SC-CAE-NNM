"""Training process for the SC-CAE-NNM framework."""

from pathlib import Path
import time

import numpy as np
import scipy.io
import tensorflow as tf
import tensorflow_probability as tfp

from encoder_decoder_dynamic import build_decoder, build_encoder, build_fluid_dynamics
from viscous_force import structure_dynamics_from_velocity_gradients


class SCCaeNnmTrainer:
    """Container for the three trainable modules and the training loop."""

    def __init__(self, input_shape, latent_dim, fluid_dynamic_dim, struc_output_dim,
                 show_summary=True):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.fluid_dynamic_dim = fluid_dynamic_dim
        self.struc_output_dim = struc_output_dim
        self.grid_height = input_shape[0]
        self.grid_width = input_shape[1]
        self.fluid_viscosity = tf.constant(0.01, dtype=tf.float64)

        self.initial_learning_rate = 1.0e-3
        self.final_learning_rate = 1.0e-5
        self.fixed_lr_epochs = 2000

        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate)
        self.decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate)
        self.fluid_dynamics_optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate)

        self.encoder = build_encoder(input_shape, latent_dim, show_summary=show_summary)
        self.decoder = build_decoder(latent_dim, show_summary=show_summary)
        self.fluid_dynamics = build_fluid_dynamics(latent_dim, fluid_dynamic_dim, show_summary=show_summary)
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

    def save_models(self, save_dir):
        """Save the three trained TensorFlow models."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save(save_dir / "encoder")
        self.decoder.save(save_dir / "decoder")
        self.fluid_dynamics.save(save_dir / "fluid_dynamics")
        print(f"Models saved to {save_dir}")

    def train_step(self, viv_uvp, eta_structure, velocity_structure,
                   scalers, time_advance, coupling_sub_iterations):
        alfa_recons = 1
        alfa_latent = 10
        alfa_pred = 1
        alfa_corr = 1
        identity = tf.constant(np.eye(self.latent_dim), dtype=tf.float64)

        with tf.GradientTape() as tape_enc, tf.GradientTape() as tape_dec, tf.GradientTape() as tape_fsi_dyn:
            enc_inp_u = viv_uvp[:, :, :, 2:3]
            enc_out_u = self.encoder(enc_inp_u)

            enc_inp_v = viv_uvp[:, :, :, 3:4]
            enc_out_v = self.encoder(enc_inp_v)

            enc_inp_p = viv_uvp[:, :, :, 4:5]
            enc_out_p = self.encoder(enc_inp_p)

            dec_out_u = self.decoder(enc_out_u)
            dec_out_v = self.decoder(enc_out_v)
            dec_out_p = self.decoder(enc_out_p)
            dec_out_uvp = tf.keras.backend.concatenate([dec_out_u, dec_out_v, dec_out_p], axis=-1)
            recons_loss = alfa_recons * self.mse(viv_uvp[:, :, :, 2:5], dec_out_uvp)

            enc_out_us = tf.keras.backend.concatenate([enc_out_u, velocity_structure], axis=-1)
            enc_out_vs = tf.keras.backend.concatenate([enc_out_v, velocity_structure], axis=-1)
            enc_out_ps = tf.keras.backend.concatenate([enc_out_p, velocity_structure], axis=-1)
            enc_out_uvps = tf.stack([enc_out_us, enc_out_vs, enc_out_ps], axis=-1)

            fluid_dyn_inp_us = tf.keras.backend.reshape(
                enc_out_us, (-1, time_advance, self.latent_dim + 1))
            fluid_dyn_inp_vs = tf.keras.backend.reshape(
                enc_out_vs, (-1, time_advance, self.latent_dim + 1))
            fluid_dyn_inp_ps = tf.keras.backend.reshape(
                enc_out_ps, (-1, time_advance, self.latent_dim + 1))

            x_grid = tf.keras.backend.reshape(
                viv_uvp[:, :, :, 0], (-1, time_advance, self.grid_height, self.grid_width))
            y_grid = tf.keras.backend.reshape(
                viv_uvp[:, :, :, 1], (-1, time_advance, self.grid_height, self.grid_width))
            eta_structure = tf.keras.backend.reshape(eta_structure, (-1, time_advance))

            for i in range(time_advance):
                if i == 0:
                    fluid_dynamics_out_us = fluid_dyn_inp_us[:, 0, :]
                    fluid_dynamics_out_us = tf.keras.backend.reshape(
                        fluid_dynamics_out_us, (-1, 1, self.latent_dim + 1))

                    fluid_dynamics_out_vs = fluid_dyn_inp_vs[:, 0, :]
                    fluid_dynamics_out_vs = tf.keras.backend.reshape(
                        fluid_dynamics_out_vs, (-1, 1, self.latent_dim + 1))

                    fluid_dynamics_out_ps = fluid_dyn_inp_ps[:, 0, :]
                    fluid_dynamics_out_ps = tf.keras.backend.reshape(
                        fluid_dynamics_out_ps, (-1, 1, self.latent_dim + 1))

                    eta_structure_inp = eta_structure[:, 0]
                    eta_structure_inp = tf.keras.backend.reshape(eta_structure_inp, (-1, 1))
                else:
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
                        corrected_y_body, corrected_velocity = structure_dynamics_from_velocity_gradients(
                            x_grid[:, i, :, :],
                            y_grid[:, i, :, :],
                            eta_structure_inp[:, -1],
                            base_velocity[:, 0],
                            struc_inp_u,
                            struc_inp_v,
                            struc_inp_p,
                            scalers,
                            fluid_viscosity=self.fluid_viscosity,
                            grid_height=self.grid_height,
                        )

                        corrected_y_body = tf.keras.backend.reshape(
                            corrected_y_body, (-1, self.struc_output_dim))
                        corrected_velocity = tf.keras.backend.reshape(
                            corrected_velocity, (-1, self.struc_output_dim))

                    fluid_dyn_out_u = tf.keras.backend.reshape(corrected_phi_u, (-1, 1, self.latent_dim))
                    fluid_dyn_out_v = tf.keras.backend.reshape(corrected_phi_v, (-1, 1, self.latent_dim))
                    fluid_dyn_out_p = tf.keras.backend.reshape(corrected_phi_p, (-1, 1, self.latent_dim))
                    struc_dyn_out_v_body = tf.keras.backend.reshape(
                        corrected_velocity, (-1, 1, self.struc_output_dim))

                    fsi_dyn_out_us = tf.keras.backend.concatenate(
                        [fluid_dyn_out_u, struc_dyn_out_v_body], axis=-1)
                    fsi_dyn_out_vs = tf.keras.backend.concatenate(
                        [fluid_dyn_out_v, struc_dyn_out_v_body], axis=-1)
                    fsi_dyn_out_ps = tf.keras.backend.concatenate(
                        [fluid_dyn_out_p, struc_dyn_out_v_body], axis=-1)

                    fluid_dynamics_out_us = tf.keras.backend.concatenate(
                        [fluid_dynamics_out_us, fsi_dyn_out_us], axis=1)
                    fluid_dynamics_out_vs = tf.keras.backend.concatenate(
                        [fluid_dynamics_out_vs, fsi_dyn_out_vs], axis=1)
                    fluid_dynamics_out_ps = tf.keras.backend.concatenate(
                        [fluid_dynamics_out_ps, fsi_dyn_out_ps], axis=1)
                    eta_structure_inp = tf.keras.backend.concatenate(
                        [eta_structure_inp, corrected_y_body], axis=-1)

            fsi_dynamics_output_us = tf.keras.backend.reshape(
                fluid_dynamics_out_us, (-1, self.latent_dim + 1, 1))
            fsi_dynamics_output_vs = tf.keras.backend.reshape(
                fluid_dynamics_out_vs, (-1, self.latent_dim + 1, 1))
            fsi_dynamics_output_ps = tf.keras.backend.reshape(
                fluid_dynamics_out_ps, (-1, self.latent_dim + 1, 1))

            fsi_dynamics_output_uvps = tf.keras.backend.concatenate(
                [fsi_dynamics_output_us, fsi_dynamics_output_vs, fsi_dynamics_output_ps], axis=-1)
            latent_loss = alfa_latent * self.mse(fsi_dynamics_output_uvps, enc_out_uvps)

            dec_prd_inp_u = fsi_dynamics_output_uvps[:, 0:self.latent_dim, 0]
            dec_prd_inp_v = fsi_dynamics_output_uvps[:, 0:self.latent_dim, 1]
            dec_prd_inp_p = fsi_dynamics_output_uvps[:, 0:self.latent_dim, 2]

            dec_prd_out_u = self.decoder(dec_prd_inp_u)
            dec_prd_out_v = self.decoder(dec_prd_inp_v)
            dec_prd_out_p = self.decoder(dec_prd_inp_p)

            predict_u_loss = alfa_pred * self.mse(viv_uvp[:, :, :, 2:3], dec_prd_out_u)
            predict_v_loss = alfa_pred * self.mse(viv_uvp[:, :, :, 3:4], dec_prd_out_v)
            predict_p_loss = alfa_pred * self.mse(viv_uvp[:, :, :, 4:5], dec_prd_out_p)
            predict_loss = predict_u_loss + predict_v_loss + predict_p_loss

            u_modal = tf.keras.backend.reshape(enc_out_u, (-1, self.latent_dim))
            v_modal = tf.keras.backend.reshape(enc_out_v, (-1, self.latent_dim))
            p_modal = tf.keras.backend.reshape(enc_out_p, (-1, self.latent_dim))
            correlation_loss_u = self.mse(
                tfp.stats.correlation(u_modal, sample_axis=0, event_axis=-1) - identity,
                tf.zeros_like(identity),
            )
            correlation_loss_v = self.mse(
                tfp.stats.correlation(v_modal, sample_axis=0, event_axis=-1) - identity,
                tf.zeros_like(identity),
            )
            correlation_loss_p = self.mse(
                tfp.stats.correlation(p_modal, sample_axis=0, event_axis=-1) - identity,
                tf.zeros_like(identity),
            )
            correlation_loss = alfa_corr * (
                correlation_loss_u + correlation_loss_v + correlation_loss_p)

            enc_loss = recons_loss + predict_loss + latent_loss + correlation_loss
            dec_loss = recons_loss + predict_loss
            dyn_fsi_loss = predict_loss + latent_loss

        gradients_of_enc = tape_enc.gradient(enc_loss, self.encoder.trainable_variables)
        gradients_of_dec = tape_dec.gradient(dec_loss, self.decoder.trainable_variables)
        gradients_of_fsi_dyn = tape_fsi_dyn.gradient(
            dyn_fsi_loss, self.fluid_dynamics.trainable_variables)

        self.encoder_optimizer.apply_gradients(
            zip(gradients_of_enc, self.encoder.trainable_variables))
        self.decoder_optimizer.apply_gradients(
            zip(gradients_of_dec, self.decoder.trainable_variables))
        self.fluid_dynamics_optimizer.apply_gradients(
            zip(gradients_of_fsi_dyn, self.fluid_dynamics.trainable_variables))

        return recons_loss, latent_loss, predict_u_loss, predict_v_loss, predict_p_loss, correlation_loss

    def train(self, xyuvp_train, eta_structure_train, velocity_structure_train,
              scalers, save_dir, num_epochs, batch_size,
              time_advance=5, coupling_sub_iterations=5):
        """Train the coupled model using batch processing."""
        if batch_size % time_advance != 0:
            raise ValueError("batch_size must be divisible by time_advance.")
        if xyuvp_train.shape[0] % batch_size != 0:
            raise ValueError("Number of training snapshots must be divisible by batch_size.")

        loss_records = {
            "learning_rate": [],
            "reconstruction": [],
            "latent": [],
            "prediction_u": [],
            "prediction_v": [],
            "prediction_p": [],
            "correlation": [],
        }

        n_samples = xyuvp_train.shape[0]
        for epoch in range(num_epochs):
            current_lr = self.learning_rate_for_epoch(epoch, num_epochs)
            self.set_optimizer_learning_rate(current_lr)

            for it in range(0, n_samples, batch_size):
                start_time = time.time()
                idx = np.arange(it, it + batch_size)

                losses = self.train_step(
                    xyuvp_train[idx, :],
                    eta_structure_train[idx, :],
                    velocity_structure_train[idx, :],
                    scalers,
                    time_advance,
                    coupling_sub_iterations,
                )
                elapsed = time.time() - start_time
                loss_values = [float(loss.numpy()) for loss in losses]

                print(
                    "Epoch: %d, It: %d, Time: %.2f, lr: %.3e, "
                    "reco_loss: %.3e, late_loss: %.3e, pred_loss_u: %.3e, "
                    "pred_loss_v: %.3e, pred_loss_p: %.3e, corr_loss: %.3e"
                    % (
                        epoch,
                        it / batch_size,
                        elapsed,
                        current_lr,
                        loss_values[0],
                        loss_values[1],
                        loss_values[2],
                        loss_values[3],
                        loss_values[4],
                        loss_values[5],
                    )
                )

                loss_records["learning_rate"].append(current_lr)
                loss_records["reconstruction"].append(loss_values[0])
                loss_records["latent"].append(loss_values[1])
                loss_records["prediction_u"].append(loss_values[2])
                loss_records["prediction_v"].append(loss_values[3])
                loss_records["prediction_p"].append(loss_values[4])
                loss_records["correlation"].append(loss_values[5])

        self.save_models(save_dir)
        save_dir = Path(save_dir)
        scipy.io.savemat(save_dir / "loss.mat", loss_records)
        return loss_records

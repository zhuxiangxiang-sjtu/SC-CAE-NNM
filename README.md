# SC-CAE-NNM

Sample code for the **Strong-Coupling Convolutional Autoencoder Nonlinear
Normal Mode (SC-CAE-NNM)** framework for fluid-structure interaction prediction
of vortex-induced vibration.

The code is organized for academic reproducibility and method inspection. It
implements the training pipeline described in the paper, including CAE-based
flow-field reduction, latent-space dynamics prediction, pressure-lift structure
response update, and strong fluid-structure coupling iterations.

## Reference

Xiangxiang Zhu, Shanwu Li, Yong Cao, Shengqi Zhang, Shubin Fu, Zhiping Mao,
Yongchao Yang, and Shiyi Chen, "A strongly coupled fluid-structure interaction
network for predictive modelling of vortex-induced vibrations," *Journal of
Fluid Mechanics*, 2026, vol. 0, A1. doi:10.1017/jfm.2026.11766.

## Method Overview

The framework contains three trainable modules:

- a convolutional autoencoder (CAE) encoder for mapping full-order flow fields
  to latent modal coordinates;
- a CAE decoder for reconstructing streamwise velocity, transverse velocity,
  and pressure fields from the latent coordinates;
- a latent-space fluid dynamics prediction module for advancing the modal
  state with the structural velocity as a coupling input.

During training, the predicted latent flow state is decoded back to the physical
field. The structural response is then updated from the pressure-induced lift
on the cylinder surface. This release keeps the coupling mechanism explicit and
focuses on the pressure contribution to the lift force.

For each time-advancement step, the fluid and structural states are corrected
through strong-coupling sub-iterations. The default implementation uses
`l = 5` sub-iterations.

## Repository Structure

```text
SC-CAE-NNM/
+-- main.py
+-- data_utils.py
+-- encoder_decoder_dynamic.py
+-- compute_force.py
+-- training.py
+-- README.md
+-- .gitignore
```

| File | Description |
| --- | --- |
| `main.py` | Main script for data loading, data splitting, model construction, and training |
| `data_utils.py` | MATLAB data loading, normalization, and train/validation/test splitting |
| `encoder_decoder_dynamic.py` | CAE encoder, CAE decoder, and latent fluid-dynamics network definitions |
| `compute_force.py` | Pressure-induced lift calculation and structure update |
| `training.py` | Strong-coupling training loop, loss calculation, optimizer update, and model saving |

The recommended entry point is:

```text
main.py
```

## Requirements

- Python 3.x
- TensorFlow
- TensorFlow Probability
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- h5py

Install the Python dependencies with:

```bash
pip install tensorflow tensorflow-probability numpy scipy matplotlib scikit-learn h5py
```

The implementation was developed for TensorFlow-GPU. The default script enables
GPU memory growth. GPU device selection can be adjusted in `main.py`.

## Dataset Preparation

The DNS data files are not included in this repository because of file-size and
data-management constraints.

Place the required MATLAB files under a local `Data/` directory:

```text
Data/
+-- VIV_displacement_velocity_lift_drag_Re100_hrk2.mat
+-- VIV_1dof_Re100_XY_UVP.mat
```

Expected variables are:

| Variable | Description | Expected shape |
| --- | --- | --- |
| `XY` | flow-grid coordinates | `(n_snapshots, 128, 160, 2)` |
| `UVP` | streamwise velocity, transverse velocity, and pressure | `(n_snapshots, 128, 160, 3)` |
| `eta_structure` | structural displacement | `(n_snapshots, 1)` |
| `velocity_structure` | structural velocity | `(n_snapshots, 1)` |
| `lift_structure` | total lift force | `(n_snapshots, 1)` |
| `drag_structure` | total drag force | `(n_snapshots, 1)` |

The code uses the following snapshot split:

| Dataset | Index range | Number of snapshots |
| --- | --- | --- |
| Training | `0:1400` | 1400 |
| Validation | `1400:1800` | 400 |
| Test | `1800:2300` | 500 |

The loader checks that all arrays contain at least 2300 aligned snapshots. The
normalization scalers are fitted on the training snapshots only.

## Quick Start

1. Install the dependencies.
2. Create a local `Data/` directory.
3. Put the two required `.mat` files into `Data/`.
4. Run the training script:

```bash
python main.py
```

Trained models and loss histories are saved to:

```text
models/trial1/
+-- encoder/
+-- decoder/
+-- fluid_dynamics/
+-- loss.mat
```

The `Data/` and `models/` directories are ignored by Git.

## CAE Architecture

The CAE input size is:

```text
128 x 160 x 1
```

The encoder maps each flow variable to a 4-dimensional latent coordinate. The
decoder first maps the latent coordinate to a `2 x 3 x 4` feature block and then
upsamples it back to `128 x 160 x 1`.

The decoder path is:

```text
(2, 3, 4)
-> (4, 5, 4)
-> (8, 10, 4)
-> (16, 20, 8)
-> (32, 40, 8)
-> (64, 80, 16)
-> (128, 160, 32)
-> (128, 160, 1)
```

CAE convolutional layers use `tanh` activation. The final CAE output layer uses
linear activation. The latent fluid dynamics prediction module uses ReLU
activation in the hidden layers.

## Training Settings

Default settings in `main.py` and `training.py` are:

| Parameter | Value |
| --- | --- |
| Optimizer | Adam |
| Initial learning rate | `1e-3` |
| Final learning rate | `1e-5` |
| Total epochs | `5000` |
| Fixed fine-tuning stage | last `2000` epochs |
| Batch size | `350` |
| Time window | `5` |
| Latent dimension | `4` |
| Strong-coupling sub-iterations | `5` |

The learning rate decays exponentially from `1e-3` to `1e-5` before the final
fine-tuning stage. During the last 2000 epochs, the learning rate is fixed at
`1e-5`.

The training loss includes:

- CAE reconstruction loss;
- latent-space prediction loss;
- decoded flow-field prediction losses for `u`, `v`, and `p`;
- modal correlation loss.

## Pressure Lift and Structure Update

The structural dynamics update is implemented in `compute_force.py`.

For each decoded flow state, the module:

1. converts the normalized pressure field back to physical values;
2. extracts the cylinder-surface pressure values from the first grid column;
3. integrates the pressure-induced lift around the cylinder surface;
4. updates structural acceleration, velocity, and displacement.

This pressure-only release is intended to make the strong-coupling calculation
clear and stable for the open-source sample code.

## Notes

- The code is provided for academic research and educational clarity.
- The authors provide no guarantees for production use.
- The sample implementation prioritizes transparent correspondence with the
  paper over maximum computational speed.
- This release uses pressure-induced lift only in the structural update.

## Dependency List

```text
tensorflow
tensorflow-probability
numpy
scipy
matplotlib
scikit-learn
h5py
```

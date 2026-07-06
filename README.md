# SC-CAE-NNM

This repository provides the sample implementation of the strong-coupling
convolutional autoencoder nonlinear normal mode (SC-CAE-NNM) framework for
vortex-induced vibration prediction.

The code is organized as a reproducible reference implementation corresponding
to the model settings reported in the paper. The original simulation data and
trained weights are not included because of file size and data-management
constraints.

## Repository Structure

```text
SC-CAE-NNM/
+-- OpenSource_SC_CAE_NNM/
|   +-- SampleCodes_SC-CAE-NNM.py
+-- requirements.txt
+-- .gitignore
+-- README.md
```

## Code

The main script is:

```text
OpenSource_SC_CAE_NNM/SampleCodes_SC-CAE-NNM.py
```

It contains:

- the CNN encoder and decoder for flow-field compression and reconstruction;
- the fluid dynamics prediction module in latent space;
- the structure dynamics update coupled with pressure and viscous lift;
- full-field velocity-gradient evaluation for viscous force calculation;
- strong-coupling sub-iterations with `l = 5`;
- the training loop and learning-rate schedule.

## Data Format

The code expects the following MATLAB `.mat` files under a local `Data/`
directory:

```text
Data/
+-- VIV_displacement_velocity_lift_drag_Re100_hrk2.mat
+-- VIV_1dof_Re100_XY_UVP.mat
```

Expected arrays:

| Variable | Description | Expected shape |
| --- | --- | --- |
| `XY` | grid coordinates | `(n_snapshots, 128, 160, 2)` |
| `UVP` | streamwise velocity, transverse velocity, pressure | `(n_snapshots, 128, 160, 3)` |
| `eta_structure` | structural displacement | `(n_snapshots, 1)` |
| `velocity_structure` | structural velocity | `(n_snapshots, 1)` |
| `lift_structure` | total lift force | `(n_snapshots, 1)` |
| `drag_structure` | total drag force | `(n_snapshots, 1)` |
| `drag_viscous_structure` | viscous drag force | `(n_snapshots, 1)` |

The flow-field input size used by the CAE is:

```text
128 x 160 x 1
```

## Dataset Split

The snapshots are split according to the paper:

| Set | Snapshot index range | Number of snapshots |
| --- | --- | --- |
| Training | `0:1400` | 1400 |
| Validation | `1400:1800` | 400 |
| Test | `1800:2300` | 500 |

The script checks that all loaded arrays contain at least 2300 aligned
snapshots before training. The normalization scalers are fitted on the training
snapshots only to avoid validation/test leakage.

## CAE Architecture

The CAE follows the encoder-decoder structure reported in the paper:

- input: `(128, 160, 1)`;
- latent value: `(4, 1, 1)`;
- fully connected decoder seed: `(2, 3, 4)`;
- decoder upsampling path: `(4, 5, 4) -> (8, 10, 4) -> (16, 20, 8) -> (32, 40, 8) -> (64, 80, 16) -> (128, 160, 32)`;
- output: `(128, 160, 1)`.

The CAE convolutional layers use `tanh` activation, and the final decoder
output layer uses a linear activation. The fluid dynamics prediction module uses
ReLU activation.

## Training Settings

The three trainable modules, namely the encoder, decoder, and fluid dynamics
prediction module, are trained simultaneously.

Default settings in the sample code:

| Parameter | Value |
| --- | --- |
| Optimizer | Adam |
| Initial learning rate | `1e-3` |
| Final fine-tuning learning rate | `1e-5` |
| Total epochs | `5000` |
| Fixed fine-tuning stage | last `2000` epochs |
| Batch size | `350` |
| Strong-coupling sub-iterations | `l = 5` |

The learning rate decays exponentially from `1e-3` to `1e-5` before the final
fine-tuning stage. During the last 2000 epochs, the learning rate is fixed at
`1e-5`.

## Installation

Create an environment with Python and install the dependencies:

```bash
pip install -r requirements.txt
```

The implementation was developed for TensorFlow-GPU. GPU configuration may need
to be adjusted according to the available hardware.

## Usage

Place the required `.mat` files in the `Data/` directory, then run:

```bash
python OpenSource_SC_CAE_NNM/SampleCodes_SC-CAE-NNM.py
```

Model outputs and loss histories are saved under:

```text
models/trial1/
```

## Notes

- The `Data/` and `models/` directories are ignored by Git.
- The sample code keeps all major physical and training settings explicit so
  that readers can verify the implementation against the paper.
- The viscous lift contribution is computed from full-field velocity gradients
  rather than from a preloaded viscous-lift array.

## Citation

If this code is useful for your research, please cite the corresponding paper.

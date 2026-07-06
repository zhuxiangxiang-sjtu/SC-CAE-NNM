"""Main entry point for training the SC-CAE-NNM sample implementation."""

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import tensorflow as tf

from data_utils import (
    load_flow_data,
    load_structure_data,
    normalize_flow_data,
    split_structure_data,
    validate_aligned_snapshot_count,
)
from training import SCCaeNnmTrainer


def reset_random_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def configure_tensorflow():
    tf.keras.backend.set_floatx("float64")
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "Data"
    save_dir = project_root / "models" / "trial1"

    reset_random_seeds(135)
    configure_tensorflow()

    structure_data = load_structure_data(
        data_dir / "VIV_displacement_velocity_lift_drag_Re100_hrk2.mat"
    )
    flow_data = load_flow_data(data_dir / "VIV_1dof_Re100_XY_UVP.mat")

    print("XY shape:", flow_data.xy.shape)
    print("UVP shape:", flow_data.uvp.shape)
    print("velocity_structure shape:", structure_data.velocity_structure.shape)

    train_snapshots = 1400
    validation_snapshots = 400
    test_snapshots = 500

    validate_aligned_snapshot_count(
        flow_data,
        structure_data,
        train_snapshots=train_snapshots,
        validation_snapshots=validation_snapshots,
        test_snapshots=test_snapshots,
    )
    structure_split = split_structure_data(
        structure_data,
        train_snapshots=train_snapshots,
        validation_snapshots=validation_snapshots,
        test_snapshots=test_snapshots,
    )

    flow_split = normalize_flow_data(
        flow_data.xy,
        flow_data.uvp,
        structure_data.velocity_structure,
        train_snapshots=train_snapshots,
        validation_snapshots=validation_snapshots,
        test_snapshots=test_snapshots,
    )

    print("XYUVP_train:", flow_split.xyuvp_train.shape)
    print("XYUVP_valid:", flow_split.xyuvp_valid.shape)
    print("XYUVP_test:", flow_split.xyuvp_test.shape)
    print("eta_train:", structure_split.eta_train.shape)
    print("eta_valid:", structure_split.eta_valid.shape)
    print("eta_test:", structure_split.eta_test.shape)
    print("velocity_train:", flow_split.velocity_train.shape)
    print("velocity_valid:", flow_split.velocity_valid.shape)
    print("velocity_test:", flow_split.velocity_test.shape)

    trainer = SCCaeNnmTrainer(
        input_shape=(128, 160, 1),
        latent_dim=4,
        fluid_dynamic_dim=128,
        struc_output_dim=1,
        show_summary=True,
    )

    trainer.train(
        flow_split.xyuvp_train,
        structure_split.eta_train,
        flow_split.velocity_train,
        flow_split.scalers,
        save_dir=save_dir,
        num_epochs=5000,
        batch_size=350,
        time_advance=5,
        coupling_sub_iterations=5,
    )


if __name__ == "__main__":
    main()

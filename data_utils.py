"""Data loading, normalization, and snapshot splitting utilities."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler


@dataclass
class StructureData:
    drag_structure: np.ndarray
    eta_structure: np.ndarray
    lift_structure: np.ndarray
    velocity_structure: np.ndarray


@dataclass
class FlowData:
    xy: np.ndarray
    uvp: np.ndarray


@dataclass
class StructureSplit:
    drag_train: np.ndarray
    drag_valid: np.ndarray
    drag_test: np.ndarray
    eta_train: np.ndarray
    eta_valid: np.ndarray
    eta_test: np.ndarray
    lift_train: np.ndarray
    lift_valid: np.ndarray
    lift_test: np.ndarray


@dataclass
class NormalizedFlowSplit:
    minn: np.ndarray
    maxx: np.ndarray
    xyuvp_train: np.ndarray
    xyuvp_valid: np.ndarray
    xyuvp_test: np.ndarray
    velocity_train: np.ndarray
    velocity_valid: np.ndarray
    velocity_test: np.ndarray
    scalers: list


def load_structure_data(path):
    """Load displacement, velocity, lift, and drag data from the structure file."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        return StructureData(
            drag_structure=np.array(f["drag_structure"]).T,
            eta_structure=np.array(f["eta_structure"]).T,
            lift_structure=np.array(f["lift_structure"]).T,
            velocity_structure=np.array(f["velocity_structure"]).T,
        )


def load_flow_data(path):
    """Load full-field coordinates and flow variables from the flow file."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        return FlowData(
            xy=np.array(f["XY"]).T,
            uvp=np.array(f["UVP"]).T,
        )


def split_bounds(train_snapshots=1400, validation_snapshots=400, test_snapshots=500):
    train_end = train_snapshots
    validation_end = train_snapshots + validation_snapshots
    test_end = train_snapshots + validation_snapshots + test_snapshots
    return train_end, validation_end, test_end


def validate_aligned_snapshot_count(flow_data, structure_data,
                                    train_snapshots=1400,
                                    validation_snapshots=400,
                                    test_snapshots=500):
    """Ensure all arrays have enough aligned snapshots for the paper split."""
    _, _, test_end = split_bounds(train_snapshots, validation_snapshots, test_snapshots)
    available_snapshots = min(
        flow_data.xy.shape[0],
        flow_data.uvp.shape[0],
        structure_data.velocity_structure.shape[0],
        structure_data.eta_structure.shape[0],
        structure_data.lift_structure.shape[0],
        structure_data.drag_structure.shape[0],
    )
    if available_snapshots < test_end:
        raise ValueError(
            "The paper split requires 2300 aligned snapshots "
            "(1400 train + 400 validation + 500 test), "
            f"but the shortest loaded array contains {available_snapshots} snapshots."
        )


def split_structure_data(structure_data, train_snapshots=1400,
                         validation_snapshots=400, test_snapshots=500):
    """Split structure quantities into train, validation, and test intervals."""
    train_end, validation_end, test_end = split_bounds(
        train_snapshots, validation_snapshots, test_snapshots
    )
    return StructureSplit(
        drag_train=structure_data.drag_structure[:train_end, :],
        drag_valid=structure_data.drag_structure[train_end:validation_end, :],
        drag_test=structure_data.drag_structure[validation_end:test_end, :],
        eta_train=structure_data.eta_structure[:train_end, :],
        eta_valid=structure_data.eta_structure[train_end:validation_end, :],
        eta_test=structure_data.eta_structure[validation_end:test_end, :],
        lift_train=structure_data.lift_structure[:train_end, :],
        lift_valid=structure_data.lift_structure[train_end:validation_end, :],
        lift_test=structure_data.lift_structure[validation_end:test_end, :],
    )


def normalize_flow_data(xy, uvp, velocity_structure,
                        train_snapshots=1400,
                        validation_snapshots=400,
                        test_snapshots=500):
    """Normalize UVP and split XYUVP snapshots as reported in the paper."""
    first_dim = uvp.shape[0]
    num_rows = uvp.shape[1]
    num_cols = uvp.shape[2]
    required_snapshots = train_snapshots + validation_snapshots + test_snapshots
    if first_dim < required_snapshots:
        raise ValueError(
            "The paper split requires at least "
            f"{required_snapshots} snapshots "
            f"({train_snapshots} train + {validation_snapshots} validation + "
            f"{test_snapshots} test), but UVP contains {first_dim} snapshots."
        )

    train_end, validation_end, test_end = split_bounds(
        train_snapshots, validation_snapshots, test_snapshots
    )

    reshaped_u = uvp[:, :, :, 0].reshape(first_dim, -1)
    reshaped_v = uvp[:, :, :, 1].reshape(first_dim, -1)
    reshaped_p = uvp[:, :, :, 2].reshape(first_dim, -1)

    scaler_u = MinMaxScaler(feature_range=(-1, 1))
    scaler_v = MinMaxScaler(feature_range=(-1, 1))
    scaler_p = MinMaxScaler(feature_range=(-1, 1))

    scaler_u.fit(reshaped_u[:train_end])
    u_norm = scaler_u.transform(reshaped_u).reshape(first_dim, num_rows, num_cols, 1)
    scaler_v.fit(reshaped_v[:train_end])
    v_norm = scaler_v.transform(reshaped_v).reshape(first_dim, num_rows, num_cols, 1)
    scaler_p.fit(reshaped_p[:train_end])
    p_norm = scaler_p.transform(reshaped_p).reshape(first_dim, num_rows, num_cols, 1)

    uvp_norm = np.concatenate((u_norm, v_norm, p_norm), axis=-1)
    xyuvp_norm = np.concatenate((xy, uvp_norm), axis=-1)

    minn_u = np.reshape(np.min(reshaped_u[validation_end:test_end], axis=0), (-1, 1))
    maxx_u = np.reshape(np.max(reshaped_u[validation_end:test_end], axis=0), (-1, 1))
    minn_v = np.reshape(np.min(reshaped_v[validation_end:test_end], axis=0), (-1, 1))
    maxx_v = np.reshape(np.max(reshaped_v[validation_end:test_end], axis=0), (-1, 1))
    minn_p = np.reshape(np.min(reshaped_p[validation_end:test_end], axis=0), (-1, 1))
    maxx_p = np.reshape(np.max(reshaped_p[validation_end:test_end], axis=0), (-1, 1))

    return NormalizedFlowSplit(
        minn=np.concatenate((minn_u, minn_v, minn_p), axis=-1),
        maxx=np.concatenate((maxx_u, maxx_v, maxx_p), axis=-1),
        xyuvp_train=xyuvp_norm[:train_end, :, :, :],
        xyuvp_valid=xyuvp_norm[train_end:validation_end, :, :, :],
        xyuvp_test=xyuvp_norm[validation_end:test_end, :, :, :],
        velocity_train=velocity_structure[:train_end, :],
        velocity_valid=velocity_structure[train_end:validation_end, :],
        velocity_test=velocity_structure[validation_end:test_end, :],
        scalers=[scaler_u, scaler_v, scaler_p],
    )


def denormalize_uvp(uvp_normalized, scalers):
    """Convert normalized UVP arrays back to physical values."""
    shape = uvp_normalized.shape
    uvp_original = np.zeros_like(uvp_normalized)
    for channel in range(3):
        reshaped = uvp_normalized[:, :, :, channel].reshape(shape[0], -1)
        unscaled = scalers[channel].inverse_transform(reshaped)
        uvp_original[:, :, :, channel] = unscaled.reshape(shape[0], shape[1], shape[2])
    return uvp_original

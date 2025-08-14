# ftt_config.py
from __future__ import annotations
import torch
from scripts.pipelines.models.ftt_model import InputSpec, TargetSpec

# ==============================
# Repro & verbosity
# ==============================
SEED = 54
VERBOSE = True

# ==============================
# General settings
# ==============================
LOCAL_FLAG = False
SUBSET_OF_SHOTS = 1
NUM_WORKERS = 0  # 64
OUTPUT_SUB_FOLDER = 'ftt_output/'
SAVE_RESULTS = True
RUN_EVALUATION = True
RUN_TRAINING = True

# ==============================
# SEGNAL LIST
# ==============================

SOURCE_SIGNAL_LIST = [
    ('magnetics', 'flux_loop_flux'),
    ('magnetics', 'b_field_pol_probe_ccbv_field'),
    ('magnetics', 'b_field_pol_probe_obr_field'),
    ('magnetics', 'b_field_pol_probe_obv_field'),
    ('pf_active', 'solenoid_current'),
    ('pf_active', 'coil_voltage'),
    ('pf_active', 'coil_current'),
    ('pulse_schedule', 'i_plasma'),
    ('summary', 'power_nbi'),
    ('equilibrium', 'elongation'),
    ('equilibrium', 'elongation_axis'),
    ('equilibrium', 'triangularity_upper'),
    ('equilibrium', 'triangularity_lower'),
    ('equilibrium', 'minor_radius'),
    ('equilibrium', 'magnetic_axis_r'),
    ('equilibrium', 'magnetic_axis_z')
]

# ==============================
# INPUT specs (no shapes; keys define x_keys)
# Encoders are assigned later using DEFAULT_INPUT_ENCODER_BY_DIM
# ==============================
INPUT_SPECS = {
    # Time series & profiles you’re using as inputs
    "magnetics-flux_loop_flux": InputSpec(
        name="magnetics-flux_loop_flux",
        # encoder_name="flatten_bspline_1d",
        # encoder_kwargs={"degree": 3, "num_basis": 5}
    ),
    # "magnetics-b_field_pol_probe_ccbv_field": InputSpec(
    #     name="magnetics-b_field_pol_probe_ccbv_field"
    # ),
    # "magnetics-b_field_pol_probe_obr_field": InputSpec(
    #     name="magnetics-b_field_pol_probe_obr_field"
    # ),
    # "magnetics-b_field_pol_probe_obv_field": InputSpec(
    #     name="magnetics-b_field_pol_probe_obv_field"
    # ),
    # "pf_active-solenoid_current": InputSpec(
    #     name="pf_active-solenoid_current"
    # ),
    # "pf_active-coil_voltage": InputSpec(
    #     name="pf_active-coil_voltage"
    # ),
    # "pf_active-coil_current": InputSpec(
    #     name="pf_active-coil_current"
    # ),
    # "pulse_schedule-i_plasma": InputSpec(
    #     name="pulse_schedule-i_plasma"
    # ),
    # "summary-power_nbi": InputSpec(
    #     name="summary-power_nbi"),
}

# ==============================
# TARGET specs (no shapes; keys define y_keys)
# Encoders are assigned later using DEFAULT_TARGET_ENCODER
# ==============================
TARGET_SPECS = {
    "equilibrium-elongation": TargetSpec(
        name="equilibrium-elongation",
        # encoder_name="flatten_bspline_1d",
        # encoder_kwargs={"degree": 3, "num_basis": 5}
        head_hidden=128,
        loss="mse",
        loss_weight=1.0
    ),
    "equilibrium-elongation_axis": TargetSpec(
        name="equilibrium-elongation_axis",
        head_hidden=128,
        loss="mse",
        loss_weight=1.0
    ),
    # "equilibrium-triangularity_upper": TargetSpec(
    #     name="equilibrium-triangularity_upper",
    #     head_hidden=128,
    #     loss="mse",
    #     loss_weight=1.0
    # ),
    # "equilibrium-triangularity_lower": TargetSpec(
    #     name="equilibrium-triangularity_lower",
    #     head_hidden=128,
    #     loss="mse",
    #     loss_weight=1.0
    # ),
    # "equilibrium-minor_radius": TargetSpec(
    #     name="equilibrium-minor_radius",
    #     head_hidden=128,
    #     loss="mse",
    #     loss_weight=1.0
    # ),
    # "equilibrium-magnetic_axis_r": TargetSpec(
    #     name="equilibrium-magnetic_axis_r",
    #     head_hidden=128,
    #     loss="mse",
    #     loss_weight=1.0)
    # ,
    # "equilibrium-magnetic_axis_z": TargetSpec(
    #     name="equilibrium-magnetic_axis_z",
    #     head_hidden=128,
    #     loss="mse",
    #     loss_weight=1.0
    # ),
}

# Optional: exclude some targets from training
INACTIVE_TARGETS: list[str] = []

# ==============================
# Modality-based encoder default
# Applied - if not defined manually - AFTER shapes are bound from the first shot
# ==============================

DEFAULT_INPUT_ENCODERS_BY_MOD = {
    # "timeseries": dict(encoder_name="flatten_bspline_1d", encoder_kwargs={"degree": 4, "num_basis": 5}),
    "profile": dict(encoder_name="fpca_3d", encoder_kwargs={"num_components": 5, "pca_dim": "space"}),
    # "profile": dict(encoder_name="per_channel_bspline_1d", encoder_kwargs={"degree": 4, "num_basis": 5}),
    # "profile": dict(encoder_name=None),
    # "image": dict(encoder_name="dct_2d", encoder_kwargs={"keep_h": 8, "keep_w": 8}),
    # "video": dict(encoder_name="fpca_3d", encoder_kwargs={"num_components": 5, "pca_dim": "space"}),
    "video": dict(encoder_name="dct_3d", encoder_kwargs={"keep_h": 5, "keep_w": 5, "keep_t": 5}),
    "scalar": dict(encoder_name=None, encoder_kwargs=None),
    "vector": dict(encoder_name=None, encoder_kwargs=None),
}

DEFAULT_TARGET_ENCODERS_BY_MOD = {
    "timeseries": dict(encoder_name="flatten_bspline_1d", encoder_kwargs={"degree": 4, "num_basis": 5}),
    "profile": dict(encoder_name="per_channel_bspline_1d", encoder_kwargs={"degree": 4, "num_basis": 5}),
    "image": dict(encoder_name="dct_2d", encoder_kwargs={"keep_h": 8, "keep_w": 8}),
    "video": dict(encoder_name="dct_3d", encoder_kwargs={"keep_h": 5, "keep_w": 5, "keep_t": 5}),
    "scalar": dict(encoder_name=None, encoder_kwargs=None),
    "vector": dict(encoder_name=None, encoder_kwargs=None),
}

# ==============================
# Transforms parameters
# x_keys and y_keys are derived from the specs above
# ==============================
WINDOW_SEGMENTER_PARAMS = {
    "x_keys": list(INPUT_SPECS.keys()),  # auto from INPUT_SPECS
    "y_keys": list(TARGET_SPECS.keys()),  # auto from TARGET_SPECS
    "x_window_sec": 0.01,
    "y_window_sec": 0.1,
    "dt_sec": 0.025,
    "stride_sec": None,
    "stride_unitary": True,
    "min_samples_per_window": 1,
    "verbose": False,
}

REF_FREQ = 0.005  # for SamplingToReferenceTimeTransform, which we comment out

# ==============================
# Training params (placeholder – not used in init-only step)
# ==============================
BATCH_SIZE = 100
EPOCHS = 2
LR_TRUNK = 1e-3
LR_HEADS = 1e-3
USE_ADAMW = False
LOSS_SPACE = "native"  # 'pred' or 'native'
EARLY_STOP_PATIENCE = 5


# ==============================
# Device & dtype
# ==============================
MODEL_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

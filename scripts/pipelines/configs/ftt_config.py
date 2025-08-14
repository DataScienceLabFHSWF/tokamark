# ftt_config.py
from __future__ import annotations
import torch
from scripts.pipelines.models.ftt_model import InputSpec, TargetSpec

# ==============================
# Repro & verbosity
# ==============================
SEED = 54
VERBOSE = True
OUTPUT_SUB_FOLDER = 'ftt_output/'
LOCAL_FLAG = False

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
        name="magnetics-flux_loop_flux"),
    "magnetics-b_field_pol_probe_ccbv_field": InputSpec(
        name="magnetics-b_field_pol_probe_ccbv_field"),
    "magnetics-b_field_pol_probe_obr_field": InputSpec(
        name="magnetics-b_field_pol_probe_obr_field"),
    "magnetics-b_field_pol_probe_obv_field": InputSpec(
        name="magnetics-b_field_pol_probe_obv_field"),
    "pf_active-solenoid_current": InputSpec(
        name="pf_active-solenoid_current"),
    "pf_active-coil_voltage": InputSpec(
        name="pf_active-coil_voltage"),
    "pf_active-coil_current": InputSpec(
        name="pf_active-coil_current"),
    "pulse_schedule-i_plasma": InputSpec(
        name="pulse_schedule-i_plasma"),
    "summary-power_nbi": InputSpec(
        name="summary-power_nbi"),
}

# ==============================
# TARGET specs (no shapes; keys define y_keys)
# Encoders are assigned later using DEFAULT_TARGET_ENCODER
# ==============================
TARGET_SPECS = {
    "equilibrium-elongation": TargetSpec(
        name="equilibrium-elongation",
        head_hidden=128,
        loss="mse",
        loss_weight=1.0),
    "equilibrium-elongation_axis": TargetSpec(
        name="equilibrium-elongation_axis",
        head_hidden=128,
        loss="mse",
        loss_weight=1.0),
    "equilibrium-triangularity_upper": TargetSpec(
        name="equilibrium-triangularity_upper",
        head_hidden=128,
        loss="mse",
        loss_weight=1.0),
    "equilibrium-triangularity_lower": TargetSpec(
        name="equilibrium-triangularity_lower",
        head_hidden=128,
        loss="mse",
        loss_weight=1.0),
    "equilibrium-minor_radius": TargetSpec(
        name="equilibrium-minor_radius",
        head_hidden=128,
        loss="mse",
        loss_weight=1.0),
    "equilibrium-magnetic_axis_r": TargetSpec(
        name="equilibrium-magnetic_axis_r",
        head_hidden=128,
        loss="mse",
        loss_weight=1.0),
    "equilibrium-magnetic_axis_z": TargetSpec(
        name="equilibrium-magnetic_axis_z",
        head_hidden=128,
        loss="mse",
        loss_weight=1.0),
}

# Optional: exclude some targets from training
INACTIVE_TARGETS: list[str] = []

# ==============================
# Dimension-based encoder rules
# Applied AFTER shapes are bound from the first batch
# - We assume "dimension" here is the modality index inferred from shape analysis.
# ==============================

# Inputs:
#   dim 0 (time series)  -> bspline(4,5)
#   dim 1 (profile)      -> fpca_3d(num_components=3, pca_dim="space")
DEFAULT_INPUT_ENCODER_BY_DIM = {
    0: dict(encoder_name="flatten_bspline_1d", encoder_kwargs={"degree": 4, "num_basis": 5}),
    1: dict(encoder_name="fpca_3d",            encoder_kwargs={"num_components": 3, "pca_dim": "space"}),
}

# Targets:
#   encoder: per_channel_bspline_1d(4,5)  (applied regardless; override per-target if needed)
DEFAULT_TARGET_ENCODER = dict(
    encoder_name="per_channel_bspline_1d",
    encoder_kwargs={"degree": 4, "num_basis": 5},
)

# If you ever need different raw y_key vs target name:
Y_KEY_TO_TARGET = None  # or e.g. {"equilibrium-elongation": "elongation"}

# ==============================
# Transforms parameters
# x_keys and y_keys are derived from the specs above
# ==============================
WINDOW_SEGMENTER_PARAMS = {
    "x_window_sec": 0,
    "y_window_sec": 0,
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
SUBSET_OF_SHOTS = 4  # <- This can be None for the entire dataset, or a small integer.
NUM_WORKERS = 2  # 64
EPOCHS = 100
LR_TRUNK = 1e-3
LR_HEADS = 1e-3
USE_ADAMW = False
LOSS_SPACE = "native"  # 'pred' or 'native'

# ==============================
# Device & dtype
# ==============================
MODEL_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

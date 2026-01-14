import os
from typing import List
import pickle 

from MAST_benchmark.tools.path import METADATA_DIR
from MAST_benchmark.tools.transforms.compose_transform import (
    ComposeTransforms
)
from MAST_benchmark.tools.transforms.stdscale_transform import (
    StdScalingTransform,
)
from MAST_benchmark.tools.transforms.reshape_lcfs_transform import (
    ReshapeLcfsTransform,
)
from MAST_benchmark.tools.transforms.fill_profile_with_zeros_imputer_transform import (
    FillProfileWithZerosTransform,
)
from MAST_benchmark.tools.transforms.downsample_transform import (
    DownsampleTransform,
)

# ----------------------------------------------------------------------------------------------------------------------
def build_common_signal_transform_map(
    source_signal_list: List[tuple],
    use_std_scaling: bool = True,   # <--- new flag
):
    """Builds the signal transform map for each variable."""

    def maybe_std(var):
        """Return StdScalingTransform if enabled, else empty list."""
        if use_std_scaling:
            with open(os.path.join(METADATA_DIR, "dict_metadata.pkl"), "rb") as f:
                dict_metadata = pickle.load(f)
            return [StdScalingTransform(dict_metadata[var]['mean'], dict_metadata[var]['std'])]
        return []

    # Define base signal_transform_map
    signal_transform_map = {
        var: ComposeTransforms(
            maybe_std(var)
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }

    # Specific case of profiles with NaNs in full channel
    for var in [
        "magnetics-flux_loop_flux",
        "magnetics-b_field_pol_probe_ccbv_field",
        "magnetics-b_field_pol_probe_obr_field",
        "magnetics-b_field_pol_probe_obv_field",
        "magnetics-b_field_tor_probe_saddle_voltage",
        "thomson_scattering-t_e", 
        "thomson_scattering-n_e",
    ]:
        signal_transform_map[var] = ComposeTransforms(
            maybe_std(var) + [
                FillProfileWithZerosTransform(),
            ]
        )

    # Specific case of reformating LCFS
    for var in ["equilibrium-lcfs_r", "equilibrium-lcfs_z"]:
        signal_transform_map[var] = ComposeTransforms(
            [
                ReshapeLcfsTransform(),
            ] + maybe_std(var)
        )

    # Specific case of x-soft rays
    for var in [
        "soft_x_rays-horizontal_cam_upper",
        "soft_x_rays-horizontal_cam_lower",
    ]:
        signal_transform_map[var] = ComposeTransforms(
            [
                DownsampleTransform(factor=50),
            ]
            + maybe_std(var)
        )

    return signal_transform_map


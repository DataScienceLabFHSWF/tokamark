"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import os
import yaml

from MAST_benchmark.tools.path import METADATA_DIR
from MAST_benchmark.tools.transforms.compose_transform import ComposeTransforms
from MAST_benchmark.tools.transforms.stdscale_transform import StdScalingTransform
from MAST_benchmark.tools.transforms.reshape_lcfs_transform import ReshapeLcfsTransform
from MAST_benchmark.tools.transforms.fill_profile_with_zeros_imputer_transform import FillProfileWithZerosTransform
from MAST_benchmark.tools.transforms.stft_transform import STFTTransform


# ----------------------------------------------------------------------------------------------------------------------
def build_common_signal_transform_map(
    source_signal_list: list[tuple],
    use_std_scaling: bool = True
) -> dict[str, ComposeTransforms]:
    """
    Build the signal transform map for each variable.

    Parameters
    ----------
    source_signal_list : list[tuple]
        List of source-signal tuples.
    use_std_scaling: bool
        If True, use STD scaling.
        Optional. Default: True.

    Returns
    -------
    dict[str, ComposeTransforms]
        Signal transform map for each variable.

    """

    # ..................................................................................................................
    def maybe_std(
            var: str
    ) -> list:  # TODO: Find better typehint
        """
        Return [StdScalingTransform] if enabled, else empty list.  # FIXME: StdScalingTransform returns None. So?

        Parameters
        ----------
        var : str
            Input variable.

        Returns
        -------
        list  # TODO: Find better typehint
            [StdScalingTransform] if enabled, else empty list.  # FIXME: StdScalingTransform returns None. So?

        """

        if use_std_scaling:
            with open(os.path.join(METADATA_DIR, "dict_stats_metadata.yaml"), "r") as f:
                dict_stats_metadata = yaml.safe_load(f)
            return [
                StdScalingTransform(mean=dict_stats_metadata[var]["mean"], std=dict_stats_metadata[var]["std"])
            ]  # TODO: Check this, as StdScalingTransform returns None.
        return []

    # ..................................................................................................................

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
            transforms=maybe_std(var) + [
                FillProfileWithZerosTransform(),
            ]
        )

    # Specific case of reformating LCFS
    for var in ["equilibrium-lcfs_r", "equilibrium-lcfs_z"]:
        signal_transform_map[var] = ComposeTransforms(
            transforms=maybe_std(var) + [
                ReshapeLcfsTransform(),
            ]
        )

    # Specific case of profiles with NaNs in full channel
    for var in [
        "magnetics-b_field_tor_probe_cc_field",
        "magnetics-b_field_pol_probe_omv_voltage"
    ]:
        signal_transform_map[var] = ComposeTransforms(
            transforms=maybe_std(var) + [
                STFTTransform(support_n=512),
            ]
        )

    return signal_transform_map

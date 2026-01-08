from scripts.MAST_tools.MAST_dataset import MastDataset

from scripts.pipeline_tools.build_common_signal_transform_map import (
    build_common_signal_transform_map,
)

# ----------------------------------------------------------------------------------------------------------------------
def initialize_MAST_dataset(
    config,
    shots_list,
    use_std_scaling = True,
    return_incomplete_shots=True
):
    
    # ..................................................................................................................
    # Get local flag
    local_flag = config["local"]

    # ..................................................................................................................
    # Get unique source-signal
    source_signal_list = (
        (config["sources_and_signals"].get("input_name") or [])
        + (config["sources_and_signals"].get("actuator_name") or [])
        + (config["sources_and_signals"].get("output_name") or [])
    )
    source_signal_list = [
        s for i, s in enumerate(source_signal_list) if s not in source_signal_list[:i]
    ]  # Unicity  

    # ..................................................................................................................
    # Create common transform map      
    signal_transform_map = build_common_signal_transform_map(
        source_signal_list, use_std_scaling
    )

    MAST_dataset = MastDataset(
        local=local_flag,
        shots_list=shots_list,
        source_signal_list=source_signal_list,
        signal_level_transform_map=signal_transform_map,
        shot_level_transform=None,
        return_incomplete_shots=return_incomplete_shots,
    )

    return MAST_dataset
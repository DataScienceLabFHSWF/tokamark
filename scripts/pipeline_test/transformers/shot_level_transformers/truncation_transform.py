
class TruncationTransform:
    """
    A shot-level transform that truncates all time series signals in a shot
    to the shortest common length across all signals.

    This ensures all signals are aligned and have the same number of time steps.
    Non-signal metadata (e.g., 'shot_id') is ignored during truncation and preserved.

    Returns
    -------
    new_shot : dict
        A new shot dictionary with each signal truncated to the common minimum length.
        Non-signal fields (like 'shot_id') are preserved.
    """

    def __call__(self, shot):
        # Filter only valid signal entries: dicts that contain 'time' and 'values'
        valid_data = {
            var: data for var, data in shot.items()
            if isinstance(data, dict) and "time" in data and "values" in data
        }

        # Compute the minimum available time length across all valid signals
        min_common_time = min(len(data["time"]) for data in valid_data.values())

        # Construct new truncated shot
        new_shot = {}
        for var, data in valid_data.items():
            new_shot[var] = {
                "time": data["time"][:min_common_time],
                "values": data["values"][..., :min_common_time]
            }

        # Preserve shot-level metadata (e.g., shot_id)
        if "shot_id" in shot:
            new_shot["shot_id"] = shot["shot_id"]

        return new_shot

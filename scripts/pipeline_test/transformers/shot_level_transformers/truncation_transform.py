
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

        # Compute the minimum available time length across all valid signals
        min_common_time = min(len(data["time"]) for data in shot.values())

        # Construct new truncated shot
        new_shot = {}
        for var, data in shot.items():
            new_shot[var] = {
                "time": data["time"][:min_common_time],
                "values": data["values"][..., :min_common_time]
            }

        # Preserve shot-level metadata (e.g., shot_id)
        if "shot_id" in shot:
            new_shot["shot_id"] = shot["shot_id"]

        return new_shot

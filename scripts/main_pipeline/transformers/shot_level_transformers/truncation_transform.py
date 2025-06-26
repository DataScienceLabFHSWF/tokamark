
# class TruncationTransform:
#     """
#     A shot-level transform that truncates all time series signals in a shot
#     to the shortest common length across all signals.

#     This ensures all signals are aligned and have the same number of time steps.
#     Non-signal metadata (e.g., 'shot_id') is ignored during truncation and preserved.

#     Returns
#     -------
#     new_shot : dict
#         A new shot dictionary with each signal truncated to the common minimum length.
#         Non-signal fields (like 'shot_id') are preserved.
#     """

#     def __call__(self, shot):

#         # Compute the minimum available time length across all valid signals
#         min_common_time = min(len(data["time"]) for data in shot.values())

#         # Construct new truncated shot
#         new_shot = {}
#         for var, data in shot.items():
#             new_shot[var] = {
#                 "time": data["time"][:min_common_time],
#                 "values": data["values"][..., :min_common_time]
#             }

#         # Preserve shot-level metadata (e.g., shot_id)
#         if "shot_id" in shot:
#             new_shot["shot_id"] = shot["shot_id"]

#         return new_shot



import numpy as np

class TruncationTransform:
    """
    Shot-level transform that truncates all signals in a shot to the minimum end time
    across all available time-series. This ensures consistent signal duration when
    signals have different time extents or sampling frequencies.

    Usage
    -----
    transform = TruncationTransform()
    truncated_shot = transform(shot)

    Input
    -----
    shot : dict
        Dictionary of signals:
            {
                'signal_name': {
                    'time': np.ndarray,     # shape (T,)
                    'values': np.ndarray    # shape (C, T)
                },
                ...,
                'shot_id': str or int      # optional metadata
            }

    Output
    ------
    truncated_shot : dict
        Same structure as input, but all signals truncated to share the same end time.

    Notes
    -----
    - Only signal entries (dicts with 'time' and 'values') are truncated.
    - Signals with no valid data before the truncation point will raise an error.
    """

    def __call__(self, shot):
        # Step 1: Determine min end time
        min_end_time = min(
            np.max(np.asarray(data["time"]))
            for key, data in shot.items()
            # if isinstance(data, dict) and "time" in data and "values" in data
        )

        # Step 2: Truncate signals to that end time
        new_shot = {}
        for key, data in shot.items():
            # if not (isinstance(data, dict) and "time" in data and "values" in data):
            #     continue

            time = np.asarray(data["time"])
            values = np.asarray(data["values"])

            mask = time <= min_end_time
            idx = np.where(mask)[0]

            if idx.size == 0:
                raise ValueError(f"Signal '{key}' has no valid samples before {min_end_time:.5f} s")

            new_shot[key] = {
                "time": time[idx],
                "values": values[..., idx]
            }

        # Preserve metadata
        if "shot_id" in shot:
            new_shot["shot_id"] = shot["shot_id"]

        return new_shot
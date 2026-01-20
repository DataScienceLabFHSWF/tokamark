

# ======================================================================================================================
class StdScalingTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, d):
        """
        Normalize each sample individually: subtract mean, divide by std.

        Input: dict with 'time' and 'values' [features, time]
        Output: same dict, with values normalized per feature
        """

        time = d['time']
        values = d['values']

        if values is not None:
            values = (values - self.mean) / self.std
        return {
            'time': time,
            'values': values
        }

    # ------------------------------------------------------------------------------------------------------------------

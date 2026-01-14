import numpy as np


# ======================================================================================================================
class DownsampleTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, factor: int):
        """Create a transform that downsamples a signal along the time dimension.

        Parameters
        ----------
        factor:
            Integer downsampling factor. Uses simple decimation (every `factor`-th sample).
            factor=1 returns the input unchanged.

        Notes
        -----
        This transform assumes time is the last axis for multi-dimensional arrays (e.g. (C, T), (H, W, T)).
        For robustness, if `values.shape[0] == len(time)` it will treat axis=0 as the time axis instead.
        """
        f = int(factor)
        if f < 1:
            raise ValueError(f"DownsampleTransform: factor must be >= 1, got {factor!r}")
        self.factor = f

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, dict_):
        """Downsample `dict_['time']` and `dict_['values']` by `self.factor`.

        Input: dict with keys:
            - 'time': 1D array-like of time points (length T) or None
            - 'values': array-like with a time dimension

        Output: same dict structure, with time/values decimated by `factor`.
        """
        time = dict_.get('time', None)
        values = dict_.get('values', None)

        if values is None:
            return {'time': time, 'values': values}

        x = np.asarray(values)
        f = self.factor

        if f == 1:
            return {'time': time, 'values': x}

        # Downsample time (if provided and indexable).
        time_ds = time
        T_time = None
        if time is not None:
            try:
                time_arr = np.asarray(time)
                if time_arr.ndim >= 1 and time_arr.shape[0] > 0:
                    time_ds = time_arr[::f]
                    T_time = int(time_arr.shape[0])
            except Exception:
                # Leave time untouched if it cannot be indexed.
                time_ds = time
                T_time = None

        # Decide which axis is time.
        axis = -1
        if T_time is not None:
            if x.ndim == 1 and x.shape[0] == T_time:
                axis = 0
            elif x.ndim >= 2 and x.shape[-1] == T_time:
                axis = -1
            elif x.ndim >= 2 and x.shape[0] == T_time:
                axis = 0

        # Apply decimation along selected axis.
        if axis == -1:
            x_ds = x[..., ::f]
        else:
            x_ds = x[::f, ...]

        return {'time': time_ds, 'values': x_ds}

    # ------------------------------------------------------------------------------------------------------------------

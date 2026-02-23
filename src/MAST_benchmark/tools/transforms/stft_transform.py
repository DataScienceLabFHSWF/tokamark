import numpy as np


# ======================================================================================================================
def stft(data, times, support_n=512):
    """
    data:   1D array, shape (T,)
    times:  1D array, shape (T,)
    return:
        spectrum: (n_frames, support_n//2)
        frame_times: (n_frames,)
    """
    xx = np.linspace(0, 2 * np.pi, support_n)
    window = (1 - np.cos(xx)) / 2

    nnp = int(len(data) / support_n)  # number of non-overlapping windows

    if nnp == 0:
        # No full window fits -> return empty arrays
        return (
            np.zeros((0, support_n // 2), dtype=float),
            np.zeros((0,), dtype=float),
        )

    window = np.tile(window, nnp)

    data = data[:nnp * support_n] * window
    data = data.reshape(nnp, support_n)
    times = times[:nnp * support_n].reshape(nnp, support_n)

    # Simple frame time: mean time within each window
    # frame_times = times.mean(axis=1)  # (n_frames,)
    frame_times = [t[-1] for t in times ]  # (n_frames,)

    spectrum = np.fft.fft(data)
    spectrum = spectrum[:, :support_n // 2]

    spectrum = np.abs(spectrum * np.conjugate(spectrum))

    # spectrum shape: (n_frames, support_n//2)
    return spectrum, frame_times


# ======================================================================================================================
class STFTTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, support_n=512):
        self.support_n = support_n

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, dict_):
        """
        Input: dict with keys 'time' and 'values' (values shape: (C,T) or (T,))
        Returns: dict with 'time' and STFT 'values' (shape: (C,n_frames,support_n//2))
        """
        time = np.asarray(dict_['time'])
        values = np.asarray(dict_['values'])

        if values.ndim == 1:
            values = values[None, :]

        if time.shape[0] != values.shape[-1]:
            raise ValueError(
                f"Time length ({time.shape[0]}) does not match values length ({values.shape[-1]})."
            )

        spectra = []
        frame_times = None
        for ch in range(values.shape[0]):
            spectrum, frame_times = stft(values[ch], time, support_n=self.support_n)
            spectra.append(spectrum)

        if not spectra:
            spectrum_stack = np.zeros((0, 0, self.support_n // 2), dtype=float)
        else:
            spectrum_stack = np.stack(spectra, axis=0)

        # print(spectrum_stack.transpose(0, 2, 1).shape)

        return {
            'time': frame_times,
            # 'values': spectrum_stack
            'values': spectrum_stack.transpose(0, 2, 1)
        }

    # ------------------------------------------------------------------------------------------------------------------

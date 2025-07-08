import numpy as np
from collections import defaultdict


# ======================================================================================================================
class CNNTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, list_samples):
        """
        Input:
        ------
        torch : dict
            Dictionary with key 'source-signal' containing dict with keys 'time' and key 'values'

        Returns:
        --------
        listtorch : dict
            Dictionary with key 'time' and key 'values with NaNs forward-filled
        """
        # print('CNN-specific formatting')
        
        cnn_samples = []

        for sample in list_samples:
            # reduce data to have same time 
            array_x = [data['values'].T for var, data in sample['x'].items()]
            array_y = [data['values'].T for var, data in sample['y'].items()]

            # group arrays by shape for CNN branch
            reshaped_x = self._group_arrays_by_shape(array_x)
            reshaped_x = [arr.squeeze(0) for arr in reshaped_x]
            reshaped_x = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in reshaped_x]
            reshaped_y = self._group_arrays_by_shape(array_y)
            reshaped_y = [arr.squeeze(0) for arr in reshaped_y]
            reshaped_y = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in reshaped_y]

            cnn_samples.append((reshaped_x, reshaped_y))
        
        return cnn_samples

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _group_arrays_by_shape(arrays):
        grouped = defaultdict(list)
        for arr in arrays:
            grouped[arr.shape].append(arr)
        reshaped_list = []
        for same_shape_group in grouped.values():
            stacked = np.stack(
                        [a for a in same_shape_group]
                    )
            transposed = np.transpose(stacked, axes=[1, 0] + list(range(2, stacked.ndim)))
            reshaped_list.append(transposed)

        return reshaped_list

    # ------------------------------------------------------------------------------------------------------------------

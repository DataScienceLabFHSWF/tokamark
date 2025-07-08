import numpy as np
from collections import defaultdict
import torch

# ======================================================================================================================
class TTMTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, parameters_ttm):
        self.context_length = parameters_ttm['context_length']
        self.forecast_length = parameters_ttm['forecast_length']
        self.target_columns = parameters_ttm['target_columns']
        self.observable_columns = parameters_ttm['observable_columns']

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, list_samples):
        # print('TTM-specific formatting')
        
        ttm_samples = []

        for sample in list_samples:

            past_target_tensor = np.concatenate([sample['x'][var]['values'] for var in self.target_columns])
            past_observable_tensor = np.concatenate([sample['x'][var]['values'] for var in self.observable_columns])
            past_tensor = torch.tensor(np.concatenate([past_target_tensor, past_observable_tensor])).T[-self.context_length:]
            # print(past_tensor.shape)

            future_target_tensor = np.concatenate([sample['y'][var]['values'] for var in self.target_columns])
            future_observable_tensor = np.concatenate([sample['y'][var]['values'] for var in self.observable_columns])
            future_tensor = torch.tensor(np.concatenate([future_target_tensor, future_observable_tensor])).T[:self.forecast_length]
            # print(future_tensor.shape)

            ttm_samples.append({
                'past_values': past_tensor.to(torch.bfloat16),
                'future_values': future_tensor.to(torch.bfloat16)
            })
        
        return ttm_samples

    # ------------------------------------------------------------------------------------------------------------------
    

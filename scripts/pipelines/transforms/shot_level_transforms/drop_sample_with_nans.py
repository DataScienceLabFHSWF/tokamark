import numpy as np 


# ======================================================================================================================
class DropSampleWithNans:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, list_samples):
        # print('TTM-specific formatting')
        
        no_nans_samples = []

        for sample in list_samples:
            # print(d.keys())
            # window_index = sample['window_index']
            d_x = sample['x']
            d_y = sample['y']

            accepted = True
            for var, d_var in d_x.items():
                if np.isnan(d_var['values']).any():
                    # print(f"Nans still present in variable {var}")
                    accepted = False
                    break
                else:
                    continue
            
            for var, d_var in d_y.items():
                if np.isnan(d_var['values']).any():
                    # print(f"Nans still present in variable {var}")
                    accepted = False
                    break
                else:
                    continue
            
            if accepted:
                no_nans_samples.append(sample)

        if self.verbose:
            print(f"Going from {len(list_samples)} windows to {len(no_nans_samples)} windows due to Nans in shot!!!")
        return no_nans_samples

    # ------------------------------------------------------------------------------------------------------------------

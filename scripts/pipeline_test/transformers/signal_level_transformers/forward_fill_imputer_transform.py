
import pandas as pd

class ForwardFillImputerTransform:
    def __call__(self, dict):
        """
        Input: torch dict with key 'time' and key 'values'
        Returns: torch dict with key 'time' and key 'values with NaNs forward-filled
        """
        # print('\nCECILE TRANSFORM')
        time = dict['time']
        values = dict['values']
        df = pd.DataFrame(values.T)
        # print('\nBefore forward filling: ', list(df.isna().sum(axis=0)))
        df = df.ffill(axis=0)
        # print('After forward filling: ', list(df.isna().sum(axis=0)))
        return {'time': time,
                'values': df.values.T}


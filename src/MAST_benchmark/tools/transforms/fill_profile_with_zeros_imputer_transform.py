"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import numpy as np
from typing import Any
from collections.abc import Mapping


# ======================================================================================================================
class FillProfileWithZerosTransform:
    """
    Transform to fill profile with zeros.

    Attributes
    ----------
    None.

    Methods
    -------

    __call__(dict_)
        Call method for the class to behave like a function.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(
            self,
            dict_: Mapping[str, Any]
    ) -> dict[str, Any]:
        """
        Call method for the class to behave like a function.

        Parameters
        ----------
        dict_ : Mapping[str, Any]
            Torch dict with "time" and "key" keys and corresponding values.

        Returns
        -------
        dict[str, Any]
            Torch dict with "time" and "key" keys where the NaNs of profiles (i.e., when one full channel in the
            profile is missing) are filled with zeros.

        """
                
        time = dict_["time"]
        values = dict_["values"].copy()

        # Checking for indexes with NaN values
        nan_ind = np.isnan(values)

        # Excluding columns comprised of NaNs only
        nan_cols = np.isnan(values).all(axis=0)
        nan_ind[:, nan_cols] = False

        # Replacing NaNs in profile components
        values[nan_ind] = 0

        return {
            "time": time,
            "values": values
        }

    # ------------------------------------------------------------------------------------------------------------------

# FIXME: Should we remove the commented code below? [Cecile, Mike]

# import pandas as pd
# import matplotlib.pyplot as plt

# class FillProfileWithZerosTransform:

#     # ----------------------------------------------------------------------------------------------------------------
#     def __call__(self, dict_):
#         """
#         Input: torch dict with key 'time' and key 'values'.

#         Returns: torch dict with key 'time' and key 'values with NaNs of profiles (i.e., when one full channel in the
#         profile is missing) filled with zeros. Also saves before/after plot.
#         """

#         time = dict_['time']
#         values = dict_['values']
#         df = pd.DataFrame(values)

#         # Save before-filling plot
#         plt.figure(figsize=(10, 6))
#         plt.imshow(df.T, aspect='auto', interpolation='none', cmap='viridis')
#         plt.colorbar(label='Value')
#         plt.xlabel('Time Index')
#         plt.ylabel('Channel')
#         plt.title('Profile Before Filling NaNs')
#         plt.savefig('thomson_profile_before.png')
#         plt.close()

#         # Fill NaNs with zeros for columns that are not entirely NaN
#         for col in df.columns:
#             if not df[col].isna().all():
#                 df[col] = df[col].fillna(value=0)

#         # Save after-filling plot
#         plt.figure(figsize=(10, 6))
#         plt.imshow(df.T, aspect='auto', interpolation='none', cmap='viridis')
#         plt.colorbar(label='Value')
#         plt.xlabel('Time Index')
#         plt.ylabel('Channel')
#         plt.title('Profile After Filling NaNs')
#         plt.savefig('thomson_profile_after.png')
#         plt.close()

#         return {
#             'time': time,
#             'values': df.values
#         }

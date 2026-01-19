import numpy as np


# ======================================================================================================================
class FillProfileWithZerosTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, dict_):
        time = dict_['time']
        values = dict_['values'].copy()

        # checking for indeces with NaN values
        nan_ind = np.isnan(values)
        # excluding columns comprised of NaNs only
        nan_cols = np.isnan(values).all(axis=0)
        nan_ind[:,nan_cols] = False
        # replacing NaNs in profile components
        values[nan_ind] = 0

        return {
            'time': time,
            'values': values
        }

    # ------------------------------------------------------------------------------------------------------------------



# import pandas as pd
# import matplotlib.pyplot as plt

# class FillProfileWithZerosTransform:

#     # ------------------------------------------------------------------------------------------------------------------
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


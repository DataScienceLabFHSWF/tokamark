import pandas as pd


# ======================================================================================================================
class FillProfileWithZerosTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, dict_):
        """
        Input: torch dict with key 'time' and key 'values'.

        Returns: torch dict with key 'time' and key 'values with NaNs of profiles (i.e., when one full channel in the
        profile is missing) filled with zeros.
        """

        time = dict_['time']
        values = dict_['values']
        df = pd.DataFrame(values)
        # print('\nBefore filling with zeros: ', list(df.isna().sum(axis=0)))
        for col in df.columns:
            if not df[col].isna().all():
                df[col] = df[col].fillna(value=0)
        # print('After filling with zeros: ', list(df.isna().sum(axis=0)))
        
        return {
            'time': time,
            'values': df.values
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


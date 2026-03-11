"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import numpy as np
from scipy.ndimage import zoom
from typing import Any
from collections.abc import Mapping


# ======================================================================================================================
class ReshapeLcfsTransform:
    """
    Transform to reshape LCFS profiles.

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
            Torch dict with "time" and "key" keys where NaNs of profiles (i.e., when one full channel in the profile
            is missing) are filled with zeros.
            # FIXME: This description is verbatim copy from other transform, and it does not fit here. [Cecile]

        """

        time = dict_["time"]
        values = dict_["values"]

        new_profile = []

        for vector in values.T:
            cleaned_vector = vector[~np.isnan(vector)]
            len_cleaned_vector = len(cleaned_vector)
            if len_cleaned_vector == 0:
                resized_vector = np.full(shape=(170,), fill_value=np.nan)
            else:
                scale_factor = 170 / len_cleaned_vector
                resized_vector = zoom(input=cleaned_vector, output=scale_factor, order=1)  # order=1 = linear interp.
            new_profile.append(resized_vector)

        return {
            "time": time,
            "values": np.stack(new_profile).T
        }

    # ------------------------------------------------------------------------------------------------------------------

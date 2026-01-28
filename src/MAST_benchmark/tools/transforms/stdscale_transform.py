"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

from typing import Mapping


# ======================================================================================================================
class StdScalingTransform:
    """
    STD scaling transform.

    Methods
    -------
    __call__
        Call method for the class to behave like a function. It normalizes each sample individually: subtract mean,
        divide by std.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self,
            mean: float,
            std: float
    ) -> None:
        """
        Initialise class attributes.

        Parameters
        ----------
        mean : float
            Input mean.
        std : float
            Input STD.

        """

        self.mean = mean
        self.std = std

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(
            self,
            d: Mapping
    ):
        """
        Call method for the class to behave like a function. It normalizes each sample individually: subtract mean,
        divide by std.

        Parameters
        ----------
        d : Mapping
            Dictionary with 'time' and 'values' [features, time].

        Returns
        -------
        Mapping
            Augmented input dictionary with values normalized per feature.
        """

        time = d['time']
        values = d['values']

        if values is not None:
            values = (values - self.mean) / self.std
        return {
            'time': time,
            'values': values
        }

    # ------------------------------------------------------------------------------------------------------------------

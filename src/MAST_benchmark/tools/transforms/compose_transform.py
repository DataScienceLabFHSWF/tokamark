"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

from typing import Any


# ======================================================================================================================
class ComposeTransforms(object):
    """
    Compose transforms and apply them in series checking for None return values

    Attributes
    ----------
    transforms : list[callable[tuple]]
        List containing the names of the transforms

    Methods
    -------
    __call__
        Call method for the class to behave like a function.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self,
            transforms: Any
    ) -> None:
        """
        Initialise class attributes.

        Parameters
        ----------
        transforms : list[callable[tuple]]
            List containing the names of the transforms

        """

        self.transforms = transforms

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(
            self,
            sample: Any
    ) -> Any:
        """
        Call method for the class to behave like a function.

        Parameters
        ----------
        sample : Any
            Target sample.

        Returns
        -------
        Any
            Transformed sample.

        """

        for transform in self.transforms:
            if sample is None:
                return None
            sample = transform(sample)
        return sample

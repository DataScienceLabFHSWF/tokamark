"""
compose_shot_transform.py

Defines a utility class for composing multiple transforms that operate
on the entire shot dictionary.

This is analogous to per-signal ComposeTransform but is designed for
model-specific transforms that require access to all signals at once.

Typical use cases:
- Applying ShotWindowSegmenter
- Applying transformations that join or compare multiple signals
- Postprocessing entire shots for model input
"""

class ComposeShotTransforms:
    """
    A utility class to sequentially apply a list of shot-level transforms.

    Each transform in the list should accept a shot dictionary as input
    and return a (potentially modified) shot dictionary as output.

    Example usage:
        model_specific_transform = ComposeShotTransform([
            ShotWindowSegmenterTransform(...),
            SomeOtherShotLevelTransform(...)
        ])
    """

    def __init__(self, transforms):
        """
        Parameters
        ----------
        transforms : list of callables
            A list of transform objects or functions. Each must accept and return
            a full shot dictionary of the form:
            {
                'signal_name': {'time': np.ndarray, 'values': np.ndarray},
                ...
            }
        """
        self.transforms = transforms

    def __call__(self, shot):
        """
        Apply each transform in order to the input shot.

        Parameters
        ----------
        shot : dict
            Dictionary containing all signals in the shot, with each signal mapped to
            its {'time': ..., 'values': ...} structure.

        Returns
        -------
        dict
            The transformed shot dictionary after all transforms have been applied.
        """
        for transform in self.transforms:
            shot = transform(shot)
        return shot

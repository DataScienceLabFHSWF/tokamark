"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

from MAST_tools.utils.path_utils import PROJECT_ROOT_DIR


# ----------------------------------------------------------------------------------------------------------------------

ARTIFACTS_DIR = PROJECT_ROOT_DIR / "artifacts"
SHOT_STATS_DIR = ARTIFACTS_DIR / "shots_stats"

OUTPUT_DIR = PROJECT_ROOT_DIR / "output"

# TODO: Check impact of new filenames. [Cecile]
DEFAULT_SHOTS_STATS_TEST_FILE = SHOT_STATS_DIR / "shots_stats_test.csv"  # Before: shot_statistics_test_NEW.csv
DEFAULT_SHOTS_STATS_TRAIN_FILE = SHOT_STATS_DIR / "shots_stats_train.csv"  # Before: shot_statistics_train_NEW.csv
DEFAULT_SHOTS_STATS_VAL_FILE = SHOT_STATS_DIR / "shots_stats_val.csv"  # Before: shot_statistics_val_NEW.csv

SIGNALS_STATS_DIR = ARTIFACTS_DIR / "signals_stats"
DEFAULT_SIGNALS_MEAN_STD_TRAIN_FILE = SIGNALS_STATS_DIR / "dict_signals_mean_std_train.yaml"

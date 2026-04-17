"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

from MAST_tools.utils.path_utils import PROJECT_ROOT_DIR


# ----------------------------------------------------------------------------------------------------------------------

ARTIFACTS_DIR = PROJECT_ROOT_DIR / "artifacts"
SHOT_STATS_DIR = ARTIFACTS_DIR / "shots_stats"
PREPROC_STATS_DIR = ARTIFACTS_DIR / "preprocessing_stats"

OUTPUT_DIR = PROJECT_ROOT_DIR / "output"

DEFAULT_SHOTS_STATS_ALL_FILE = SHOT_STATS_DIR / "shots_stats_all.csv"
DEFAULT_SHOTS_STATS_TEST_FILE = SHOT_STATS_DIR / "shots_stats_test.csv"
DEFAULT_SHOTS_STATS_TRAIN_FILE = SHOT_STATS_DIR / "shots_stats_train.csv"
DEFAULT_SHOTS_STATS_VAL_FILE = SHOT_STATS_DIR / "shots_stats_val.csv"

SIGNALS_STATS_DIR = ARTIFACTS_DIR / "signals_stats"
DEFAULT_SIGNALS_MEAN_STD_TRAIN_FILE = SIGNALS_STATS_DIR / "dict_signals_mean_std_train.yaml"

SHOT_SUMMARY_FILE = PREPROC_STATS_DIR / "shot_summary.csv"

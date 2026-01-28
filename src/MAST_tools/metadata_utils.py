"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import urllib.request
import pandas as pd
import os
import json


# ----------------------------------------------------------------------------------------------------------------------
def download_all_signals_metadata_parquet(
        sources_signals_json_file="artifacts/2025-04-17/data_level2_sources_with_signals.json",
        level=2,
        target_subdir="all_signals",
        rewrite_files=False,
        verbose=False
) -> None:
    """
    Download signals parquet metadata for all signals in source-signal file.

    Parameters
    ----------

    sources_signals_json_file : str
        Source file for source-signal information.
    level : int
        Target level for the MAST metadata to be pulled.
        Optional. Default: 2.
    target_subdir :  str
        Target subdirectory for parquet files with signals metadata.
    rewrite_files : bool
        If True, signals parquet metadata files are rewritten.
        Optional. Default: False.
    verbose : bool
        If True, verbose mode is activated.
        Default: False.

    Return
    ------
    None

    """

    with open(sources_signals_json_file, 'r') as file:
        sources_with_signals = json.load(file)
    signal_names = sum(list(sources_with_signals.values()), [])

    full_target_subdir = f"artifacts/parquet/level{level}/{target_subdir}"
    if not os.path.isdir(full_target_subdir):
        os.makedirs(full_target_subdir)

    for signal_name in signal_names:
        target_url = f"http://mastapp.site/parquet/level{level}/signals?name={signal_name}"
        target_filename = f"{full_target_subdir}/{signal_name}.parquet"

        save_file = False
        if os.path.isfile(target_filename):
            if rewrite_files:
                save_file = True
        else:
            save_file = True

        if save_file:
            try:
                urllib.request.urlretrieve(
                    url=target_url,
                    filename=target_filename
                )
                if verbose:
                    print(f"Saved parquet metadata file for signal {signal_name}.")
            except Exception as ee:
                print(ee)


# ----------------------------------------------------------------------------------------------------------------------
def download_shots_metadata_parquet(
        level=2,
        rewrite_file=False,
        verbose=False
) -> None:
    """
    Download parquet file with shots metadata.

    Parameters
    ----------
    level : int
        Target level for the MAST metadata to be pulled.
        Optional. Default: 2.
    rewrite_file : bool
        If True, parquet file with shots metadata is rewritten.
        Optional. Default: False.
    verbose : bool
        If True, verbose mode is activated.
        Default: False.

    Return
    ------
    None

    """

    target_subdir = f"artifacts/parquet/level{level}"
    target_url = f"https://mastapp.site/parquet/level{level}/shots"
    target_filename = f"{target_subdir}/shots_metadata.parquet"

    save_file = False
    if os.path.isfile(target_filename):
        if rewrite_file:
            save_file = True
    else:
        save_file = True

    if save_file:
        try:
            urllib.request.urlretrieve(url=target_url, filename=target_filename)
            if verbose:
                print("Saved parquet file with shots metadata.")
        except Exception as ee:
            print(ee)


# ----------------------------------------------------------------------------------------------------------------------
def download_signals_per_shot_metadata_parquet(
        level=2,
        local_metadata_file=True,
        target_subdir="signals_per_shot",
        rewrite_files=False,
        verbose=False
) -> None:
    """
    Download parquet files with signals per shot metadata.

    Parameters
    ----------
    level :  int
        Target level for the MAST metadata to be pulled.
        Optional. Default: 2.
    local_metadata_file : bool
        If True, the local parquet file with shots metadata is used, otherwise a remote one is pulled.
        Optional. Default: True.
    target_subdir : str
        Target subdirectory for parquet files with signals per shot metadata.
    rewrite_files : bool
        If True, parquet files with signals per shot metadata are rewritten.
        Optional. Default: False.
    verbose : bool
        If True, verbose mode is activated.
        Default: False.

    Return
    ------
    None

    """

    if local_metadata_file:
        level_metadata = pd.read_parquet(f"artifacts/parquet/level{level}/shots_metadata.parquet")
    else:
        level_metadata = pd.read_parquet(f"https://mastapp.site/parquet/level{level}/shots")

    shot_ids = level_metadata["shot_id"].values.tolist()

    full_target_subdir = f"artifacts/parquet/level{level}/{target_subdir}"
    if not os.path.isdir(full_target_subdir):
        os.makedirs(full_target_subdir)

    for shot_id in shot_ids:

        target_url = f"https://mastapp.site/parquet/level{level}/signals?shot_id={shot_id}"
        target_filename = f"{full_target_subdir}/signals_{shot_id}.parquet"

        save_file = False
        if os.path.isfile(target_filename):
            if rewrite_files:
                save_file = True
        else:
            save_file = True

        if save_file:
            try:
                urllib.request.urlretrieve(url=target_url, filename=target_filename)
                if verbose:
                    print(f"Saved parquet file with signal metadata for shot {shot_id}.")
            except Exception as ee:
                print(ee)


# ----------------------------------------------------------------------------------------------------------------------
def tests() -> None:
    """
    Quick tests for module functionality.

    Return
    ------
    None

    """

    TESTS_TO_RUN = {  # noqa
        "download_all_signals_parquet": False,
        "download_shots_metadata_parquet": False,
        "download_signals_per_shot_parquet": False,
    }

    # ..................................................................................................................
    if TESTS_TO_RUN["download_all_signals_parquet"]:
        download_all_signals_metadata_parquet(
            level=2,
            sources_signals_json_file="artifacts/2025-04-17/data_level2_sources_with_signals.json",
            rewrite_files=False,
            verbose=True
        )

    # ..................................................................................................................

    if TESTS_TO_RUN["download_shots_metadata_parquet"]:
        download_shots_metadata_parquet(
            level=2,
            rewrite_file=False,
            verbose=True
        )

    # ..................................................................................................................
    if TESTS_TO_RUN["download_signals_per_shot_parquet"]:
        download_signals_per_shot_metadata_parquet(
            level=2,
            local_metadata_file=False,
            target_subdir="signals_per_shot",
            rewrite_files=False,
            verbose=True
        )


# ======================================================================================================================
if __name__ == "__main__":
    tests()

"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import os

# Dynamically find the repo root (no hardcoding!)
PACKAGE_ROOT = os.path.abspath(  # FIXME: Make PACKAGE_ROOT match the new architecture. [Rodrigo]
    os.path.join(os.path.dirname(__file__), "..")
)

# Optional: define commonly used subdirectories
TASKS_CONFIGS_DIR = os.path.join(PACKAGE_ROOT, "tasks_configs")  # FIXME: Make TASKS_CONFIGS_DIR match the new architecture. [Rodrigo]
METADATA_DIR = os.path.join(PACKAGE_ROOT, "metadata")  # FIXME: Make this METADATA_DIR match the new architecture. [Rodrigo]


# ======================================================================================================================
if __name__ == "__main__":

    # One can print to verify when developing

    print("Repo root:", PACKAGE_ROOT)
    print("Tasks configs dir:", TASKS_CONFIGS_DIR)
    print("Metadata dir:", METADATA_DIR)

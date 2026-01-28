"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import os

# Dynamically find the repo root (no hardcoding!)
PACKAGE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Optional: define commonly used subdirectories
TASKS_CONFIGS_DIR = os.path.join(PACKAGE_ROOT, "tasks_configs")
METADATA_DIR = os.path.join(PACKAGE_ROOT, "metadata")


# ======================================================================================================================
# You can print to verify when developing
if __name__ == "__main__":
    print("Repo root:", PACKAGE_ROOT)
    print("Tasks configs dir:", TASKS_CONFIGS_DIR)
    print("Metadata dir:", METADATA_DIR)

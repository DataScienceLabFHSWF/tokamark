import os
import sys

# Dynamically find the repo root (no hardcoding!)
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Optional: define commonly used subdirectories
CONFIG_DIR = os.path.join(REPO_ROOT, "scripts", "pipelines", "configs")
DATA_DIR = os.path.join(REPO_ROOT, "metadata")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")

# You can print to verify when developing
if __name__ == "__main__":
    print("Repo root:", REPO_ROOT)
    print("Config dir:", CONFIG_DIR)
    print("Data dir:", DATA_DIR)

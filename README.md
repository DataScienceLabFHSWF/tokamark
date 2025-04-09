# FAIR-MAST Data Preprocessing

For usage on CSD3.

## Setup

1. Install [Miniforge](https://github.com/conda-forge/miniforge):
   ```bash
   wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```

2. Set up conda env:
   ```bash
   source ~/miniforge3/bin/activate
   conda env create -f environment.yml
   conda activate fairmast-data-preprocessing
   ```
# FAIR-MAST Data Preprocessing

Data preprocessing of the FAIR-MAST data for foundation modelling.

Initialised as part of HNCDI project Fusion Plasma Modelling HT07632.

For usage on [CSD3](https://docs.hpc.cam.ac.uk/hpc/index.html).

## Initial setup

1. Connect to CSD3 following the instructions in the [user guide](https://docs.hpc.cam.ac.uk/hpc/user-guide/quickstart.html).
2. Install Miniforge by following the instructions [here](https://github.com/conda-forge/miniforge). In brief:  
   1. Find the latest [release](https://github.com/conda-forge/miniforge/releases/).
   2. Download the installer. For example:
      ```bash
      cd ~
      wget https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh
      ```
   3. Run the installer. For example:
      ```bash
      chmod +x Miniforge3-24.11.3-2-Linux-x86_64.sh
      ./Miniforge3-24.11.3-2-Linux-x86_64.sh
      ```
   4. Source the base environment:
      ```bash
      source ~/miniforge3/bin/activate
      ```
3. Set up the project virtual environment:
   1. Move to the directory containing this repository. For example:
      ```bash
      cd ~/hncdi-fusion-plasma/fairmast-data-preprocessing
      ```
   2. Use conda to set up the virtual environment and install the dependencies:
      ```bash
      conda env create -f environment.yml
      ```
4. Activate the project conda environment:
   ```bash
   conda activate fairmast-data-preprocessing
   ```

## CSD3 instructions

Running computational workloads on CSD3 (including Jupyter Notebooks) should be done using the worker nodes.

To get a terminal shell running on 4 cores of worker node for 1 hour:

```bash
sintr -A ukaea-ap002-cpu -p ukaea-icl -N1 -n4 -t 1:00:00
```

To run a Jupyter notebook on a worker node using VSCode:

1. On the local machine, add an entry to `~/.ssh/config`:
   ```
   Host csd3
     User <user-name>
     HostName login-cpu.hpc.cam.ac.uk
     IdentityFile </path/to/id_rsa>
   ```
2. In VSCode, install the "Remote Development" extension pack (if not already installed).
3. In VSCode, run the command "Remote-SSH: Connect to Host" and select "csd3" from the dropdown menu. VSCode will now connect to a VSCode server running on a CSD3 login node.
4. In the VSCode terminal, start an interactive session using the command above. Note down the node address, which appears as part of the terminal prompt prefix with the format `[<user-name>@<node-address>]` and will be something like `cpu-q-123`.
5. In the interactive session, start a Jupyter notebook server:
   ```bash
   jupyter notebook --no-browser --ip=* --port=8081
   ```
   Wait for the server to start then copy the address, which should have the format:
   ```
   http://localhost:8081/tree?token=<token>
   ```
6. In another VSCode terminal (i.e., not the interactive one), tunnel to the worker node and forward the Jupyter server port:
   ```bash
   ssh -L 8081:localhost:8081 <user-name>@<node-address>
   ```
7. In VSCode, open up a `.ipynb` file and connect to the server running on the worker node:
   1. Run the command `Notebook: Select Notebook Kernel`.
   2. Click "Select Another Kernel".
   3. Select "Existing Jupyter Server".
   4. Paste in the server address noted down before and hit enter. Hit enter on the subsequent input boxes to accept the default values, or modify them if needed.

Any cells executed in the notebook will now execute on the kernel running on the reserved cores of the worker node.
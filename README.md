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
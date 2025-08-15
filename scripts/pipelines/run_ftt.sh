#!/bin/bash
# Example SLURM job for Wilkes3

#SBATCH -J ftt_tobia_gpujob
#SBATCH -A ukaea-ap002-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ukaea-amp
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# --- Environment --------------------------------------------------------------
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

# If you need extra modules or conda activation, do it here, e.g.:
# module load Miniconda3/4.12.0
# source ~/.bashrc
# conda activate ftt-evn

# --- App paths ---------------------------------------------------------------
application="/home/ir-rous1/conda/envs/ftt-evn/bin/python3.11"
options="/home/ir-rous1/hncdi-fusion-plasma/fairmast-data-preprocessing/scripts/main_pipeline/ftt_pipeline.py"

workdir="$SLURM_SUBMIT_DIR"
mkdir -p logs
cd "$workdir"

echo "Changed directory to: $(pwd)"
echo "JobID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Command: $application $options"

# --- Launch (srun binds resources correctly) ---------------------------------
srun "$application" "$options"

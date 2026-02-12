#!/bin/bash
# Usage: sbatch --account=$ACCOUNT_NAME ./scripts/download_data.sh
# source .env && sbatch -A $ACCOUNT_NAME ./scripts/download_data.sh

#SBATCH --time=3-00:00:00
#SBATCH --job-name=download_data
#SBATCH --ntasks=1
#SBATCH --partition=H8V141_SAP112M2000_L
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -e ./logs/err_%j.log
#SBATCH -o ./logs/out_%j.log
#SBATCH --export=NONE

unset LD_LIBRARY_PATH

module load ccs/singularity
module load ccs/cuda/12.2.0_535.54.03

IMG=/share/singularity/images/ccs/rocky/rocky8.sinf
# IMG=/share/singularity/images/ccs/conda/lcc-jupyter-rocky8.sinf

# Load environment variables from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

echo "---- Running inside container ----"
singularity exec --nv "$IMG" bash -lc '
  set -euo pipefail
  echo "Container OS: $(grep PRETTY_NAME /etc/os-release)"
  echo "whoami: $(whoami)"
  echo "pwd: $(pwd)"
  echo "python: $(which python || true)"
  echo "uv: $(which uv || true)"
  echo "TMPDIR=${TMPDIR:-unset}"
  echo "ulimit -a:"
  ulimit -a

  # Print job info (from SBATCH directives)
  echo "Job ID: ${SLURM_JOB_ID:-unknown}"
  echo "Job Name: ${SLURM_JOB_NAME:-unknown}"
  echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
  echo "CPUs per Task: ${SLURM_CPUS_PER_TASK:-unknown}"
  echo "GPUs: ${SLURM_GPUS:-unknown}"
  
  nvidia-smi || true
  
  uv sync
  uv run -m dataset_builder.generate \
    --dataset_name ruslanmv/ai-medical-chatbot \
    --base_model meta-llama/Llama-3.1-8B \
    --output_dir ./data/modified \
    --batch_size 64 \
    --temperature 0.7 \
    --sim_threshold 0.90 \
    --sim_keep_high_frac 0.05 \
    --enable_judge_filter \
    --judge_min_margin 0.20 \
    --judge_min_chosen_score 0.15 \
    --checkpoint_dir ./data/modified \
    --checkpoint_every 500
  echo "---- Container execution completed ----"
'

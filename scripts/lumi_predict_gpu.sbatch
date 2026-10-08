#!/bin/bash
#SBATCH --job-name=psmpredict
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=standard-g
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
mkdir -p logs /tmp/${USER}-miopen-cache
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/${USER}-miopen-cache
: "${LUMI_TORCH_SIF:?Set LUMI_TORCH_SIF to the current LUMI PyTorch .sif}"
srun singularity exec --rocm -B "$PWD:$PWD" "$LUMI_TORCH_SIF" bash -lc '
  source .venv/bin/activate
  psmpredict predict --checkpoint results/model/best.pt --input examples/peptides.tsv --output results/predicted_spectra.tsv
'

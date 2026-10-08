#!/bin/bash
#SBATCH --job-name=psmtrain
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=standard-g
#SBATCH --time=12:00:00
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
# Set this to the current LUMI AI Factory PyTorch Apptainer image documented for LUMI.
: "${LUMI_TORCH_SIF:?Set LUMI_TORCH_SIF to the current LUMI PyTorch .sif}"
srun singularity exec --rocm -B "$PWD:$PWD" "$LUMI_TORCH_SIF" bash -lc '
  source .venv/bin/activate
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  psmpredict train --data work/psms.pt --output-dir results/model --batch-size 512 --epochs 15 --num-workers 7
'

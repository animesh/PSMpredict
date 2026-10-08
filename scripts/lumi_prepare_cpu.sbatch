#!/bin/bash
#SBATCH --job-name=psmprep
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module purge
source .venv/bin/activate
psmpredict prepare data/msms.txt --output work/psms.pt --num-workers 16

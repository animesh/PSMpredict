# PSMpredict

GPU-ready PyTorch pipeline that learns HCD MS/MS fragment intensities from MaxQuant `msms.txt` and predicts spectra for peptide sequences. It runs on CPU, NVIDIA CUDA, and AMD ROCm through the normal PyTorch `cuda` API, including LUMI MI250X GPUs.

## Scope

The model predicts normalized intensities for b/y ions, fragment charges 1 and 2, and no-loss, H2O-loss, and NH3-loss channels at every peptide cleavage. It accepts peptide sequence, MaxQuant modified sequence, and precursor charge. Training labels come from `Matches` and `Intensities`; `Masses2/Intensities2` are not needed for supervised labels. Predictions include calculated fragment m/z and intensity.

This repository implements a self-contained transformer inspired by the sequence-to-spectrum approach used by Prosit, pDeep, and AlphaPeptDeep. It does not redistribute their weights. Training on your own MaxQuant data avoids CUDA-only dependencies and makes the exact runtime portable to ROCm.

## Literature basis

- Prosit established deep sequence-to-intensity and retention-time prediction for synthetic spectral libraries: Gessulat et al., *Nature Methods* 2019, DOI: 10.1038/s41592-019-0426-7.
- pDeep introduced recurrent prediction and later transfer-learning variants: Zhou et al., *Analytical Chemistry* 2017, DOI: 10.1021/acs.analchem.7b00006.
- MS2PIP is a widely used machine-learning intensity predictor: Degroeve and Martens, *Bioinformatics* 2013, DOI: 10.1093/bioinformatics/btt447.
- AlphaPeptDeep provides modular PyTorch models and transfer learning for MS2, RT, and CCS: Zeng et al., *Nature Communications* 2022, DOI: 10.1038/s41467-022-34904-3.

## Repository layout

- `src/psmpredict/core.py`: MaxQuant parser, labels, model, checkpoint handling.
- `src/psmpredict/workflows.py`: training and prediction workflows.
- `scripts/`: CPU preparation and LUMI GPU SLURM jobs.
- `tools/plotPSM.py`: the supplied spectrum plotting program, preserved unchanged.
- `tests/`: parser and model shape tests.

## Install

Use a PyTorch environment appropriate for the machine. On LUMI, start from the current LUMI AI Factory PyTorch Apptainer image so ROCm support is already matched to the system.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . pytest
pytest -q
```

Do not install a CUDA PyTorch wheel on LUMI. The LUMI container supplies ROCm PyTorch. The package intentionally does not list `torch` in `pyproject.toml` to avoid replacing that build.

## Prepare MaxQuant data

```bash
mkdir -p work results/model logs
psmpredict prepare data/msms.txt --output work/psms.pt --min-score 40
```

Multiple files are accepted:

```bash
psmpredict prepare data/run1/msms.txt data/run2/msms.txt --output work/psms.pt
```

The preparation report is written beside the tensor file as `work/psms.json`. Unknown modification names are reported and currently encoded with zero mass shift. Supported named modifications are oxidation, carbamidomethylation, phosphorylation, and protein N-terminal acetylation.

## Train

```bash
psmpredict train --data work/psms.pt --output-dir results/model --epochs 15 --batch-size 256
```

Outputs are `best.pt`, `last.pt`, and `history.tsv`. Validation uses a deterministic PSM split and mean spectral angle. Start with one GPU. Multi-GPU DDP is intentionally not included because a single MI250X GCD is sufficient for this compact model and is simpler to reproduce.

## Predict

Input TSV:

```text
Sequence    Modified sequence    Charge
PEPTIDE     _PEPTIDE_            2
```

Run:

```bash
psmpredict predict --checkpoint results/model/best.pt --input examples/peptides.tsv --output results/predicted_spectra.tsv
```

Each output row is one predicted fragment with peptide row ID, sequence, precursor charge, annotation, calculated m/z, and normalized predicted intensity.

## LUMI

1. Clone the repository on a login node and create the environment inside the current LUMI PyTorch container or a compatible writable overlay.
2. Replace `project_XXXXXXXXX` in each SLURM script with your allocation.
3. Set `LUMI_TORCH_SIF` to the current LUMI AI Factory PyTorch `.sif` path.
4. Prepare on CPU, then train and predict on `standard-g`:

```bash
sbatch scripts/lumi_prepare_cpu.sbatch
export LUMI_TORCH_SIF=/path/to/current-lumi-pytorch.sif
sbatch scripts/lumi_train_gpu.sbatch
sbatch scripts/lumi_predict_gpu.sbatch
```

The GPU scripts use `singularity exec --rocm`, one GPU/GCD, seven CPU cores, host-project binding, and local `/tmp` MIOpen caches. They print the exact PyTorch version and detected AMD device before training. Update container paths from current LUMI documentation rather than pinning a potentially obsolete image in this repository.

## Data leakage and benchmarking

The default split is by PSM, so repeated peptide sequences can occur in both train and validation sets. This is suitable for monitoring optimization but optimistic for publication claims. For an external benchmark, prepare a separate MaxQuant file from held-out raw files or instruments and compare predicted versus observed vectors by spectral angle. Do not mix technical replicates across benchmark boundaries.

## Limitations

- Only MaxQuant-annotated b/y ions with charges 1/2 and common neutral losses are modeled.
- Collision energy and instrument are not included because the supplied MaxQuant examples do not expose a reliable collision-energy column.
- Modified-sequence parsing covers common MaxQuant names. The preparation report identifies unsupported names.
- Predicted intensities are relative, not absolute ion counts.

## Reproducibility

The pipeline fixes Python, NumPy, and PyTorch random seeds. Exact GPU results can still differ slightly across PyTorch/ROCm versions. Record the container digest, SLURM job log, package commit, and `history.tsv` in publications.

from __future__ import annotations
import json, math, random, re
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AA)}
PROTON, H2O, NH3 = 1.007276466812, 18.010564684, 17.026549101
AA_MASS = dict(zip(AA, [71.037114,103.009185,115.026943,129.042593,147.068414,57.021464,137.058912,113.084064,128.094963,113.084064,131.040485,114.042927,97.052764,128.058578,156.101111,87.032028,101.047679,99.068414,186.079313,163.063329]))
ION_TYPES = tuple(f"{s}{z}{loss}" for s in ("b", "y") for z in (1, 2) for loss in ("", "-H2O", "-NH3"))
ION_TO_ID = {ion: i for i, ion in enumerate(ION_TYPES)}
MATCH_RE = re.compile(r"^([by])(\d+)(?:\((\d+)\+\))?(-H2O|-NH3)?$")

@dataclass
class Config:
    max_len: int = 60
    d_model: int = 192
    nhead: int = 6
    layers: int = 4
    dropout: float = 0.1
    batch_size: int = 256
    epochs: int = 15
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    seed: int = 42
    val_fraction: float = 0.1
    num_workers: int = 4
    min_score: float | None = None
    max_psms: int | None = None


def clean_sequence(x: object) -> str:
    return "".join(re.findall(r"[A-Z]", str(x).upper()))


def parse_floats(x: object) -> list[float]:
    if pd.isna(x): return []
    out = []
    for v in str(x).split(";"):
        try: out.append(float(v))
        except ValueError: out.append(float("nan"))
    return out


def modification_vector(sequence: str, modified: object) -> np.ndarray:
    vec = np.zeros(len(sequence), dtype=np.float32)
    text = str(modified)
    # Common MaxQuant modifications. Unknown names remain zero and are reported during preparation.
    for m in re.finditer(r"([A-Z])\(([^)]+)\)", text):
        residue, name = m.group(1), m.group(2).lower()
        prefix = text[:m.start()]
        idx = sum(c in AA for c in prefix) - 1
        shift = 15.994915 if "ox" in name else 57.021464 if "carbamidomethyl" in name else 79.966331 if "phospho" in name else 0.0
        if 0 <= idx < len(vec): vec[idx] += shift
    if "acetyl (protein n-term)" in text.lower() and len(vec): vec[0] += 42.010565
    return vec


def ion_key(label: str) -> tuple[int, int] | None:
    m = MATCH_RE.match(label.strip())
    if not m: return None
    series, ordinal, z, loss = m.groups()
    z = int(z or 1)
    if z not in (1, 2): return None
    channel = ION_TO_ID.get(f"{series}{z}{loss or ''}")
    return (int(ordinal), channel) if channel is not None else None


def row_to_record(row: pd.Series, max_len: int) -> dict | None:
    seq = clean_sequence(row.get("Sequence", ""))
    if not 2 <= len(seq) <= max_len or any(a not in AA_TO_ID for a in seq): return None
    matches = str(row.get("Matches", "")).split(";")
    intensities = parse_floats(row.get("Intensities", ""))
    target = np.zeros((len(seq) - 1, len(ION_TYPES)), dtype=np.float32)
    for label, intensity in zip(matches, intensities):
        parsed = ion_key(label)
        if parsed is None or not np.isfinite(intensity) or intensity < 0: continue
        ordinal, channel = parsed
        series = label[0]
        cut = ordinal - 1 if series == "b" else len(seq) - ordinal - 1
        if 0 <= cut < len(seq) - 1: target[cut, channel] = max(target[cut, channel], intensity)
    scale = float(target.max())
    if scale <= 0: return None
    target /= scale
    return {"sequence": seq, "modified_sequence": str(row.get("Modified sequence", seq)), "charge": int(row.get("Charge", 2)), "target": target, "score": float(row.get("Score", np.nan)), "raw_file": str(row.get("Raw file", "")), "scan": int(row.get("Scan number", -1))}


def prepare_msms(inputs: list[str], output: str, config: Config) -> dict:
    records, skipped, unknown_mods = [], 0, set()
    usecols = ["Sequence", "Modified sequence", "Charge", "Matches", "Intensities", "Score", "Raw file", "Scan number"]
    for path in inputs:
        for chunk in pd.read_csv(path, sep="\t", usecols=lambda c: c in usecols, chunksize=50000, low_memory=False):
            if config.min_score is not None and "Score" in chunk: chunk = chunk[pd.to_numeric(chunk["Score"], errors="coerce") >= config.min_score]
            for _, row in chunk.iterrows():
                rec = row_to_record(row, config.max_len)
                if rec is None: skipped += 1
                else:
                    rec["mod_mass"] = modification_vector(rec["sequence"], rec["modified_sequence"])
                    records.append(rec)
                    for name in re.findall(r"\(([^)]+)\)", rec["modified_sequence"]):
                        if not any(k in name.lower() for k in ("ox", "carbamidomethyl", "phospho", "acetyl")): unknown_mods.add(name)
                if config.max_psms and len(records) >= config.max_psms: break
            if config.max_psms and len(records) >= config.max_psms: break
        if config.max_psms and len(records) >= config.max_psms: break
    if not records: raise ValueError("No usable annotated PSMs were found")
    torch.save(records, output)
    report = {"records": len(records), "skipped": skipped, "ion_types": ION_TYPES, "unknown_modifications_encoded_as_zero": sorted(unknown_mods)}
    Path(output).with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report

class SpectrumDataset(Dataset):
    def __init__(self, records): self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, i): return self.records[i]


def collate(batch):
    length = max(len(x["sequence"]) for x in batch)
    b, c = len(batch), len(ION_TYPES)
    aa = torch.zeros(b, length, dtype=torch.long); mods = torch.zeros(b, length); mask = torch.ones(b, length, dtype=torch.bool)
    charge = torch.zeros(b, dtype=torch.long); target = torch.zeros(b, length - 1, c); cut_mask = torch.zeros(b, length - 1, dtype=torch.bool)
    for i, r in enumerate(batch):
        n = len(r["sequence"]); aa[i,:n] = torch.tensor([AA_TO_ID[a] for a in r["sequence"]]); mods[i,:n] = torch.as_tensor(r["mod_mass"])
        mask[i,:n] = False; charge[i] = min(max(int(r["charge"]), 1), 8); target[i,:n-1] = torch.as_tensor(r["target"]) if "target" in r else 0; cut_mask[i,:n-1] = True
    return aa, mods, charge, mask, target, cut_mask

class SpectrumTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__(); self.cfg = cfg
        self.aa = nn.Embedding(len(AA_TO_ID)+1, cfg.d_model, padding_idx=0); self.mod = nn.Linear(1, cfg.d_model, bias=False); self.charge = nn.Embedding(9, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, cfg.max_len, cfg.d_model)*0.02)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.nhead, cfg.d_model*4, cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers); self.head = nn.Sequential(nn.Linear(cfg.d_model*2, cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, len(ION_TYPES)), nn.Softplus())
    def forward(self, aa, mods, charge, padding_mask):
        n = aa.shape[1]; x = self.aa(aa)+self.mod(mods.unsqueeze(-1))+self.charge(charge)[:,None,:]+self.pos[:,:n]
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.head(torch.cat([x[:,:-1], x[:,1:]], dim=-1))

def spectral_angle(pred, target, mask):
    p = pred[mask]; t = target[mask]
    cos = torch.nn.functional.cosine_similarity(p, t, dim=-1).clamp(-1,1)
    return 1 - 2*torch.acos(cos)/math.pi

def save_checkpoint(path, model, cfg, extra=None):
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "ion_types": ION_TYPES, "extra": extra or {}}, path)

def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False); cfg = Config(**ck["config"]); model = SpectrumTransformer(cfg).to(device); model.load_state_dict(ck["model"]); model.eval(); return model, cfg, ck

def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

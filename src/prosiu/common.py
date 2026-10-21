"""Shared paths, fixed assay identifiers, and serialization."""

import json
from pathlib import Path

import torch

ASSETS = Path(__file__).resolve().parent / "assets"
DEVICE = "cuda:0"
ASSAYS = ["amacGFP", "cgreGFP", "avGFP"]
SEEDS = [0, 1, 2, 3, 4]
METRICS = ["CA", "RA", "TA", "PA"]
IDENTIFIERS = dict(
    zip(
        ASSAYS,
        [
            "Q8WTC7_9CNID_Somermeyer_2022",
            "D7PM05_CLYGR_Somermeyer_2022",
            "GFP_AEQVI_Sarkisyan_2016",
        ],
    )
)
ENCODER = "facebook/esm2_t12_35M_UR50D"
LANGUAGE_ENCODER = "facebook/esm2_t33_650M_UR50D"


def rng(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def configuration(assay):
    return read_json(ASSETS / "configs" / f"{assay}.json")

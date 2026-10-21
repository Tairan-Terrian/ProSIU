"""Published fluorescence contrasts and fixed evidence-withdrawal requests."""

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import zipfile

import pandas as pd
import torch

from .common import DEVICE, IDENTIFIERS, SEEDS, write_json


def genotype(sequence, wt):
    return tuple((i, aa) for i, (aa, before) in enumerate(zip(sequence, wt)) if aa != before)


def sequence_of(g, wt):
    result = list(wt)
    for position, aa in g:
        result[position] = aa
    return "".join(result)


def prepare_data(data_dir, assays):
    data_dir = Path(data_dir)
    original = pd.read_csv(data_dir / "raw/GFP_original_aminoacid.csv")
    av = pd.read_csv(data_dir / "raw/avGFP_original.csv")
    for assay in assays:
        with zipfile.ZipFile(data_dir / "raw/DMS_ProteinGym_substitutions.zip") as archive:
            name = next(x for x in archive.namelist() if x.endswith(IDENTIFIERS[assay] + ".csv"))
            frame = pd.read_csv(archive.open(name))
        reference = list(frame.iloc[0].mutated_sequence)
        for mutation in frame.iloc[0].mutant.split(":"):
            reference[int(mutation[1:-1]) - 1] = mutation[0]
        wt = "".join(reference)
        convert = float if assay == "avGFP" else math.log10
        measure = {
            genotype(r.mutated_sequence, wt): convert(r.DMS_score) for r in frame.itertuples()
        }
        if assay == "avGFP":
            measure[()] = float(av.loc[av.aa_genotype_native == "wt", "log_brightness"].iloc[0])
        else:
            measure[()] = math.log10(
                float(
                    original.loc[
                        (original.gene == assay) & (original.aa_genotype_native == "wt"),
                        "replicates_mean_brightness",
                    ].iloc[0]
                )
            )
        targets = [
            g for g in sorted(measure) if len(g) == 2 and (g[0],) in measure and (g[1],) in measure
        ]
        order = list(range(len(targets)))
        random.Random(650017).shuffle(order)
        a, b, c = int(0.4 * len(order)), int(0.6 * len(order)), int(0.8 * len(order))
        parts = {
            "train": order[:a][:4096],
            "dev_train": order[a:b][:2048],
            "dev": order[b:c][:1024],
            "test": order[c:][:2048],
        }
        rows = []
        for partition, indices in parts.items():
            for index in indices:
                g = targets[index]
                four = [(), (g[0],), (g[1],), g]
                rows.append(
                    {
                        "q": [sequence_of(v, wt) for v in four],
                        "y": [measure[v] for v in four],
                        "positions": [v[0] for v in g],
                        "depth": 2,
                        "partition": partition,
                    }
                )
        anchors = [
            {"sequence": sequence_of(g, wt), "y": measure[g]}
            for g in sorted(measure)
            if len(g) <= 1
        ]
        source = {
            "assay": assay,
            "identifier": IDENTIFIERS[assay],
            "wt": wt,
            "rows": rows,
            "anchors": anchors,
            "anchor_ids": list(range(len(anchors))),
            "eligible_targets": len(targets),
            "split_seed": 650017,
            "counts": {p: len(v) for p, v in parts.items()},
            "measured_wt": measure[()],
        }
        write_json(data_dir / "processed" / f"{assay}.json", source)
        write_json(
            data_dir / "processed" / f"{assay}_requests.json", [request(rows, s) for s in SEEDS]
        )
        print(
            json.dumps(
                {
                    "assay": assay,
                    "eligible": len(targets),
                    "anchors": len(anchors),
                    "counts": source["counts"],
                    "WT": measure[()],
                }
            ),
            flush=True,
        )


def request(rows, seed, partition="train"):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row["partition"] == partition:
            groups[tuple(row["positions"])].append(i)
    randomizer = random.Random(650100 + seed)
    keys = sorted(groups)
    randomizer.shuffle(keys)
    count = sum(len(v) for v in groups.values())
    selected, forgotten, shifts = [], [], []
    for key in keys:
        selected.append(key)
        direction = randomizer.choice([-1.0, 1.0])
        forgotten.extend(groups[key])
        shifts.extend([direction] * len(groups[key]))
        if len(forgotten) >= math.ceil(0.2 * count):
            break
    training = [i for v in groups.values() for i in v]
    retained = sorted(set(training) - set(forgotten))
    focused = set(p for key in selected for p in key)
    adjacent = [i for i, r in enumerate(rows) if any(p in focused for p in r["positions"])]
    return {
        "seed": seed,
        "partition": partition,
        "forgotten": forgotten,
        "retained": retained,
        "shifts": shifts,
        "selected_position_pairs": selected,
        "adjacent": adjacent,
        "offset_scale": "one valid-anchor standard deviation",
        "affected_fraction": len(forgotten) / count,
    }


def load(assay, data_dir):
    source = torch.load(
        Path(data_dir) / "features" / f"{assay}.pt", map_location=DEVICE, weights_only=False
    )
    truth = (source["y"][:, 3] - source["anchor"]) / source["scale"]
    data = {k: source[k] for k in ["x", "scale", "positions", "anchor", "partitions"]}
    data["y"] = source["y"][:, :3]
    data["offset"] = (source["anchor"] - source["y"][:, 0]) / source["scale"]
    return data, truth


def observed(data, truth, request):
    training = torch.tensor(sorted(request["retained"] + request["forgotten"]), device=DEVICE)
    forgotten = torch.tensor(request["forgotten"], device=DEVICE)
    values = torch.zeros_like(truth)
    values[training] = truth[training]
    values[forgotten] += torch.tensor(request["shifts"], device=DEVICE)
    return dict(data, eps=values)

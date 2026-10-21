"""Frozen protein representations for Epistatic Program Selection."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, EsmForMaskedLM

from .common import DEVICE, ENCODER, LANGUAGE_ENCODER, read_json, rng


@torch.inference_mode()
def build_language_features(data_dir, model_name=LANGUAGE_ENCODER, tokenizer_name=ENCODER):
    """Encode amacGFP sequences for the selected language-feature preference head."""
    data_dir = Path(data_dir)
    source = read_json(data_dir / "processed/amacGFP.json")
    anchors = [source["anchors"][i] for i in source["anchor_ids"]]
    sequences = [a["sequence"] for a in anchors] + [r["q"][3] for r in source["rows"]]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = EsmForMaskedLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(DEVICE).eval()
    embeddings = []
    for start in range(0, len(sequences), 16):
        tokens = tokenizer(sequences[start : start + 16], return_tensors="pt", padding=True).to(
            DEVICE
        )
        embeddings.append(model.esm(**tokens).last_hidden_state[:, 1:-1].float().mean(1))
        if start % 1024 == 0:
            print(
                json.dumps(
                    {"stage": "language_features", "encoded": start, "total": len(sequences)}
                ),
                flush=True,
            )
    torch.save(
        {"embeddings": torch.cat(embeddings).cpu()}, data_dir / "features/amacGFP_language.pt"
    )


@torch.inference_mode()
def build_context_features(data_dir, assays, model_name=ENCODER):
    data_dir = Path(data_dir)
    (data_dir / "features").mkdir(parents=True, exist_ok=True)
    partitions = ["train", "dev_train", "dev", "test"]
    feature_seed = 650027
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = (
        AutoModel.from_pretrained(model_name, add_pooling_layer=False, torch_dtype=torch.bfloat16)
        .to(DEVICE)
        .eval()
    )
    for assay in assays:
        source = json.loads((data_dir / "processed" / f"{assay}.json").read_text())
        rows = source["rows"]
        backgrounds = sorted({r["q"][0] for r in rows})
        lookup = {s: i for i, s in enumerate(backgrounds)}
        contexts = []
        for start in range(0, len(backgrounds), 32):
            inputs = tokenizer(
                backgrounds[start : start + 32], return_tensors="pt", padding=True
            ).to(DEVICE)
            contexts.append(model(**inputs).last_hidden_state[:, 1 : len(source["wt"]) + 1].cpu())
            if start % 1024 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "ESM_context",
                            "assay": assay,
                            "encoded": start,
                            "total": len(backgrounds),
                        }
                    ),
                    flush=True,
                )
        contexts = torch.cat(contexts).to(DEVICE)
        ids = torch.tensor([lookup[r["q"][0]] for r in rows], device=DEVICE)
        positions = torch.tensor([r["positions"] for r in rows], device=DEVICE)
        emb = model.get_input_embeddings().weight.float()
        raw_parts = []
        for j in range(2):
            before = tokenizer.convert_tokens_to_ids([r["q"][0][r["positions"][j]] for r in rows])
            after = tokenizer.convert_tokens_to_ids([r["q"][3][r["positions"][j]] for r in rows])
            delta = (
                emb[torch.tensor(after, device=DEVICE)] - emb[torch.tensor(before, device=DEVICE)]
            )
            raw_parts.append(torch.cat([contexts[ids, positions[:, j]].float(), delta], 1))
        rng = torch.Generator(device=DEVICE).manual_seed(feature_seed)
        projection = (
            torch.randn(raw_parts[0].shape[1], 48, generator=rng, device=DEVICE)
            / raw_parts[0].shape[1] ** 0.5
        )
        parts = [raw @ projection for raw in raw_parts]
        y = torch.tensor([r["y"] for r in rows], dtype=torch.float64, device=DEVICE)
        anchor_y = torch.tensor([r["y"] for r in source["anchors"]], device=DEVICE)
        scale = anchor_y.std().double()
        d = (y[:, 1:3] - y[:, :1]) / scale
        numeric = torch.stack(
            [
                d.sum(1),
                d.min(1).values,
                d.max(1).values,
                d.prod(1),
                d.square().sum(1),
                (d[:, 0] - d[:, 1]).abs(),
                (positions[:, 0] - positions[:, 1]).abs() / len(source["wt"]),
            ],
            1,
        ).float()
        raw = torch.cat(
            [parts[0] + parts[1], (parts[0] - parts[1]).abs(), parts[0] * parts[1], numeric], 1
        )
        train = torch.tensor(
            [i for i, r in enumerate(rows) if r["partition"] == "train"], device=DEVICE
        )
        varying = raw[train].std(0) > 1e-6
        raw = raw[:, varying]
        standardized = (raw - raw[train].mean(0)) / raw[train].std(0)
        # Unsupervised landmarks and moments are fixed features, independent of the deletion labels.
        centers = train[torch.randperm(len(train), generator=rng, device=DEVICE)[:256]]
        distances = (
            torch.cdist(standardized, standardized[centers]).square() / standardized.shape[1]
        )
        kernel = torch.exp(-distances / 2)
        basis = torch.linalg.cholesky(
            kernel[centers] + 0.001 * torch.eye(len(centers), device=DEVICE)
        )
        nystrom = torch.linalg.solve_triangular(basis, kernel.T, upper=False).T
        num_mean, num_std = numeric[train].mean(0), numeric[train].std(0)
        x = torch.cat(
            [torch.ones(len(rows), 1, device=DEVICE), (numeric - num_mean) / num_std, nystrom], 1
        ).double()
        output = {
            "x": x.cpu(),
            "y": y.cpu(),
            "scale": scale.cpu(),
            "positions": positions.cpu(),
            "anchor": (y[:, 1] + y[:, 2] - y[:, 0]).cpu(),
            "partitions": {
                p: torch.tensor([i for i, r in enumerate(rows) if r["partition"] == p])
                for p in partitions
            },
            "encoder": ENCODER,
            "feature_definition": "conditional ESM residue context, measured constituent effects, fixed Nystrom map",
        }
        torch.save(output, data_dir / "features" / f"{assay}.pt")
        print(
            json.dumps({"stage": "context_features", "assay": assay, "shape": list(x.shape)}),
            flush=True,
        )


def build_selector_features(assay, data, data_dir):
    """Anchor-derived maps and explicit identities use sequence and valid singles."""
    source = json.loads((Path(data_dir) / "processed" / f"{assay}.json").read_text())
    rows = source["rows"]
    train = data["partitions"]["dev_train"]
    effects = ((data["y"][:, 1:3] - data["y"][:, :1]) / data["scale"]).sort(1).values
    effect_map = torch.asinh(effects / 0.1)
    generator = rng(651007)
    centers = train[torch.randperm(len(train), generator=generator, device=DEVICE)[:192]]
    distance = torch.cdist(effect_map, effect_map[centers]).square()
    kernels = []
    for width in [0.2, 0.7, 2.0]:
        kernel = torch.exp(-distance / (2 * width**2))
        factor = torch.linalg.cholesky(
            kernel[centers] + 0.001 * torch.eye(len(centers), device=DEVICE)
        )
        kernels.append(torch.linalg.solve_triangular(factor, kernel.T, upper=False).T)
    numeric = torch.cat([data["x"][:, :8], effect_map, effect_map.prod(1, keepdim=True)], 1)
    anchor_map = torch.cat([numeric] + kernels, 1)
    mutations = sorted({(p, r["q"][3][p]) for r in rows for p in r["positions"]})
    lookup = {key: i for i, key in enumerate(mutations)}
    identities = torch.tensor(
        [[lookup[p, r["q"][3][p]] for p in r["positions"]] for r in rows], device=DEVICE
    )
    mutation = F.one_hot(identities, len(lookup)).sum(1).double()
    position = F.one_hot(data["positions"], len(source["wt"])).sum(1).double()
    esm = data["x"][:, 8:]
    result = {
        "identity": torch.cat([anchor_map, mutation, position], 1),
        "context": torch.cat([anchor_map, mutation, position, esm], 1),
    }
    if assay == "amacGFP":
        assets = torch.load(
            Path(data_dir) / "features" / f"{assay}_language.pt",
            map_location=DEVICE,
            weights_only=False,
        )
        anchors = [source["anchors"][i] for i in source["anchor_ids"]]
        anchor_lookup = {a["sequence"]: i for i, a in enumerate(anchors)}
        quad = torch.tensor(
            [[anchor_lookup[r["q"][j]] for j in range(3)] for r in rows], device=DEVICE
        )
        embedding = assets["embeddings"].double()
        joint = embedding[len(anchors) :]
        contrast = joint - embedding[quad[:, 1]] - embedding[quad[:, 2]] + embedding[quad[:, 0]]
        maps = []
        for values in [joint, contrast]:
            centered = values - values[train].mean(0)
            _, singular, vectors = torch.linalg.svd(centered[train], full_matrices=False)
            projected = centered @ vectors[:128].T
            maps.append(projected / (singular[:128] / len(train) ** 0.5 + 0.01))
        language = torch.cat(maps, 1)
        result["language"] = torch.cat([anchor_map, mutation, position, language / 4], 1)
    return result

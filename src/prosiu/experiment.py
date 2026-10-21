"""Train ProSIU, revoke dependent evidence, and evaluate fixed requests."""

from datetime import datetime, timezone
import platform
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import transformers

from .common import DEVICE, configuration, read_json, write_json
from .data import load, observed
from .epistatic_program_selection import expand_pairs, program_pairs
from .features import build_selector_features
from .metrics import evaluate_scores
from .mutation_preference_optimization import SelectorObjective
from .selective_epistasis_forgetting import selective_forgetting


def fit_episode(stream, request, representations, config):
    """Fit both heads using the observed training stream and its retained subset."""
    retained = torch.tensor(request["retained"], device=DEVICE)
    training = torch.tensor(sorted(request["retained"] + request["forgotten"]), device=DEVICE)
    seed = request["seed"]
    old_pairs = expand_pairs(program_pairs(stream, training, training, seed), training, seed)
    new_pairs = expand_pairs(program_pairs(stream, retained, retained, seed), retained, seed)
    predictions, parameters, diagnostics = {}, {}, {}
    for head, setting in config.items():
        x = representations[setting["features"]]
        original_objective = SelectorObjective(x, stream["eps"], training, old_pairs, setting)
        retained_objective = SelectorObjective(x, stream["eps"], retained, new_pairs, setting)
        original, original_fit = original_objective.fit(iterations=600)
        theta, details = selective_forgetting(x, request, retained_objective, original)
        reference, reference_fit = retained_objective.fit(iterations=600)
        predictions[head] = x @ theta - setting.get("threshold", 0.0)
        parameters[head] = {
            "theta": theta.cpu(),
            "original": original.cpu(),
            "reference": reference.cpu(),
        }
        diagnostics[head] = {
            "original_fit": original_fit,
            "update": details,
            "reference_fit": reference_fit,
            "parameter_distance": (theta - reference).norm().item(),
            "objective_gap": (
                retained_objective.loss(theta) - retained_objective.loss(reference)
            ).item(),
            "prediction_max_distance": (x @ (theta - reference)).abs().max().item(),
            "feature_dimension": x.shape[1],
        }
    return predictions, parameters, diagnostics


def run(data_dir, output_dir, assays, seeds):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    for assay in assays:
        data, truth = load(assay, data_dir)
        representations = build_selector_features(assay, data, data_dir)
        config = configuration(assay)
        requests = read_json(data_dir / "processed" / f"{assay}_requests.json")
        for seed in seeds:
            started = time.perf_counter()
            request = requests[seed]
            stream = observed(data, truth, request)
            predictions, parameters, diagnostics = fit_episode(
                stream, request, representations, config
            )
            retained = torch.tensor(request["retained"], device=DEVICE)
            records = program_pairs(stream, data["partitions"]["test"], retained, 10000 + seed)
            metrics, tensors = evaluate_scores(
                predictions["classification"],
                truth,
                request,
                data["partitions"]["test"],
                records,
                utility=predictions["preference"],
            )
            tensors["utility"] = predictions["preference"].cpu()
            directory = output_dir / assay / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(parameters, directory / "checkpoint.pt")
            torch.save(tensors, directory / "predictions.pt")
            result = {
                "method": "ProSIU",
                "assay": assay,
                "seed": seed,
                "metrics": metrics,
                "configuration": config,
                "diagnostics": diagnostics,
                "seconds": time.perf_counter() - started,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                    "transformers": transformers.__version__,
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                },
            }
            write_json(directory / "result.json", result)
            print({"assay": assay, "seed": seed, "metrics": metrics}, flush=True)


def evaluate(data_dir, output_dir, assays, seeds):
    """Reconstruct metrics from saved predictions and published clean targets."""
    rows = []
    for assay in assays:
        data, truth = load(assay, data_dir)
        requests = read_json(Path(data_dir) / "processed" / f"{assay}_requests.json")
        for seed in seeds:
            directory = Path(output_dir) / assay / f"seed_{seed}"
            tensors = torch.load(
                directory / "predictions.pt", map_location=DEVICE, weights_only=True
            )
            records = {"first": tensors["program_first"], "second": tensors["program_second"]}
            metrics, _ = evaluate_scores(
                tensors["interaction"],
                truth,
                requests[seed],
                data["partitions"]["test"],
                records,
                utility=tensors["utility"],
            )
            reported = read_json(directory / "result.json")["metrics"]
            error = max(abs(metrics[key] - reported[key]) for key in metrics)
            assert error < 1e-10
            rows.append(
                {"assay": assay, "seed": seed, "metrics": metrics, "reconstruction_error": error}
            )
    write_json(Path(output_dir) / "evaluation.json", {"runs": rows})
    print(
        {"evaluated_runs": len(rows), "maximum_error": max(r["reconstruction_error"] for r in rows)}
    )

"""Summarize the main experiment across all fixed seeds."""

import csv
from pathlib import Path
import statistics

from .common import ASSAYS, METRICS, SEEDS, read_json, write_json


def summarize(output_dir):
    output_dir = Path(output_dir)
    rows = []
    lines = ["| Dataset | CA ↑ | RA ↑ | TA ↑ | PA ↑ |", "|---|---:|---:|---:|---:|"]
    for assay in ASSAYS:
        runs = [read_json(output_dir / assay / f"seed_{seed}/result.json") for seed in SEEDS]
        metrics = {
            key: {
                "mean": statistics.mean(r["metrics"][key] for r in runs),
                "std": statistics.stdev(r["metrics"][key] for r in runs),
            }
            for key in METRICS
        }
        rows.append({"assay": assay, "seeds": SEEDS, "metrics": metrics})
        cells = [f"{metrics[k]['mean']:.2f}±{metrics[k]['std']:.2f}" for k in METRICS]
        lines.append("| " + " | ".join([assay] + cells) + " |")
    write_json(output_dir / "summary.json", {"method": "ProSIU", "runs": 15, "summaries": rows})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["assay"] + [f"{k}_{stat}" for k in METRICS for stat in ["mean", "std"]])
        for row in rows:
            writer.writerow(
                [row["assay"]]
                + [f"{row['metrics'][k][s]:.3f}" for k in METRICS for s in ["mean", "std"]]
            )
    (output_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

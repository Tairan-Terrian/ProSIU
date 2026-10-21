"""Command-line entry point for the released ProSIU pipeline."""

import argparse
from pathlib import Path

import torch

from .common import ASSAYS, ENCODER, LANGUAGE_ENCODER, SEEDS


def main():
    parser = argparse.ArgumentParser(description="ProSIU reproducible GFP evidence unlearning")
    parser.add_argument(
        "stage",
        choices=["download", "prepare", "features", "run", "evaluate", "summarize"],
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--assays", nargs="+", choices=ASSAYS, default=ASSAYS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument(
        "--encoder", default=ENCODER, help="Public ESM-2 35M ID or local model directory"
    )
    parser.add_argument(
        "--language-encoder",
        default=LANGUAGE_ENCODER,
        help="Public ESM-2 650M ID or local model directory",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == "download":
        from .download import download

        download(args.data_dir)
    elif args.stage == "prepare":
        from .data import prepare_data

        prepare_data(args.data_dir, args.assays)
    elif args.stage == "features":
        from .features import build_context_features, build_language_features

        build_context_features(args.data_dir, args.assays, args.encoder)
        if "amacGFP" in args.assays:
            build_language_features(args.data_dir, args.language_encoder, args.encoder)
    elif args.stage == "run":
        from .experiment import run

        run(args.data_dir, args.output_dir, args.assays, args.seeds)
    elif args.stage == "evaluate":
        from .experiment import evaluate

        evaluate(args.data_dir, args.output_dir, args.assays, args.seeds)
    elif args.stage == "summarize":
        from .report import summarize

        summarize(args.output_dir)

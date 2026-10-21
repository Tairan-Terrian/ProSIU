#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python -m prosiu download &&
python -m prosiu prepare &&
python -m prosiu features &&
python -m prosiu run &&
python -m prosiu evaluate &&
python -m prosiu summarize

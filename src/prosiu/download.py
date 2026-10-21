"""Download the fixed ProteinGym release and the published GFP source tables."""

from pathlib import Path
from urllib.request import urlretrieve

from .common import ASSETS, read_json


def download(data_dir):
    raw = Path(data_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, source in read_json(ASSETS / "sources.json")["downloads"].items():
        print(f"Downloading {name}", flush=True)
        urlretrieve(source, raw / name)

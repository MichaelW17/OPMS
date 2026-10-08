"""Expand compact endpoint archives into the layout used by simulation scripts."""

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def unpack(output):
    data = ROOT / "data/multistart"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("configuration.json", "candidate_summary.csv"):
        (output / name).write_bytes((data / name).read_bytes())
    for link in "ABCD":
        for kind in ("gsm", "opms"):
            folder = output / link / kind
            folder.mkdir(parents=True, exist_ok=True)
            endpoints = torch.load(
                data / f"{link}_{kind}_endpoints.pt",
                map_location="cpu",
                weights_only=False,
            )
            records = json.loads((data / f"{link}_{kind}_metrics.json").read_text())
            for index, (endpoint, record) in enumerate(zip(endpoints, records)):
                path = folder / f"run_{index:03d}.pt"
                if not path.exists():
                    torch.save(endpoint, path)
                (folder / f"run_{index:03d}_screening.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "work/endpoints")
    args = parser.parse_args()
    print(unpack(args.output))

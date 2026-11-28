"""Safely extract the provided UCF101 subset archive."""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def extract_dataset(archive: str | Path = "data/raw/UCF101_subset.tar", output_dir: str | Path = "data/processed") -> Path:
    archive, output_dir = Path(archive), Path(output_dir)
    if not archive.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        root = output_dir.resolve()
        members = tar.getmembers()
        for member in members:
            target = (output_dir / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.name}")
        for member in members:
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsupported archive member: {member.name}")
        tar.extractall(output_dir, members=members)
    extracted = output_dir / "UCF101_subset"
    if not extracted.is_dir():
        raise RuntimeError(f"Expected extracted dataset at {extracted}")
    print(f"Dataset extracted to {extracted}")
    return extracted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="data/raw/UCF101_subset.tar")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    extract_dataset(args.archive, args.output_dir)

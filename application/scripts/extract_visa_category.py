"""Extract one VisA category from the official archive without unpacking other categories."""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def extract_category(archive: Path, destination: Path, category: str) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, mode="r|") as source:
        for member in source:
            parts = Path(member.name).parts
            if not parts or parts[0] != category:
                if count and parts and parts[0] != category:
                    break
                continue
            if member.isdir():
                continue
            if not member.isfile() or ".." in parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe archive path: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = source.extractfile(member)
            if payload is None:
                raise ValueError(f"Missing member bytes: {member.name}")
            with target.open("wb") as output:
                while chunk := payload.read(1024 * 1024):
                    output.write(chunk)
            count += 1
    if not count:
        raise ValueError(f"Category {category} was absent or incomplete")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--category", default="capsules")
    args = parser.parse_args()
    print({"category": args.category, "files": extract_category(args.archive, args.destination, args.category)})

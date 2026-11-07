"""Create a portable private release without local accounts, captures, or source photos."""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT.parent / "ForgeCareAI-Release-PyCharm-Python-3.14.7.zip"
SKIP_DIRS = {".git", ".pytest_cache", ".venv", "__pycache__", "data"}
SKIP_NAMES = {".phone-federation-data-path", ".env", ".env.local", "first-login.txt"}
SOURCE_PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in SKIP_DIRS for part in relative.parts):
        return False
    if path.name in SKIP_NAMES or path.suffix in {".pyc", ".sqlite3", ".db"}:
        return False
    if path.name.endswith((".log", ".pid")):
        return False
    if relative.parts[0] == "research" and path.suffix.lower() in SOURCE_PHOTO_EXTENSIONS:
        return False
    return True


def main() -> None:
    destination = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT
    if destination.is_relative_to(ROOT):
        raise SystemExit("ZIP output must be outside the release folder")
    paths = sorted(path for path in ROOT.rglob("*") if path.is_file() and included(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for path in paths:
            archive.write(path, (Path(ROOT.name) / path.relative_to(ROOT)).as_posix())
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f"ZIP: {destination}")
    print(f"Files: {len(paths)}")
    print(f"Bytes: {destination.stat().st_size}")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()

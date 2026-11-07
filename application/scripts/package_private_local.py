"""Build a private Windows transfer ZIP with current data, datasets, and keys.

Secrets and a replacement admin password are written only into the ZIP, not to
the source checkout. Do not publish this archive.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import secrets
import sqlite3
import sys
import winreg
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = Path(ROOT.name)
DATA = Path(os.environ.get("FORGECARE_DATA") or Path(os.environ["LOCALAPPDATA"]) / "ForgeCareAI" / "data")
DATASETS = ROOT.parents[1] / "work" / "datasets"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
TORCH_CACHE = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
WHEELHOUSE = ROOT.parents[1] / "work" / "wheelhouse-3147"
PYTHON_INSTALLER = ROOT.parents[1] / "work" / "python-3.14.7-amd64.exe"
PYTHON_INSTALLER_SHA256 = "9d9eb2709ef81bf5cd30db3c2096bdbc4ea10087c22e62f27d356b36f6ae9649"
DEFAULT_OUTPUT = ROOT.parent / "ForgeCareAI-Private-Local-Complete-Python-3.14.7.zip"
SKIP_DIRS = {".git", ".pytest_cache", ".venv", "__pycache__", "private_data", "datasets", "model_cache"}
SKIP_NAMES = {".phone-federation-data-path", ".private-setup-complete", ".env", ".env.local",
              "private_config.json", "PRIVATE_LOGIN.txt"}


def user_key(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        try:
            value, _ = winreg.QueryValueEx(key, name)
        except FileNotFoundError as exc:
            raise RuntimeError(f"{name} is missing from Windows user environment") from exc
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{name} is empty in Windows user environment")
    return value.strip()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def database_snapshot(path: Path) -> sqlite3.Connection:
    source = sqlite3.connect(path)
    destination = sqlite3.connect(":memory:")
    try:
        source.backup(destination)
    finally:
        source.close()
    return destination


def portable_capture_path(original: str) -> str:
    path = Path(original)
    if not path.is_file():
        raise RuntimeError(f"Capture image is missing: {path.name}")
    parts = path.parts
    positions = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not positions:
        raise RuntimeError(f"Capture path is outside images: {path.name}")
    relative = Path(*parts[positions[-1]:])
    candidate = DATA / relative
    if not candidate.is_file():
        raise RuntimeError(f"Capture image is not in local data: {relative}")
    return (Path("private_data") / relative).as_posix()


def portable_research_value(value: object) -> object:
    if isinstance(value, str):
        try:
            relative = Path(value).relative_to(DATASETS)
        except (ValueError, OSError):
            return value
        return (Path("datasets") / relative).as_posix()
    if isinstance(value, list):
        return [portable_research_value(item) for item in value]
    if isinstance(value, dict):
        return {key: portable_research_value(item) for key, item in value.items()}
    return value


def project_file_bytes(path: Path) -> bytes | None:
    if path.suffix.lower() == ".csv" and "research" in path.relative_to(ROOT).parts:
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            fields = reader.fieldnames
            if not fields:
                return None
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=fields)
            writer.writeheader()
            writer.writerows(portable_research_value(row) for row in reader)
            return buffer.getvalue().encode("utf-8")
    if path.suffix.lower() == ".json" and "research" in path.relative_to(ROOT).parts:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return None
        return json.dumps(portable_research_value(value), ensure_ascii=False, indent=2).encode("utf-8")
    return None


def included_project_file(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (not any(part in SKIP_DIRS for part in relative.parts)
            and path.name not in SKIP_NAMES
            and path.suffix.lower() not in {".pyc", ".sqlite3", ".db", ".pid", ".log"})


def main() -> None:
    output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT
    if output.is_relative_to(ROOT) or output.is_relative_to(DATASETS) or output.is_relative_to(DATA):
        raise SystemExit("Choose a ZIP path outside the source, data, and dataset folders")
    if not DATASETS.is_dir() or not (DATA / "forgecare.sqlite3").is_file():
        raise SystemExit("Current dataset or application data folder is missing")
    if not WHEELHOUSE.is_dir() or not any(WHEELHOUSE.glob("*.whl")):
        raise SystemExit("The Python 3.14.7 Windows wheelhouse is missing")
    if not PYTHON_INSTALLER.is_file() or digest_file(PYTHON_INSTALLER) != PYTHON_INSTALLER_SHA256:
        raise SystemExit("The verified Python 3.14.7 Windows installer is missing or changed")
    keys = {name: user_key(name) for name in ("GROQ_API_KEY", "ROBOFLOW_API_KEY")}
    admin_password = secrets.token_urlsafe(24)
    db = database_snapshot(DATA / "forgecare.sqlite3")
    try:
        captures = db.execute("SELECT id,image_path FROM captures").fetchall()
        for capture_id, image_path in captures:
            db.execute("UPDATE captures SET image_path=? WHERE id=?",
                       (portable_capture_path(image_path), capture_id))
        for model_id, artifact_path in db.execute("SELECT id,artifact_path FROM models WHERE artifact_path IS NOT NULL"):
            portable = portable_research_value(artifact_path)
            if portable == artifact_path:
                try:
                    portable = (Path("private_data") / Path(artifact_path).relative_to(DATA)).as_posix()
                except ValueError:
                    pass
            db.execute("UPDATE models SET artifact_path=? WHERE id=?", (portable, model_id))
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", admin_password.encode(), salt, 300_000)
        db.execute("UPDATE users SET password_hash=?,active=1 WHERE username='admin'",
                   (f"{salt.hex()}:{digest.hex()}",))
        if db.execute("SELECT changes()").fetchone()[0] != 1:
            raise RuntimeError("The copied admin account could not be prepared")
        db.execute("DELETE FROM sessions")
        db.execute("UPDATE cameras SET stopped_at=datetime('now'),last_frame_at=NULL,last_error='Worker stopped',last_error_at=datetime('now')")
        db.commit()
        operations_bytes = db.serialize()
    finally:
        db.close()
    knowledge = database_snapshot(DATA / "knowledge.sqlite3")
    try:
        knowledge_bytes = knowledge.serialize()
    finally:
        knowledge.close()

    project_paths = sorted(path for path in ROOT.rglob("*") if path.is_file() and included_project_file(path))
    data_paths = sorted(path for path in DATA.rglob("*") if path.is_file()
                        and path.name not in {"forgecare.sqlite3", "knowledge.sqlite3"}
                        and not path.name.endswith((".pid", ".log"))
                        and "logs" not in path.relative_to(DATA).parts)
    dataset_paths = sorted(path for path in DATASETS.rglob("*") if path.is_file())
    model_cache_paths = sorted(path for path in HF_CACHE.rglob("*") if path.is_file()
                               and ".locks" not in path.relative_to(HF_CACHE).parts) if HF_CACHE.is_dir() else []
    torch_cache_paths = sorted(path for path in TORCH_CACHE.rglob("*") if path.is_file()) if TORCH_CACHE.is_dir() else []
    wheel_paths = sorted(WHEELHOUSE.glob("*.whl"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        for path in project_paths:
            payload = project_file_bytes(path)
            name = (ARCHIVE_ROOT / path.relative_to(ROOT)).as_posix()
            if payload is not None:
                archive.writestr(name, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
            else:
                archive.write(path, name, compress_type=zipfile.ZIP_STORED
                              if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".pt", ".ckpt", ".zip"}
                              else zipfile.ZIP_DEFLATED, compresslevel=6)
        archive.writestr((ARCHIVE_ROOT / "private_data/forgecare.sqlite3").as_posix(), operations_bytes)
        archive.writestr((ARCHIVE_ROOT / "private_data/knowledge.sqlite3").as_posix(), knowledge_bytes)
        for path in data_paths:
            archive.write(path, (ARCHIVE_ROOT / "private_data" / path.relative_to(DATA)).as_posix(),
                          compress_type=zipfile.ZIP_STORED if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".pt"}
                          else zipfile.ZIP_DEFLATED, compresslevel=6)
        for path in dataset_paths:
            archive.write(path, (ARCHIVE_ROOT / "datasets" / path.relative_to(DATASETS)).as_posix(),
                          compress_type=zipfile.ZIP_STORED if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".zip", ".pt"}
                          else zipfile.ZIP_DEFLATED, compresslevel=6)
        for path in model_cache_paths:
            archive.write(path, (ARCHIVE_ROOT / "model_cache/hub" / path.relative_to(HF_CACHE)).as_posix(),
                          compress_type=zipfile.ZIP_STORED if path.stat().st_size > 10 * 1024 * 1024
                          else zipfile.ZIP_DEFLATED, compresslevel=6)
        for path in torch_cache_paths:
            archive.write(path, (ARCHIVE_ROOT / "model_cache/torch/hub/checkpoints" / path.relative_to(TORCH_CACHE)).as_posix(),
                          compress_type=zipfile.ZIP_STORED)
        archive.write(PYTHON_INSTALLER, (ARCHIVE_ROOT / "setup/python-3.14.7-amd64.exe").as_posix(),
                      compress_type=zipfile.ZIP_STORED)
        for path in wheel_paths:
            archive.write(path, (ARCHIVE_ROOT / "setup/wheelhouse" / path.name).as_posix(),
                          compress_type=zipfile.ZIP_STORED)
        archive.writestr((ARCHIVE_ROOT / "private_config.json").as_posix(),
                         json.dumps(keys, indent=2).encode("utf-8"))
        archive.writestr((ARCHIVE_ROOT / "PRIVATE_LOGIN.txt").as_posix(),
                         f"ForgeCare private local bundle\nUsername: admin\nPassword: {admin_password}\n"
                         .encode("utf-8"))
        archive.writestr((ARCHIVE_ROOT / "bundle_info.json").as_posix(),
                         json.dumps({"python": "3.14.7", "captures": len(captures),
                                     "datasets": sorted(path.name for path in DATASETS.iterdir() if path.is_dir()),
                                     "project_files": len(project_paths), "data_files": len(data_paths),
                                     "dataset_files": len(dataset_paths),
                                     "model_cache_files": len(model_cache_paths) + len(torch_cache_paths),
                                     "windows_wheels": len(wheel_paths)},
                                    indent=2).encode("utf-8"))
    print(f"ZIP: {output}")
    print(f"Bytes: {output.stat().st_size}")
    print(f"Captures: {len(captures)}, dataset files: {len(dataset_paths)}, model cache files: {len(model_cache_paths) + len(torch_cache_paths)}, wheels: {len(wheel_paths)}")
    print(f"SHA256: {digest_file(output)}")


if __name__ == "__main__":
    main()

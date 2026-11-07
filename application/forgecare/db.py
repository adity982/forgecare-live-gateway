from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import DB_PATH


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connection(path: Path = DB_PATH) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=30, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    return db


@contextmanager
def connect(path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    db = connection(path)
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS sites(id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
 role TEXT NOT NULL CHECK(role IN ('technician','supervisor','reviewer','admin')),
 site_id TEXT REFERENCES sites(id), active INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), expires_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY, site_id TEXT NOT NULL REFERENCES sites(id), name TEXT NOT NULL,
 family TEXT NOT NULL, model TEXT, criticality INTEGER NOT NULL CHECK(criticality BETWEEN 1 AND 5),
 inspection_interval_days INTEGER NOT NULL DEFAULT 30, last_inspection_at TEXT, active INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS physical_samples(id TEXT NOT NULL, site_id TEXT NOT NULL REFERENCES sites(id),
 asset_id TEXT NOT NULL REFERENCES assets(id), display_name TEXT NOT NULL, created_at TEXT NOT NULL,
 active INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(site_id,id));
CREATE TABLE IF NOT EXISTS cameras(id TEXT PRIMARY KEY, site_id TEXT NOT NULL REFERENCES sites(id), asset_id TEXT NOT NULL REFERENCES assets(id),
 name TEXT NOT NULL, source TEXT NOT NULL, roi_json TEXT NOT NULL, sample_id TEXT, enabled INTEGER NOT NULL DEFAULT 1,
 last_frame_at TEXT, last_error TEXT, last_error_at TEXT, last_quality TEXT, stopped_at TEXT);
CREATE TABLE IF NOT EXISTS relay_pairs(token_hash TEXT PRIMARY KEY, camera_id TEXT NOT NULL REFERENCES cameras(id),
 created_by INTEGER NOT NULL REFERENCES users(id), expires_at TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS captures(id TEXT PRIMARY KEY, site_id TEXT NOT NULL REFERENCES sites(id), asset_id TEXT NOT NULL REFERENCES assets(id),
 camera_id TEXT NOT NULL REFERENCES cameras(id), session_id TEXT NOT NULL, sample_id TEXT NOT NULL,
 captured_at TEXT NOT NULL, image_path TEXT NOT NULL, quality TEXT NOT NULL, blur REAL, brightness REAL,
 roi_version TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS labels(capture_id TEXT PRIMARY KEY REFERENCES captures(id), grade TEXT NOT NULL,
 binary_label INTEGER, reviewer_id INTEGER NOT NULL REFERENCES users(id), reviewed_at TEXT NOT NULL, note TEXT,
 scratch_count INTEGER, dent_count INTEGER);
CREATE TABLE IF NOT EXISTS review_drafts(anchor_capture_id TEXT PRIMARY KEY REFERENCES captures(id),
 site_id TEXT NOT NULL REFERENCES sites(id), camera_id TEXT NOT NULL REFERENCES cameras(id),
 sample_id TEXT NOT NULL, source_ids_json TEXT NOT NULL, draft_text TEXT NOT NULL,
 model_id TEXT REFERENCES models(id), created_at TEXT NOT NULL, views_json TEXT NOT NULL DEFAULT '{}',
 ai_text TEXT, ai_model TEXT, ai_image_ids_json TEXT,
 ai_scratch_count INTEGER, ai_dent_count INTEGER, ai_count_basis TEXT);
CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, site_id TEXT NOT NULL REFERENCES sites(id), task TEXT NOT NULL, status TEXT NOT NULL,
 family TEXT, equipment_model TEXT, artifact_path TEXT, dataset_hash TEXT, config_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
 created_at TEXT NOT NULL, approved_by INTEGER REFERENCES users(id), approved_at TEXT);
CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY, capture_id TEXT UNIQUE NOT NULL REFERENCES captures(id),
 model_id TEXT REFERENCES models(id), predicted_class TEXT, score REAL, evidence_status TEXT NOT NULL,
 created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rules(id INTEGER PRIMARY KEY, site_id TEXT NOT NULL REFERENCES sites(id),
 criticality_weight REAL NOT NULL, repeat_weight REAL NOT NULL, overdue_weight REAL NOT NULL,
 default_minutes INTEGER NOT NULL, created_at TEXT NOT NULL, created_by INTEGER REFERENCES users(id));
CREATE TABLE IF NOT EXISTS cases(id TEXT PRIMARY KEY, site_id TEXT NOT NULL REFERENCES sites(id), asset_id TEXT NOT NULL REFERENCES assets(id),
 observation_id TEXT REFERENCES observations(id), status TEXT NOT NULL, priority REAL NOT NULL,
 priority_band TEXT NOT NULL, reason_json TEXT NOT NULL, rule_id INTEGER REFERENCES rules(id),
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL, assigned_to INTEGER REFERENCES users(id),
 override_reason TEXT, decision_note TEXT, actual_finding TEXT, actual_minutes INTEGER,
 closure_note TEXT, inspection_performed INTEGER NOT NULL DEFAULT 0,
 approved_by INTEGER REFERENCES users(id), approved_at TEXT);
CREATE INDEX IF NOT EXISTS idx_cases_site_status ON cases(site_id,status,priority DESC);
CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, at TEXT NOT NULL, actor_id INTEGER REFERENCES users(id),
 site_id TEXT, action TEXT NOT NULL, object_type TEXT NOT NULL, object_id TEXT NOT NULL, details_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY, site_id TEXT NOT NULL REFERENCES sites(id), task TEXT NOT NULL, status TEXT NOT NULL, dataset TEXT NOT NULL,
 config_json TEXT NOT NULL, results_json TEXT NOT NULL, started_at TEXT NOT NULL, ended_at TEXT);
CREATE TABLE IF NOT EXISTS federation_rounds(id TEXT PRIMARY KEY, site_id TEXT NOT NULL REFERENCES sites(id), schema_hash TEXT NOT NULL, status TEXT NOT NULL,
 family TEXT, equipment_model TEXT, participating_json TEXT NOT NULL, results_json TEXT NOT NULL, candidate_model_id TEXT REFERENCES models(id), created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS federation_updates(round_id TEXT NOT NULL REFERENCES federation_rounds(id), client_id TEXT NOT NULL,
 schema_hash TEXT NOT NULL, sample_count INTEGER NOT NULL, weights_json TEXT NOT NULL, submitted_at TEXT NOT NULL,
 PRIMARY KEY(round_id,client_id));
CREATE TABLE IF NOT EXISTS federation_validations(round_id TEXT NOT NULL REFERENCES federation_rounds(id), client_id TEXT NOT NULL,
 sample_count INTEGER NOT NULL, positive_count INTEGER NOT NULL, recall REAL NOT NULL, false_alert_rate REAL NOT NULL,
 latency_ms REAL NOT NULL, local_recall REAL, local_false_alert_rate REAL, submitted_at TEXT NOT NULL, PRIMARY KEY(round_id,client_id));
"""


def migrate() -> None:
    with connect() as db:
        db.executescript(SCHEMA)
        columns = {r["name"] for r in db.execute("PRAGMA table_info(cameras)")}
        if "sample_id" not in columns:
            db.execute("ALTER TABLE cameras ADD COLUMN sample_id TEXT")
        if "stopped_at" not in columns:
            db.execute("ALTER TABLE cameras ADD COLUMN stopped_at TEXT")
        if "last_error_at" not in columns:
            db.execute("ALTER TABLE cameras ADD COLUMN last_error_at TEXT")
        db.execute("UPDATE cameras SET stopped_at=?,last_frame_at=NULL WHERE stopped_at IS NULL AND last_error='Worker stopped'", (now(),))
        db.execute("""INSERT OR IGNORE INTO physical_samples(id,site_id,asset_id,display_name,created_at)
                      SELECT sample_id,site_id,asset_id,sample_id,MIN(created_at)
                      FROM captures WHERE sample_id IS NOT NULL AND sample_id<>''
                      GROUP BY site_id,sample_id,asset_id""")
        db.execute("""INSERT OR IGNORE INTO physical_samples(id,site_id,asset_id,display_name,created_at)
                      SELECT sample_id,site_id,asset_id,sample_id,? FROM cameras
                      WHERE sample_id IS NOT NULL AND sample_id<>''""", (now(),))
        label_columns = {r["name"] for r in db.execute("PRAGMA table_info(labels)")}
        if "scratch_count" not in label_columns:
            db.execute("ALTER TABLE labels ADD COLUMN scratch_count INTEGER")
        if "dent_count" not in label_columns:
            db.execute("ALTER TABLE labels ADD COLUMN dent_count INTEGER")
        draft_columns = {r["name"] for r in db.execute("PRAGMA table_info(review_drafts)")}
        if "views_json" not in draft_columns:
            db.execute("ALTER TABLE review_drafts ADD COLUMN views_json TEXT NOT NULL DEFAULT '{}' ")
        for name in ("ai_text", "ai_model", "ai_image_ids_json"):
            if name not in draft_columns:
                db.execute(f"ALTER TABLE review_drafts ADD COLUMN {name} TEXT")
        for name in ("ai_scratch_count", "ai_dent_count"):
            if name not in draft_columns:
                db.execute(f"ALTER TABLE review_drafts ADD COLUMN {name} INTEGER")
        if "ai_count_basis" not in draft_columns:
            db.execute("ALTER TABLE review_drafts ADD COLUMN ai_count_basis TEXT")
        validation_columns = {r["name"] for r in db.execute("PRAGMA table_info(federation_validations)")}
        if "local_recall" not in validation_columns:
            db.execute("ALTER TABLE federation_validations ADD COLUMN local_recall REAL")
        if "local_false_alert_rate" not in validation_columns:
            db.execute("ALTER TABLE federation_validations ADD COLUMN local_false_alert_rate REAL")
        model_columns = {r["name"] for r in db.execute("PRAGMA table_info(models)")}
        if "family" not in model_columns:
            db.execute("ALTER TABLE models ADD COLUMN family TEXT")
        if "equipment_model" not in model_columns:
            db.execute("ALTER TABLE models ADD COLUMN equipment_model TEXT")
        round_columns = {r["name"] for r in db.execute("PRAGMA table_info(federation_rounds)")}
        if "family" not in round_columns:
            db.execute("ALTER TABLE federation_rounds ADD COLUMN family TEXT")
        if "equipment_model" not in round_columns:
            db.execute("ALTER TABLE federation_rounds ADD COLUMN equipment_model TEXT")
        case_columns = {r["name"] for r in db.execute("PRAGMA table_info(cases)")}
        if "closure_note" not in case_columns:
            db.execute("ALTER TABLE cases ADD COLUMN closure_note TEXT")
        if "inspection_performed" not in case_columns:
            db.execute("ALTER TABLE cases ADD COLUMN inspection_performed INTEGER NOT NULL DEFAULT 0")


def row(db: sqlite3.Connection, sql: str, args: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    found = db.execute(sql, args).fetchone()
    return dict(found) if found else None


def rows(db: sqlite3.Connection, sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(x) for x in db.execute(sql, args).fetchall()]


def audit(db: sqlite3.Connection, actor: int | None, site: str | None, action: str,
          kind: str, object_id: str, details: str = "{}") -> None:
    db.execute("INSERT INTO audit(at,actor_id,site_id,action,object_type,object_id,details_json) VALUES(?,?,?,?,?,?,?)",
               (now(), actor, site, action, kind, object_id, details))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 300_000)
    return f"{salt.hex()}:{digest.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split(":", 1)
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 300_000)
        return hmac.compare_digest(candidate, bytes.fromhex(digest_hex))
    except (ValueError, TypeError):
        return False


def create_session(db: sqlite3.Connection, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    digest = hashlib.sha256(token.encode()).hexdigest()
    expires = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()
    db.execute("INSERT INTO sessions VALUES(?,?,?)", (digest, user_id, expires))
    return token


def user_from_token(db: sqlite3.Connection, token: str) -> dict[str, Any] | None:
    digest = hashlib.sha256(token.encode()).hexdigest()
    return row(db, "SELECT u.id,u.username,u.role,u.site_id,u.active,s.expires_at FROM users u JOIN sessions s ON u.id=s.user_id WHERE s.token_hash=? AND s.expires_at>?",
               (digest, now()))

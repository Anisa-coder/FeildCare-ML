"""SQLite persistence for FieldCare scan history."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from typing import Iterator


ROOT = Path(__file__).resolve().parent


def database_path() -> Path:
    configured_path = os.getenv("FIELDCARE_DB_PATH")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return ROOT / "data" / "fieldcare.db"


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    with connect() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                image BLOB NOT NULL,
                class_key TEXT NOT NULL,
                crop TEXT NOT NULL,
                disease TEXT NOT NULL,
                display_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                confidence_percent REAL NOT NULL,
                inference_ms REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_scan_history_created_at ON scan_history(created_at DESC)"
        )


def save_scan(
    *,
    filename: str,
    content_type: str,
    image: bytes,
    prediction: dict,
) -> int:
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO scan_history (
                filename, content_type, image, class_key, crop, disease,
                display_name, confidence, confidence_percent, inference_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                content_type,
                image,
                prediction["class_key"],
                prediction["crop"],
                prediction["disease"],
                prediction["display_name"],
                prediction["confidence"],
                prediction["confidence_percent"],
                prediction["inference_ms"],
                created_at,
            ),
        )
        return int(cursor.lastrowid)


def list_scans(*, limit: int, category: str, query: str) -> list[dict]:
    conditions: list[str] = []
    parameters: list[object] = []

    if category == "healthy":
        conditions.append("class_key LIKE ?")
        parameters.append("%__healthy")
    elif category == "disease":
        conditions.append("class_key NOT LIKE ?")
        parameters.append("%__healthy")

    if query:
        conditions.append(
            "(crop LIKE ? OR disease LIKE ? OR display_name LIKE ? OR filename LIKE ?)"
        )
        search_value = f"%{query}%"
        parameters.extend([search_value] * 4)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    parameters.append(limit)
    sql = f"""
        SELECT id, filename, class_key, crop, disease, display_name,
               confidence, confidence_percent, inference_ms, created_at
        FROM scan_history
        {where_clause}
        ORDER BY created_at DESC, id DESC
        LIMIT ?
    """
    with connect() as connection:
        rows = connection.execute(sql, parameters).fetchall()

    return [
        {
            **dict(row),
            "image_url": f"/history/{row['id']}/image",
        }
        for row in rows
    ]


def get_scan_image(scan_id: int) -> tuple[bytes, str] | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT image, content_type FROM scan_history WHERE id = ?", (scan_id,)
        ).fetchone()
    if row is None:
        return None
    return bytes(row["image"]), str(row["content_type"])


def get_history_stats() -> dict:
    with connect() as connection:
        totals = connection.execute(
            """
            SELECT
                COUNT(*) AS total_scans,
                SUM(CASE WHEN class_key LIKE '%__healthy' THEN 1 ELSE 0 END) AS healthy_diagnoses,
                SUM(CASE WHEN class_key NOT LIKE '%__healthy' THEN 1 ELSE 0 END) AS diseases_detected,
                COALESCE(AVG(confidence_percent), 0) AS average_confidence
            FROM scan_history
            """
        ).fetchone()
        common = connection.execute(
            """
            SELECT display_name, COUNT(*) AS occurrence_count
            FROM scan_history
            WHERE class_key NOT LIKE '%__healthy'
            GROUP BY class_key, display_name
            ORDER BY occurrence_count DESC, display_name ASC
            LIMIT 1
            """
        ).fetchone()

    return {
        "total_scans": int(totals["total_scans"] or 0),
        "healthy_diagnoses": int(totals["healthy_diagnoses"] or 0),
        "diseases_detected": int(totals["diseases_detected"] or 0),
        "average_confidence": round(float(totals["average_confidence"] or 0), 1),
        "most_common_disease": common["display_name"] if common else None,
    }

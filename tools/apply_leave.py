from langchain.tools import tool
import os
import sqlite3
from datetime import datetime


DB_PATH = "./database/hr.db"


def _is_valid_sqlite(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            return f.read(16) == b"SQLite format 3\x00"
    except OSError:
        return False


def _init_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if not _is_valid_sqlite(DB_PATH):
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS leave_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            leave_type TEXT NOT NULL,
            reason TEXT,
            status TEXT DEFAULT 'Pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    return conn


def apply_leave_fn(
    employee_id: str,
    leave_type: str,
    start_date: str,
    end_date: str,
    reason: str,
) -> dict:
    """
    Apply for employee leave.

    Args:
        employee_id: Employee ID
        leave_type: Casual, Sick, Earned, etc.
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        reason: Reason for leave
    """

    # Validate dates
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return {
            "status": "error",
            "message": "Dates must be in YYYY-MM-DD format."
        }

    if start > end:
        return {
            "status": "error",
            "message": "Start date cannot be after end date."
        }

    conn = _init_db()
    cursor = conn.cursor()

    # Check overlapping leave
    cursor.execute(
        """
        SELECT id, start_date, end_date
        FROM leave_requests
        WHERE employee_id = ?
          AND status IN ('Pending', 'Approved')
          AND start_date <= ?
          AND end_date >= ?
        """,
        (employee_id, end_date, start_date),
    )

    overlap = cursor.fetchone()

    if overlap:
        conn.close()
        return {
            "status": "error",
            "message": "Leave request overlaps with an existing leave.",
            "existing_leave": {
                "id": overlap[0],
                "start_date": overlap[1],
                "end_date": overlap[2],
            },
        }

    # Insert leave request
    cursor.execute(
        """
        INSERT INTO leave_requests
        (
            employee_id,
            leave_type,
            start_date,
            end_date,
            reason,
            status
        )
        VALUES (?, ?, ?, ?, ?, 'Pending')
        """,
        (
            employee_id,
            leave_type,
            start_date,
            end_date,
            reason,
        ),
    )

    leave_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return {
        "status": "success",
        "leave_id": leave_id,
        "employee_id": employee_id,
        "leave_type": leave_type,
        "start_date": start_date,
        "end_date": end_date,
        "reason": reason,
        "approval_status": "Pending",
        "message": "Leave request submitted successfully."
    }


apply_leave = tool(apply_leave_fn)

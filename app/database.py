import sqlite3
from pathlib import Path

DB_PATH = Path("data/hr_agent.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS grievances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            grievance_id TEXT UNIQUE,
            employee_name TEXT,
            description TEXT NOT NULL,
            category TEXT,
            severity TEXT,
            department TEXT,
            status TEXT DEFAULT 'Open',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def create_grievance(data):
    conn = get_conn()
    conn.execute("""
        INSERT INTO grievances
        (grievance_id, employee_name, description, category, severity, department, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, tuple(data[k] for k in ["grievance_id", "employee_name", "description", "category", "severity", "department", "status"]))
    conn.commit()
    conn.close()

def list_grievances():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM grievances ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_status(grievance_id, status):
    conn = get_conn()
    cur = conn.execute("UPDATE grievances SET status=? WHERE grievance_id=?", (status, grievance_id))
    conn.commit()
    conn.close()
    return cur.rowcount > 0

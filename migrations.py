# backend/migration.py
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "e1.db")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

def table_columns(table: str):
    cur.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]

def has_column(table: str, column: str) -> bool:
    return column in table_columns(table)

def add_column_safe(table: str, sql: str, note: str = ""):
    try:
        cur.execute(sql)
        print(f"Added column on {table}: {note or sql}")
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "duplicate column name" in msg:
            print(f"Column already exists (skip): {note or sql}")
        elif "no such table" in msg:
            print(f"Table {table} not found (skip): {note or sql}")
        else:
            raise

print(f"Using DB: {DB_PATH}")

# --- Ensure orders table has special_instruction ---
try:
    cols_orders = table_columns("orders")
    print("orders columns:", cols_orders)
    if not has_column("orders", "special_instruction"):
        add_column_safe(
            "orders",
            "ALTER TABLE orders ADD COLUMN special_instruction TEXT DEFAULT ''",
            "orders.special_instruction",
        )
    else:
        print("orders.special_instruction already present")
except sqlite3.OperationalError as e:
    print("Skipping 'orders' migration (table may not exist yet):", e)

# --- Ensure stations table has type, capabilities, is_active, speed_factor ---
try:
    cols_stations = table_columns("stations")
    print("stations columns:", cols_stations)

    if not has_column("stations", "type"):
        add_column_safe(
            "stations",
            "ALTER TABLE stations ADD COLUMN type TEXT DEFAULT 'normal'",
            "stations.type",
        )
    else:
        print("stations.type already present")

    if not has_column("stations", "capabilities"):
        add_column_safe(
            "stations",
            "ALTER TABLE stations ADD COLUMN capabilities TEXT DEFAULT ''",
            "stations.capabilities",
        )
    else:
        print("stations.capabilities already present")

    # Some earlier scripts may have added 'active' instead of 'is_active'.
    # We add 'is_active' (the one used by current models) and copy data from 'active' if present.
    had_active = has_column("stations", "active")
    if not has_column("stations", "is_active"):
        add_column_safe(
            "stations",
            "ALTER TABLE stations ADD COLUMN is_active BOOLEAN DEFAULT 1",
            "stations.is_active",
        )
        if had_active:
            try:
                cur.execute("UPDATE stations SET is_active = COALESCE(active, 1)")
                print("Copied stations.active → stations.is_active")
            except sqlite3.OperationalError as e:
                print("Could not copy active → is_active:", e)
    else:
        # If both exist, ensure is_active has values; if NULL and active exists, copy.
        if had_active:
            try:
                cur.execute("UPDATE stations SET is_active = COALESCE(is_active, active, 1)")
                print("Normalized stations.is_active using active where needed")
            except sqlite3.OperationalError as e:
                print("Normalization of is_active failed:", e)
        else:
            print("stations.is_active already present")

    if not has_column("stations", "speed_factor"):
        add_column_safe(
            "stations",
            "ALTER TABLE stations ADD COLUMN speed_factor REAL DEFAULT 1.0",
            "stations.speed_factor",
        )
    else:
        print("stations.speed_factor already present")

except sqlite3.OperationalError as e:
    print("Skipping 'stations' migration (table may not exist yet):", e)

# --- Optional tidy-ups (safe, best-effort) ---
# Normalize capabilities: collapse whitespace (best-effort; SQLite lacks regex, keep simple).
try:
    cur.execute("UPDATE stations SET capabilities = TRIM(LOWER(capabilities)) WHERE capabilities IS NOT NULL")
    print("Normalized stations.capabilities (trim + lower)")
except sqlite3.OperationalError as e:
    print("Capabilities normalization skipped:", e)

# Normalize type values
try:
    cur.execute("""
        UPDATE stations
        SET type = CASE
            WHEN LOWER(IFNULL(type,'')) IN ('specialized','specialised') THEN 'specialized'
            ELSE 'normal'
        END
    """)
    print("Normalized stations.type to {'normal','specialized'}")
except sqlite3.OperationalError as e:
    print("Type normalization skipped:", e)

conn.commit()
conn.close()
print("Migration complete ✅")

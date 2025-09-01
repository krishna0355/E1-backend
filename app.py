
from __future__ import annotations

from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict
import io
import csv

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

from .db import db_session, engine, Base
from .models import Task, Station, TaskStatus, Order
from .allocator import allocate_work, offer_first_tasks, distribute_even_load

app = Flask(__name__)
CORS(app)  # allow http://localhost:5173 to call the API in dev

# Ensure tables exist at startup
Base.metadata.create_all(bind=engine)

# --- Auto-seed 6 default stations if none exist ---
with db_session() as _db:
    if _db.query(Station).count() == 0:
        for i in range(1, 7):
            _db.add(
                Station(
                    station_code=f"ST-{i}",
                    display_name=f"Station {i}",
                    speed_factor=1.0,
                    type="normal",
                    capabilities="standard",
                    is_active=True,
                )
            )

# =========================
# Station Manager (CRUD)
# =========================

def _station_to_dict(s: Station) -> dict:
    return {
        "station_code": s.station_code,
        "display_name": s.display_name,
        "type": (s.type or "normal"),
        "capabilities": [t for t in (s.capabilities or "").split(",") if t.strip()],
        "is_active": bool(s.is_active),
        "id": s.id,
    }


# -------------------------
# Health
# -------------------------
@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "E1 backend",
        "time": datetime.utcnow().isoformat(),
    }

# -------------------------
# Admin: seed / upload / allocate / overview
# -------------------------
@app.post("/admin/seed-stations")
def admin_seed_stations():
    """
    Body (optional):
    {
      "normal": 6,
      "specialized": 0,
      "capabilities": ["fragile","cold-chain"]
    }
    Creates ST-<n>… codes. Existing codes are preserved (idempotent).
    """
    data = request.get_json(silent=True) or {}
    normal = int(data.get("normal", 6) or 0)
    specialized = int(data.get("specialized", 0) or 0)
    caps_list = data.get("capabilities", [])
    if not isinstance(caps_list, list):
        caps_list = []
    caps_csv = ",".join([str(c).strip().lower() for c in caps_list if str(c).strip()])

    created = 0
    with db_session() as db:
        existing_codes = {s.station_code for s in db.query(Station).all()}
        # find next available numeric suffix
        next_idx = 1
        while f"ST-{next_idx}" in existing_codes:
            next_idx += 1

        # normals
        for i in range(normal):
            code = f"ST-{next_idx + i}"
            if code not in existing_codes:
                db.add(Station(
                    station_code=code,
                    display_name=f"Station {next_idx + i}",
                    type="normal",
                    capabilities="",
                    speed_factor=1.0,
                    is_active=True,
                ))
                created += 1
                existing_codes.add(code)

        # specialized
        start_spec = next_idx + normal
        for i in range(specialized):
            code = f"ST-{start_spec + i}"
            if code not in existing_codes:
                db.add(Station(
                    station_code=code,
                    display_name=f"Station {start_spec + i}",
                    type="specialized",
                    capabilities=caps_csv,
                    speed_factor=1.0,
                    is_active=True,
                ))
                created += 1
                existing_codes.add(code)

    return jsonify({"created": created, "normal": normal, "specialized": specialized})



# -------------------------
# Admin: Add/Edit/Remove stations dynamically
# -------------------------
@app.post("/admin/station/add")
def add_station():
    """Add a new station (normal or specialized)."""
    data = request.json or {}
    code = data.get("station_code")
    name = data.get("display_name", code)
    station_type = data.get("type", "normal")
    capabilities = ",".join(data.get("capabilities", []))

    with db_session() as db:
        if db.query(Station).filter_by(station_code=code).first():
            return jsonify({"error": "Station already exists"}), 400

        st = Station(
            station_code=code,
            display_name=name,
            speed_factor=1.0,
            type=station_type,
            capabilities=capabilities,
            is_active=True,
        )
        db.add(st)
        db.commit()
        return jsonify({"ok": True, "station": st.to_dict()})

@app.put("/admin/station/edit/<code>")
def edit_station(code):
    """Edit station display name, type, or capabilities."""
    data = request.json or {}
    with db_session() as db:
        st = db.query(Station).filter_by(station_code=code).first()
        if not st:
            return jsonify({"error": "Station not found"}), 404

        st.display_name = data.get("display_name", st.display_name)
        st.type = data.get("type", st.type)
        if "capabilities" in data:
            st.capabilities = ",".join(data["capabilities"])
        db.commit()
        return jsonify({"ok": True, "station": st.to_dict()})

@app.delete("/admin/station/remove/<code>")
def remove_station(code):
    """Soft-delete a station (set inactive)."""
    with db_session() as db:
        st = db.query(Station).filter_by(station_code=code).first()
        if not st:
            return jsonify({"error": "Station not found"}), 404
        st.is_active = False
        db.commit()
        return jsonify({"ok": True, "removed": code})

@app.get("/admin/stations")
def list_stations():
    """List all active stations."""
    with db_session() as db:
        stations = db.query(Station).filter_by(is_active=True).all()
        return jsonify([s.to_dict() for s in stations])


# =========================
# Station Manager (CRUD)
# =========================

def _station_to_dict(s: Station) -> dict:
    return {
        "station_code": s.station_code,
        "display_name": s.display_name,
        "type": (s.type or "normal"),
        "capabilities": [t for t in (s.capabilities or "").split(",") if t.strip()],
        "is_active": bool(s.is_active),
        "id": s.id,
    }

@app.get("/admin/station-manager/list")
def sm_list():
    with db_session() as db:
        rows = db.query(Station).order_by(Station.id.asc()).all()
        return jsonify([_station_to_dict(s) for s in rows])

@app.post("/admin/station-manager/add")
def sm_add():
    data = request.get_json(silent=True) or {}
    code = (data.get("station_code") or "").strip()
    name = (data.get("display_name") or code).strip()
    stype = (data.get("type") or "normal").strip().lower()
    caps = data.get("capabilities") or []
    if not code:
        return jsonify({"ok": False, "error": "station_code required"}), 400

    with db_session() as db:
        exists = db.query(Station).filter(Station.station_code == code).first()
        if exists:
            return jsonify({"ok": False, "error": "station_code already exists"}), 400

        s = Station(
            station_code=code,
            display_name=name or code,
            type=stype,
            capabilities=",".join([c.strip() for c in caps if c.strip()]),
            speed_factor=1.0,
            is_active=True,
        )
        db.add(s)
        db.flush()  # to get id
        return jsonify({"ok": True, "station": _station_to_dict(s)})

@app.put("/admin/station-manager/edit/<code>")
def sm_edit(code: str):
    data = request.get_json(silent=True) or {}
    with db_session() as db:
        s = db.query(Station).filter(Station.station_code == code).first()
        if not s:
            return jsonify({"ok": False, "error": "station not found"}), 404

        if "display_name" in data:
            s.display_name = (data["display_name"] or s.display_name).strip()
        if "type" in data:
            s.type = (data["type"] or s.type or "normal").strip().lower()
        if "capabilities" in data and isinstance(data["capabilities"], list):
            s.capabilities = ",".join([c.strip() for c in data["capabilities"] if c.strip()])
        if "is_active" in data:
            s.is_active = bool(data["is_active"])

        return jsonify({"ok": True, "station": _station_to_dict(s)})

@app.delete("/admin/station-manager/remove/<code>")
def sm_remove(code: str):
    # soft delete: set inactive
    with db_session() as db:
        s = db.query(Station).filter(Station.station_code == code).first()
        if not s:
            return jsonify({"ok": False, "error": "station not found"}), 404
        s.is_active = False
        return jsonify({"ok": True, "removed": code})

@app.post("/admin/ensure-offers")
def ensure_offers():
    from .db import SessionLocal
    try:
        db = SessionLocal()
        res = allocate_work(db)   # ensures offers + assignment
        return {"ok": True, **res}
    finally:
        db.close()


# -------------------------
# Admin: Upload Orders
# -------------------------
# --- Upload Orders (robust CSV) ---
@app.post("/admin/upload-orders")
def admin_upload_orders():
    """Upload a CSV and insert orders. Returns counts + preview.

    Required (flexible naming accepted):
      - order_id
      - items
      - qty
      - sku_mix
      - priority
      - est_pack_time_sec
      - due_by

    Optional:
      - special_instruction   (fragile, cold-chain, high value, none, etc.)
    """
    from io import BytesIO
    import pandas as pd

    if "file" not in request.files:
        return jsonify({"ok": False, "error": "file missing"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"ok": False, "error": "please upload a .csv file"}), 400

    raw = f.read()
    if not raw:
        return jsonify({"ok": False, "error": "empty file"}), 400

    # Try multiple encodings to be permissive
    df = None
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            df = pd.read_csv(BytesIO(raw), encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        return jsonify({"ok": False, "error": "failed to read CSV (encoding/format)"}), 400

    # Normalize column names: trim, lowercase, spaces/dashes -> underscores
    norm = lambda s: str(s).strip().lower().replace(" ", "_").replace("-", "_")
    df.columns = [norm(c) for c in df.columns]

    # Map flexible aliases -> canonical names
    alias = {
        "order_id": {"order_id", "id", "order"},
        "items": {"items", "item_list", "item"},
        "qty": {"qty", "quantity", "units"},
        "sku_mix": {"sku_mix", "sku", "sku_count"},
        "priority": {"priority", "prio"},
        "est_pack_time_sec": {"est_pack_time_sec", "pack_time", "estimated_pack_time_sec", "est_time_sec"},
        "due_by": {"due_by", "due", "deadline", "due_date"},
        # optional:
        "special_instruction": {"special_instruction", "special_instructions", "instructions", "tags"},
    }

    # Resolve actual column names present in df for each canonical key
    present = set(df.columns)
    resolved = {}
    missing = []
    for key, options in alias.items():
        found = next((c for c in present if c in options), None)
        if found:
            resolved[key] = found
        else:
            if key != "special_instruction":  # optional
                missing.append(key)

    if missing:
        return jsonify({
            "ok": False,
            "error": f"missing required columns: {missing}. "
                     f"Found columns: {sorted(list(present))}"
        }), 400

    # Build records with safe parsing
    def to_int(x, default=0):
        try:
            if pd.isna(x) or x is None or str(x).strip() == "":
                return default
            return int(float(x))
        except Exception:
            return default

    df = df.where(pd.notnull(df), None)

    records = []
    for _, row in df.iterrows():
        records.append({
            "order_id": str(row[resolved["order_id"]]).strip() if row[resolved["order_id"]] is not None else "",
            "items": str(row[resolved["items"]]).strip() if row[resolved["items"]] is not None else "",
            "qty": to_int(row[resolved["qty"]], 0),
            "sku_mix": to_int(row.get(resolved.get("sku_mix")), 0),
            "priority": (str(row[resolved["priority"]]).strip() if row[resolved["priority"]] is not None else "standard"),
            "est_pack_time_sec": to_int(row[resolved["est_pack_time_sec"]], 1),
            "due_by": str(row[resolved["due_by"]]).strip() if row[resolved["due_by"]] is not None else "",
            "special_instruction": (
                str(row[resolved["special_instruction"]]).strip().lower()
                if resolved.get("special_instruction") and row.get(resolved["special_instruction"]) is not None
                else ""
            ),
        })

    # Insert
    inserted, duplicates = 0, 0
    with db_session() as db:
        for r in records:
            if not r["order_id"]:
                continue
            if db.query(Order).filter_by(order_id=r["order_id"]).first():
                duplicates += 1
                continue

            db.add(Order(
                order_id=r["order_id"],
                items=r["items"],
                qty=r["qty"],
                sku_mix=r["sku_mix"],
                priority=r["priority"],
                est_pack_time_sec=r["est_pack_time_sec"],
                due_by=r["due_by"],
                # make sure your Order model has this column (migration adds it)
                special_instruction=r["special_instruction"],
            ))
            inserted += 1

    preview = [{
        "order_id": r["order_id"],
        "items": r["items"],
        "qty": r["qty"],
        "est_pack_time_sec": r["est_pack_time_sec"],
        "due_by": r["due_by"],
        "priority": r["priority"],
        "special_instruction": r["special_instruction"],
    } for r in records[:5]]

    return jsonify({
        "ok": True,
        "inserted": inserted,
        "duplicates": duplicates,
        "total_rows": len(records),
        "preview": preview,
    })


# -------------------------
# Admin: Allocate Work
# -------------------------
@app.post("/admin/allocate")
def admin_allocate():
    with db_session() as db:
        result = allocate_work(db)
        try:
            offer_first_tasks(db)
        except Exception:
            pass
        return jsonify(result)

@app.get("/admin/allocate")
def admin_allocate_get():
    with db_session() as db:
        result = allocate_work(db)
        try:
            offer_first_tasks(db)
        except Exception:
            pass
        return jsonify(result)


# -------------------------
# NEW: Even Load Distribution
# -------------------------
@app.get("/api/load-distribution")
def api_load_distribution():
    with db_session() as db:
        raw = distribute_even_load(db)

        stations_out = []
        overall_total = raw.get("overall_total", 0) or 0

        for s in raw.get("stations", []):
            sid = s.get("station")
            total = round(s.get("total_time", 0), 2)   # minutes with decimals
            pct = (total / overall_total * 100) if overall_total > 0 else 0
            stations_out.append({
                "name": f"Station {sid}",
                "load": total,
                "percent": round(pct, 2),
            })

        return jsonify({
            "stations": stations_out,
            "total_time": round(overall_total, 2)
        })

@app.get("/api/load-distribution/download")
def api_load_distribution_download():
    with db_session() as db:
        result = distribute_even_load(db)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["order_id", "est_pack_time_min", "assigned_station"])

        for s in result["stations"]:
            sid = s["station"]
            total = round(s["total_time"], 2)
            for o in s.get("orders", []):
                writer.writerow([o["order_id"], round(o["time"], 2), f"Station {sid}"])
            # Subtotal row
            writer.writerow([f"Station {sid} Total", total, ""])

        # Grand total
        writer.writerow(["Overall Total", round(result.get("overall_total", 0), 2), ""])
        csv_data = output.getvalue().encode("utf-8")   # ✅ convert string → bytes

        return Response(
            csv_data,
            mimetype="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": "attachment; filename=load_distribution.csv"
            }
        )

# -------------------------
# Admin: Overview
# -------------------------
@app.get("/admin/overview")
def admin_overview():
    with db_session() as db:
        total = db.query(Task).count()
        offered = db.query(Task).filter(Task.status == TaskStatus.OFFERED).count()
        in_prog = db.query(Task).filter(Task.status == TaskStatus.IN_PROGRESS).count()
        done = db.query(Task).filter(Task.status.in_([TaskStatus.COMPLETED, TaskStatus.LATE])).count()

        stations = db.query(Station).filter_by(is_active=True).all()
        out = []
        for s in stations:
            queue = db.query(Task).filter(Task.station_id == s.id, Task.status == TaskStatus.QUEUED).count()
            s_off = db.query(Task).filter(Task.station_id == s.id, Task.status == TaskStatus.OFFERED).count()
            s_prog = db.query(Task).filter(Task.station_id == s.id, Task.status == TaskStatus.IN_PROGRESS).count()
            s_done = db.query(Task).filter(Task.station_id == s.id, Task.status.in_([TaskStatus.COMPLETED, TaskStatus.LATE])).count()
            out.append(
    {
        "station_code": s.station_code,
        "display_name": s.display_name,
        "queue": queue,
        "offered": s_off,
        "in_progress": s_prog,
        "done": s_done,
        # new fields for UI badges
        "type": (s.type or "normal"),
        "capabilities": (s.capabilities or ""),
    }
)

        return jsonify({"tasks": {"total": total, "offered": offered, "in_progress": in_prog, "completed": done}, "stations": out})

# -------------------------
# Admin: Reset (danger)
# -------------------------
# --- Reset everything (danger) ---
@app.post("/admin/reset")
def admin_reset():
    with db_session() as db:
        # delete children first (Task -> Order), then stations
        db.query(Task).delete(synchronize_session=False)
        db.query(Order).delete(synchronize_session=False)
        db.query(Station).delete(synchronize_session=False)
        db.commit()

        return jsonify({"ok": True, "tasks": 0, "orders": 0, "stations": 0})


# -------------------------
# Admin: Performance
# -------------------------
@app.get("/admin/performance")
def admin_performance():
    from sqlalchemy import func
    with db_session() as db:
        stations = db.query(Station).filter_by(is_active=True).all()
        rows = []
        for s in stations:
            q_completed = db.query(Task).filter(
                Task.station_id == s.id,
                Task.status.in_([TaskStatus.COMPLETED, TaskStatus.LATE]),
                Task.started_at.isnot(None),
                Task.completed_at.isnot(None),
            )
            completed = q_completed.count()

            avg_handle_sec = q_completed.with_entities(
                func.avg(func.strftime('%s', Task.completed_at) - func.strftime('%s', Task.started_at))
            ).scalar() or 0

            avg_overrun_sec = q_completed.with_entities(
                func.avg(func.strftime('%s', Task.completed_at) - func.strftime('%s', Task.due_at))
            ).scalar() or 0

            on_time = q_completed.filter(Task.completed_at <= Task.due_at).count()
            on_time_pct = (on_time / completed * 100.0) if completed else 0.0

            queued = db.query(Task).filter(Task.station_id == s.id, Task.status == TaskStatus.QUEUED).count()
            offered = db.query(Task).filter(Task.station_id == s.id, Task.status == TaskStatus.OFFERED).count()
            in_prog = db.query(Task).filter(Task.station_id == s.id, Task.status == TaskStatus.IN_PROGRESS).count()

            rows.append({
                "station_code": s.station_code,
                "display_name": s.display_name,
                "completed": completed,
                "avg_handle_sec": float(avg_handle_sec),
                "avg_overrun_sec": float(avg_overrun_sec),
                "on_time_pct": on_time_pct,
                "queued": queued,
                "offered": offered,
                "in_progress": in_prog,
            })
        return jsonify({"rows": rows, "generated_at": datetime.utcnow().isoformat()})

# -------------------------
# Station flows
# -------------------------
@app.get("/station/<code>/current")
def station_current(code: str):
    with db_session() as db:
        station = db.query(Station).filter_by(station_code=code).first()
        if not station:
            return jsonify({"error": "unknown station"}), 404

        t = db.query(Task).filter(
            Task.station_id == station.id,
            Task.status.in_([TaskStatus.IN_PROGRESS, TaskStatus.OFFERED]),
        ).order_by(Task.assigned_seq.asc()).first()

        if not t:
            return jsonify({"message": "NO_TASK"})
        return jsonify(task_to_json(t))

@app.post("/station/<code>/accept")
def station_accept(code: str):
    with db_session() as db:
        station = db.query(Station).filter_by(station_code=code).first()
        if not station:
            return jsonify({"error": "unknown station"}), 404

        payload = request.get_json(silent=True) or {}
        task_id = payload.get("task_id")
        if not task_id:
            return jsonify({"error": "task_id missing"}), 400

        t = db.query(Task).filter_by(id=task_id, station_id=station.id).first()
        if not t or t.status != TaskStatus.OFFERED:
            return jsonify({"error": "no offered task"}), 400

        now = datetime.utcnow()
        t.status = TaskStatus.IN_PROGRESS
        t.accepted_at = now
        t.started_at = now
        t.due_at = now + timedelta(seconds=t.duration_sec)

        return jsonify(task_to_json(t))

@app.post("/station/<code>/complete")
def station_complete(code: str):
    with db_session() as db:
        station = db.query(Station).filter_by(station_code=code).first()
        if not station:
            return jsonify({"error": "unknown station"}), 404

        payload = request.get_json(silent=True) or {}
        task_id = payload.get("task_id")
        if not task_id:
            return jsonify({"error": "task_id missing"}), 400

        t = db.query(Task).filter_by(id=task_id, station_id=station.id).first()
        if not t or t.status != TaskStatus.IN_PROGRESS:
            return jsonify({"error": "no in-progress task"}), 400

        now = datetime.utcnow()
        t.completed_at = now
        t.status = TaskStatus.COMPLETED if (t.due_at and now <= t.due_at) else TaskStatus.LATE

        offer_first_tasks(db)

        return jsonify({"completed": task_to_json(t)})

# -------------------------
# Debug helpers
# -------------------------
@app.get("/admin/debug")
def admin_debug():
    with db_session() as db:
        return jsonify({
            "db_url": str(engine.url),
            "stations": db.query(Station).count(),
            "active_stations": db.query(Station).filter_by(is_active=True).count(),
            "orders": db.query(Order).count(),
            "tasks": db.query(Task).count(),
        })

# -------------------------
# Utilities
# -------------------------
def task_to_json(t: Task) -> Dict:
    o = t.order
    return {
        "task_id": t.id,
        "station": t.station.display_name if t.station else None,
        "status": t.status.value,
        "sequence": t.assigned_seq,
        "duration_sec": t.duration_sec,
        "offered_at": t.offered_at.isoformat() if t.offered_at else None,
        "accepted_at": t.accepted_at.isoformat() if t.accepted_at else None,
        "started_at": t.started_at.isoformat() if t.started_at else None,
        "due_at": t.due_at.isoformat() if t.due_at else None,
        "completed_at": t.completed_at.isoformat() if t.completed_at else None,
        "order": {
            "order_id": o.order_id if o else None,
            "items": o.items if o else None,
            "qty": o.qty if o else None,
            "priority": o.priority if o else None,
            "est_pack_time_sec": o.est_pack_time_sec if o else None,
            "due_by": o.due_by if o else None,
        },
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

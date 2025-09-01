from datetime import datetime
from typing import Dict, Tuple, List, Iterable, Optional, Set
from heapq import heappush, heappop
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from models import Task, Station, TaskStatus, Order
import io, csv
from flask import send_file, Response

def _compute_duration_sec(est_pack_time_sec: int, speed_factor: float) -> int:
    """Scale order pack time by station speed factor, clamp to >=1."""
    if not est_pack_time_sec or est_pack_time_sec <= 0:
        est_pack_time_sec = 1
    if not speed_factor or speed_factor <= 0:
        speed_factor = 1.0
    dur = int(round(est_pack_time_sec / speed_factor))
    return max(dur, 1)


# -------- Priority helpers (lower value = higher priority) --------
PRIORITY_RANK = {
    "rush": 1,
    "sameday": 1,
    "same-day": 1,
    "express": 1,
    "expedited": 2,
    "high": 2,
    "high-priority": 2,
    "priority": 2,
    "standard": 3,
}

def _priority_key(priority: Optional[str]) -> Tuple[int, str]:
    """
    Convert priority to a sortable key:
    - By rank (rush/sameday > expedited > standard)
    - Then by string (stable tie-breaker)
    """
    if not priority:
        return (PRIORITY_RANK.get("standard", 3), "standard")
    p = str(priority).strip().lower()
    tokens = [t.strip() for t in p.split(",")]
    rank = min((PRIORITY_RANK.get(t, 999) for t in tokens), default=PRIORITY_RANK.get("standard", 3))
    return (rank, p)


# -------- Capabilities / tags helpers --------
def _tags_from_special(instr: Optional[str]) -> Set[str]:
    """
    Extract tokens from special_instruction (e.g., "fragile, cold-chain").
    Empty/none/normal/na are treated as no special tag.
    """
    if not instr:
        return set()
    raw = [t.strip().lower() for t in str(instr).split(",") if t.strip()]
    out = {t for t in raw if t not in {"none", "normal", "na", "n/a"}}
    return out

def _cap_set(capabilities: Optional[str]) -> Set[str]:
    if not capabilities:
        return set()
    return {t.strip().lower() for t in str(capabilities).split(",") if t.strip()}


# -------- Load / heap helpers --------
def _build_load_map(db: Session, stations: List[Station]) -> Dict[int, int]:
    """Current remaining load (sum of duration) for QUEUED + OFFERED + IN_PROGRESS per station."""
    load_by_sid: Dict[int, int] = {s.id: 0 for s in stations}
    inqueue = (
        db.query(Task.station_id, func.coalesce(func.sum(Task.duration_sec), 0))
        .filter(Task.status.in_([TaskStatus.QUEUED, TaskStatus.OFFERED, TaskStatus.IN_PROGRESS]))
        .group_by(Task.station_id)
        .all()
    )
    for sid, total in inqueue:
        if sid in load_by_sid:
            load_by_sid[sid] = int(total or 0)
    return load_by_sid


def _make_heap_for(stations: Iterable[Station], load_by_sid: Dict[int, int]) -> List[Tuple[int, int]]:
    """Min-heap of (load, station_id) for a given station subset."""
    heap: List[Tuple[int, int]] = []
    for s in stations:
        heappush(heap, (load_by_sid.get(s.id, 0), s.id))
    return heap


# -------- Core allocation --------
def allocate_work(db: Session) -> dict:
    """
    Capability & priority-aware allocation:
      1) Assign ALL orders with a special_instruction to CAPABILITY-MATCHED specialized stations.
      2) Assign remaining orders to NORMAL stations only.
      3) If still remaining, allow SPECIALIZED to help like normal (fallback).
    Within each pool, use min-heap (least current load). Orders are processed by priority.
    """
    # Active stations
    stations: List[Station] = db.query(Station).filter_by(is_active=True).all()
    if not stations:
        return {"assigned": 0, "error": "no active stations"}

    # Partition stations
    specialized: List[Station] = [s for s in stations if (s.type or "").lower() == "specialized"]
    normal: List[Station] = [s for s in stations if (s.type or "").lower() != "specialized"]

    # Load map
    load_by_sid = _build_load_map(db, stations)
    station_by_id: Dict[int, Station] = {s.id: s for s in stations}
    spec_caps_by_sid: Dict[int, Set[str]] = {s.id: _cap_set(s.capabilities) for s in specialized}

    # Per-station sequence baseline
    max_seq_by_sid: Dict[int, int] = {
        s.id: (db.query(func.max(Task.assigned_seq)).filter(Task.station_id == s.id).scalar() or 0)
        for s in stations
    }

    # Unassigned orders
    unassigned: List[Order] = (
        db.query(Order)
        .outerjoin(Task, Task.order_id == Order.id)
        .filter(Task.id.is_(None))
        .all()
    )
    if not unassigned:
        offer_first_tasks(db)
        return {"assigned": 0}

    # Split orders by presence of special_instruction
    special_orders: List[Order] = []
    normal_orders: List[Order] = []
    for o in unassigned:
        tags = _tags_from_special(getattr(o, "special_instruction", None))
        if tags:
            special_orders.append(o)
        else:
            normal_orders.append(o)

    # Sort by priority (high -> low)
    special_orders.sort(key=lambda o: _priority_key(o.priority))
    normal_orders.sort(key=lambda o: _priority_key(o.priority))

    assigned = 0

    # Heaps
    normal_heap: List[Tuple[int, int]] = _make_heap_for(normal, load_by_sid)
    # We'll create tiny eligible heaps for specialized on the fly

    # -------- Pass 1: match special orders to specialized stations with intersecting capabilities --------
    leftover: List[Order] = []
    for o in special_orders:
        order_tags = _tags_from_special(getattr(o, "special_instruction", None))
        eligible_spec_ids: List[int] = []
        for s in specialized:
            caps = spec_caps_by_sid.get(s.id, set())
            if caps and (caps & order_tags):
                eligible_spec_ids.append(s.id)

        chosen_sid: Optional[int] = None
        if eligible_spec_ids:
            spec_heap: List[Tuple[int, int]] = []
            for sid in eligible_spec_ids:
                heappush(spec_heap, (load_by_sid.get(sid, 0), sid))
            if spec_heap:
                _, chosen_sid = heappop(spec_heap)

        if chosen_sid is None:
            # keep for later passes (normal then fallback)
            leftover.append(o)
            continue

        # assign to chosen specialized station
        s = station_by_id[chosen_sid]
        dur = _compute_duration_sec(o.est_pack_time_sec or 1, s.speed_factor or 1.0)
        next_seq = max_seq_by_sid[chosen_sid] + 1
        max_seq_by_sid[chosen_sid] = next_seq

        db.add(Task(
            order_id=o.id,
            station_id=chosen_sid,
            status=TaskStatus.QUEUED,
            assigned_seq=next_seq,
            duration_sec=dur,
        ))
        assigned += 1
        load_by_sid[chosen_sid] = load_by_sid.get(chosen_sid, 0) + dur
        # (no global spec heap to push back into)

    # -------- Pass 2: assign remaining orders (leftover special + normal) to NORMAL stations only --------
    remaining: List[Order] = leftover + normal_orders
    for o in remaining:
        if not normal_heap:
            break  # move to fallback pass
        _, sid = heappop(normal_heap)
        s = station_by_id[sid]

        dur = _compute_duration_sec(o.est_pack_time_sec or 1, s.speed_factor or 1.0)
        next_seq = max_seq_by_sid[sid] + 1
        max_seq_by_sid[sid] = next_seq

        db.add(Task(
            order_id=o.id,
            station_id=sid,
            status=TaskStatus.QUEUED,
            assigned_seq=next_seq,
            duration_sec=dur,
        ))
        assigned += 1
        load_by_sid[sid] = load_by_sid.get(sid, 0) + dur
        heappush(normal_heap, (load_by_sid[sid], sid))

    # Remove the ones we actually assigned in Pass 2
    assigned_ids = {t.order_id for t in db.new if isinstance(t, Task)}
    remaining = [o for o in remaining if o.id not in assigned_ids]

    # -------- Pass 3 (fallback): if anything still unassigned, allow ANY station to take them --------
    if remaining:
        any_heap = _make_heap_for(stations, load_by_sid)
        for o in remaining:
            if not any_heap:
                break
            _, sid = heappop(any_heap)
            s = station_by_id[sid]

            dur = _compute_duration_sec(o.est_pack_time_sec or 1, s.speed_factor or 1.0)
            next_seq = max_seq_by_sid[sid] + 1
            max_seq_by_sid[sid] = next_seq

            db.add(Task(
                order_id=o.id,
                station_id=sid,
                status=TaskStatus.QUEUED,
                assigned_seq=next_seq,
                duration_sec=dur,
            ))
            assigned += 1
            load_by_sid[sid] = load_by_sid.get(sid, 0) + dur
            heappush(any_heap, (load_by_sid[sid], sid))

    db.commit()

    # After assignment, offer one per station
    offer_first_tasks(db)
    return {"assigned": assigned}


def offer_first_tasks(db: Session) -> None:
    """
    For each active station, if there is no OFFERED task currently,
    promote the first QUEUED task (lowest assigned_seq) to OFFERED.
    """
    stations = db.query(Station).filter_by(is_active=True).all()
    now = datetime.utcnow()

    for s in stations:
        # If one is already offered, skip
        already = (
            db.query(Task)
            .filter(and_(Task.station_id == s.id, Task.status == TaskStatus.OFFERED))
            .first()
        )
        if already:
            continue

        first_queued = (
            db.query(Task)
            .filter(and_(Task.station_id == s.id, Task.status == TaskStatus.QUEUED))
            .order_by(Task.assigned_seq.asc())
            .first()
        )
        if first_queued:
            first_queued.status = TaskStatus.OFFERED
            first_queued.offered_at = now

    db.commit()


def pop_next_for_station(db: Session, station_id: int) -> None:
    """Ensure the given station has an OFFERED task; idempotent."""
    offer_first_tasks(db)


# -------- NEW: Even load distribution logic --------
def distribute_even_load(db: Session) -> dict:
    """
    Distribute total packing load time (est_pack_time_sec) as evenly as possible across active stations.
    Does NOT modify DB; used only for reporting/visualization.
    Returns dict with stations + overall total.
    """
    stations: List[Station] = db.query(Station).filter_by(is_active=True).all()
    orders: List[Order] = db.query(Order).all()

    if not stations or not orders:
        return {"stations": [], "overall_total": 0}

    # Convert orders to minutes
    order_times = [
        {"order_id": o.id, "time": max(int((o.est_pack_time_sec or 1) // 60), 1)}
        for o in orders
    ]

    # Sort longest first
    order_times.sort(key=lambda x: x["time"], reverse=True)

    # Init station loads
    station_loads = {s.id: {"orders": [], "total_time": 0} for s in stations}

    # Greedy allocation
    for o in order_times:
        target_sid = min(station_loads, key=lambda sid: station_loads[sid]["total_time"])
        station_loads[target_sid]["orders"].append(o)
        station_loads[target_sid]["total_time"] += o["time"]

    result = {
        "stations": [
            {
                "station": sid,
                "total_time": station_loads[sid]["total_time"],
                "orders": station_loads[sid]["orders"],
            }
            for sid in station_loads
        ],
        "overall_total": sum(station_loads[sid]["total_time"] for sid in station_loads),
    }
    return result

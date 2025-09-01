# backend/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum as PyEnum
from db import Base


# -------------------------
# Task Status Enum
# -------------------------
class TaskStatus(PyEnum):
    PENDING = "PENDING"        # created, not assigned
    QUEUED = "QUEUED"          # assigned to a station queue
    OFFERED = "OFFERED"        # visible to station, waiting accept
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    LATE = "LATE"


# -------------------------
# Orders Table
# -------------------------
class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    order_id = Column(String, unique=True, index=True)
    items = Column(String)
    qty = Column(Integer)
    sku_mix = Column(Integer)
    # Allowed values we use in allocation: rush | expedited | standard
    priority = Column(String)

    # NEW: tag(s) used for specialized routing, e.g. "fragile", "cold-chain", "high-value", "none"
    # Comma-separated is fine (e.g., "fragile, high-value"). "none" or empty => treated as normal.
    special_instruction = Column(String, default="none")

    est_pack_time_sec = Column(Integer)
    due_by = Column(String)  # ISO 8601 string (IST or general text timestamp)


# -------------------------
# Stations Table
# -------------------------
class Station(Base):
    __tablename__ = "stations"

    id = Column(Integer, primary_key=True)
    station_code = Column(String, unique=True, index=True)  # e.g., ST-1
    display_name = Column(String, nullable=False)           # Station 1
    type = Column(String, default="normal")                 # normal | specialized
    capabilities = Column(String, default="")               # comma-separated list
    speed_factor = Column(Float, default=1.0)
    is_active = Column(Boolean, default=True)

    # helpful representation for frontend
    def to_dict(self):
        return {
            "station_code": self.station_code,
            "display_name": self.display_name,
            "type": self.type,
            "capabilities": self.capabilities.split(",") if self.capabilities else [],
            "is_active": self.is_active,
        }


# -------------------------
# Tasks Table
# -------------------------
class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"), index=True)
    station_id = Column(Integer, ForeignKey("stations.id"), index=True, nullable=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING)
    assigned_seq = Column(Integer, default=None)  # sequence within station queue
    duration_sec = Column(Integer)  # computed from est_pack_time and station speed

    offered_at = Column(DateTime)
    accepted_at = Column(DateTime)
    started_at = Column(DateTime)
    due_at = Column(DateTime)
    completed_at = Column(DateTime)

    order = relationship("Order")
    station = relationship("Station")

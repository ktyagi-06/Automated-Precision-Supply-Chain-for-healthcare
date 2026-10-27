"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml"
DATA_DIR = ROOT_DIR / "data"

# The backend reuses the ML package (features, forecasting, training).
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # simulate: generate live sales | ingest: only data POSTed to /api/ingest | off: static
    live_mode: str
    autostart: bool              # run the background tick loop
    tick_seconds: float          # real seconds between ticks
    sim_minutes_per_tick: float  # simulated minutes that pass per tick
    auto_reorder: bool           # place orders automatically at the reorder point
    retrain_every_days: int      # retrain after this many closed days (0 = never)
    retrain_fast: bool           # fewer trees for background retrains
    seed: int
    sim_start_date: date | None  # first live day (default: today, UTC)
    model_dir: Path
    db_path: str
    api_key: str | None          # if set, /api/ingest/* requires header X-API-Key
    cors_origins: list[str]

    @staticmethod
    def from_env() -> "Settings":
        mode = os.getenv("LIVE_MODE", "simulate").strip().lower()
        if mode not in ("simulate", "ingest", "off"):
            raise ValueError("LIVE_MODE must be simulate, ingest or off")
        start = os.getenv("SIM_START_DATE")
        return Settings(
            live_mode=mode,
            autostart=_bool("LIVE_AUTOSTART", True),
            tick_seconds=float(os.getenv("TICK_SECONDS", "5")),
            sim_minutes_per_tick=float(os.getenv("SIM_MINUTES_PER_TICK", "60")),
            auto_reorder=_bool("AUTO_REORDER", True),
            retrain_every_days=int(os.getenv("RETRAIN_EVERY_DAYS", "7")),
            retrain_fast=_bool("RETRAIN_FAST", True),
            seed=int(os.getenv("SIM_SEED", "7")),
            sim_start_date=date.fromisoformat(start) if start else None,
            model_dir=Path(os.getenv("MODEL_DIR", str(ROOT_DIR / "Models"))),
            db_path=os.getenv("LIVE_DB_PATH", str(ROOT_DIR / "backend" / "state" / "live.db")),
            api_key=os.getenv("API_KEY") or None,
            cors_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()],
        )

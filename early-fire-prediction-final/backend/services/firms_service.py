"""
Real satellite fire detections from NASA FIRMS (VIIRS, near real-time).

Setup (one time, free):
  1. Get a MAP_KEY:  https://firms.modaps.eosdis.nasa.gov/api/map_key/
  2. Put it in backend/.env as   FIRMS_MAP_KEY=your_key_here
     (or set the FIRMS_MAP_KEY environment variable).

Optional settings (same .env file):
  FIRMS_BBOX=-123.5,36.0,-120.0,38.5   west,south,east,north  (default: SF Bay / Santa Clara region)
  FIRMS_DAYS=2                          how many recent days to include (1-5)

TWO MODES
  live    real-time detections from the FIRMS API (needs FIRMS_MAP_KEY).
  replay  real, historical detections from backend/data/Wildfire_Data_2020-2023.csv
          (the same NASA FIRMS VIIRS export your dataset was built from). No key needed.
          Great for demos: e.g. 2020-08-20 was the peak of the CZU/SCU Lightning fires.

  HOTSPOT_MODE=auto        auto (default): live if a key is set, otherwise replay
                           live | replay to force one
  HOTSPOT_REPLAY_DATE=2020-08-20   which day to replay (YYYY-MM-DD)
Any request can also pass ?date=YYYY-MM-DD to replay that specific day.
"""

import csv
import io
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import requests

FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}"
SOURCES = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]
DEFAULT_BBOX = "-123.5,36.0,-120.0,38.5"
CACHE_TTL_SECONDS = 600  # satellites pass every few hours; no point hammering the API

ARCHIVE_PATH = Path(__file__).resolve().parent.parent / "data" / "Wildfire_Data_2020-2023.csv"
DEFAULT_REPLAY_DATE = "2020-08-20"

_cache = {"t": 0.0, "data": None}


def _load_dotenv():
    """Tiny .env reader so no extra dependency is needed."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _normalise_confidence(raw: str):
    """VIIRS uses l/n/h (low/nominal/high); MODIS uses 0-100. Returns (label, keep?)."""
    c = (raw or "").strip().lower()
    if c in ("l", "low"):
        return "low", False          # drop low-confidence detections (often false alarms)
    if c in ("n", "nominal"):
        return "nominal", True
    if c in ("h", "high"):
        return "high", True
    try:
        n = float(c)
        return ("high" if n >= 80 else "nominal" if n >= 30 else "low"), n >= 30
    except ValueError:
        return c or "unknown", True


def parse_firms_csv(text: str):
    """Parse FIRMS CSV text into hotspot dicts. Raises ValueError on API error messages."""
    text = text.strip()
    if not text.lower().startswith("latitude"):
        raise ValueError(f"FIRMS error: {text[:200] or 'empty response'}")

    hotspots = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            conf_label, keep = _normalise_confidence(row.get("confidence"))
            if not keep:
                continue
            frp = float(row.get("frp") or 0.0)
            hotspots.append({
                "lat": float(row["latitude"]),
                "lng": float(row["longitude"]),
                "frp": round(frp, 2),                      # fire radiative power, MW
                "confidence": conf_label,
                "acq_date": row.get("acq_date", ""),
                "acq_time": (row.get("acq_time") or "").zfill(4),   # HHMM UTC
                "satellite": row.get("satellite", ""),
                "daynight": row.get("daynight", ""),
                # 0.25-1.0 heat intensity: stronger fires glow brighter
                "intensity": round(max(0.25, min(1.0, frp / 30.0)), 2),
            })
        except (KeyError, ValueError):
            continue  # skip malformed rows rather than failing the whole request
    return hotspots


@lru_cache(maxsize=1)
def _load_archive() -> dict:
    """Read the historical FIRMS export once -> {"YYYY-MM-DD": [hotspot, ...]}."""
    by_date: dict = {}
    if not ARCHIVE_PATH.exists():
        return by_date
    with ARCHIVE_PATH.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                conf_label, keep = _normalise_confidence(row.get("confidence"))
                if not keep:
                    continue
                d = datetime.strptime(row["acq_date"].strip(), "%m/%d/%y").date().isoformat()
                frp = float(row.get("frp") or 0.0)
                by_date.setdefault(d, []).append({
                    "lat": float(row["latitude"]), "lng": float(row["longitude"]),
                    "frp": round(frp, 2), "confidence": conf_label,
                    "acq_date": d, "acq_time": (row.get("acq_time") or "").strip().zfill(4),
                    "satellite": "VIIRS", "daynight": row.get("daynight", ""),
                    "intensity": round(max(0.25, min(1.0, frp / 30.0)), 2),
                })
            except (KeyError, ValueError):
                continue
    return by_date


def list_archive_dates() -> list:
    """[{"date": "2020-08-20", "count": 1203}, ...] oldest first."""
    return [{"date": d, "count": len(v)} for d, v in sorted(_load_archive().items())]


def get_replay(date_iso: str) -> dict:
    archive = _load_archive()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not archive:
        return {"source": "error", "count": 0, "hotspots": [], "fetched_at": fetched_at, "bbox": None,
                "message": f"Archive file not found at {ARCHIVE_PATH}. Copy Wildfire_Data_2020-2023.csv "
                           "into backend/data/, or set FIRMS_MAP_KEY for live data."}
    spots = sorted(archive.get(date_iso, []), key=lambda h: h["frp"], reverse=True)
    return {
        "source": f"NASA FIRMS archive replay ({date_iso})",
        "mode": "replay",
        "count": len(spots),
        "hotspots": spots,
        "fetched_at": fetched_at,
        "bbox": None,
        "message": "Historical detections, not live. Set FIRMS_MAP_KEY for real-time data."
                   if spots else f"No detections in the archive on {date_iso}.",
    }


def get_hotspots(force: bool = False, date: str | None = None) -> dict:
    """Return {"source", "count", "hotspots", "message", ...}. Never raises."""
    _load_dotenv()
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    mode = os.environ.get("HOTSPOT_MODE", "auto").strip().lower()

    if date:                                              # explicit day -> always replay
        return get_replay(date)
    if mode == "replay" or (mode == "auto" and not key):
        return get_replay(os.environ.get("HOTSPOT_REPLAY_DATE", DEFAULT_REPLAY_DATE).strip())
    return _get_live(force)


def _get_live(force: bool = False) -> dict:
    """Live FIRMS API detections. Never raises."""
    now = time.time()
    if not force and _cache["data"] is not None and now - _cache["t"] < CACHE_TTL_SECONDS:
        return _cache["data"]

    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    bbox = os.environ.get("FIRMS_BBOX", DEFAULT_BBOX).strip()
    days = os.environ.get("FIRMS_DAYS", "2").strip()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if not key:
        result = {"source": "none", "count": 0, "hotspots": [], "fetched_at": fetched_at, "bbox": bbox,
                  "message": "FIRMS_MAP_KEY not set. Add it to backend/.env (free key: "
                             "https://firms.modaps.eosdis.nasa.gov/api/map_key/)."}
        return result  # not cached, so adding the key takes effect immediately

    all_spots, errors = [], []
    seen = set()
    for source in SOURCES:
        try:
            resp = requests.get(FIRMS_URL.format(key=key, source=source, bbox=bbox, days=days), timeout=20)
            resp.raise_for_status()
            for h in parse_firms_csv(resp.text):
                ident = (round(h["lat"], 4), round(h["lng"], 4), h["acq_date"], h["acq_time"])
                if ident not in seen:          # two satellites can see the same fire
                    seen.add(ident)
                    all_spots.append(h)
        except Exception as exc:
            errors.append(f"{source}: {exc}")

    if not all_spots and errors:
        # Total failure: report it, don't cache, don't pretend there are zero fires.
        return {"source": "error", "count": 0, "hotspots": [], "fetched_at": fetched_at, "bbox": bbox,
                "message": "; ".join(errors)}

    all_spots.sort(key=lambda h: h["frp"], reverse=True)
    result = {
        "source": "NASA FIRMS (VIIRS SNPP + NOAA-20, near real-time)",
        "count": len(all_spots),
        "hotspots": all_spots,
        "fetched_at": fetched_at,
        "bbox": bbox,
        "message": "; ".join(errors) if errors else None,
    }
    _cache.update(t=now, data=result)
    return result

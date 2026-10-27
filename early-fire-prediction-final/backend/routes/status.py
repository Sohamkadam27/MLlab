import json
import os
from pathlib import Path

from fastapi import APIRouter

from services.firms_service import ARCHIVE_PATH, _load_dotenv
from services.model_service import get_model_info

router = APIRouter(tags=["Status"])
AUDIT_PATH = Path(__file__).resolve().parent.parent / "model" / "audit.json"


@router.get("/status")
def status():
    """One-glance health check: is the model loaded, which hotspot mode is active, etc."""
    _load_dotenv()
    problems = []
    try:
        model = get_model_info()
    except Exception as exc:                      # model or model_config.json missing
        model, problems = None, [f"Model not ready: {exc}"]

    key_set = bool(os.environ.get("FIRMS_MAP_KEY", "").strip())
    mode = os.environ.get("HOTSPOT_MODE", "auto").strip().lower()
    effective = "replay" if (mode == "replay" or (mode == "auto" and not key_set)) else "live"
    if effective == "replay" and not ARCHIVE_PATH.exists():
        problems.append(f"Replay mode needs {ARCHIVE_PATH} (copy Wildfire_Data_2020-2023.csv there).")

    audit = None
    if AUDIT_PATH.exists():
        try:
            audit = json.loads(AUDIT_PATH.read_text())
        except Exception:
            audit = None

    return {
        "ok": not problems,
        "audit": audit,
        "problems": problems,
        "model": model,
        "hotspots": {"mode": effective, "firms_key_configured": key_set,
                     "replay_date": os.environ.get("HOTSPOT_REPLAY_DATE", "2020-08-20")},
        "region": "Santa Clara County foothills / Diablo Range, California",
    }

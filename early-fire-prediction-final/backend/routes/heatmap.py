from fastapi import APIRouter, Query

from services.firms_service import get_hotspots, list_archive_dates

router = APIRouter(tags=["Heatmap"])


@router.get("/heatmap")
def get_heatmap(date: str | None = Query(None, description="Replay a past day, YYYY-MM-DD")):
    """Real NASA FIRMS detections as [[lat, lng, intensity], ...] for the heat layer."""
    data = get_hotspots(date=date)
    return [[h["lat"], h["lng"], h["intensity"]] for h in data["hotspots"]]


@router.get("/hotspots")
def get_hotspot_details(date: str | None = Query(None, description="Replay a past day, YYYY-MM-DD")):
    """Same detections with details (time, confidence, fire power) + data-source status."""
    return get_hotspots(date=date)


@router.get("/hotspots/dates")
def get_hotspot_dates():
    """Days available for replay, with detection counts (handy for picking a demo day)."""
    return list_archive_dates()

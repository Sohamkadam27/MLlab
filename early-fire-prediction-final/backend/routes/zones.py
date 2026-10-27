from fastapi import APIRouter

from services.live_weather import fetch_many
from services.model_service import get_config, predict_fire_risk

router = APIRouter(tags=["Zones"])

# IMPORTANT: the model was trained on ONE location's weather: the ERA5 archive at
# (37.3N, 121.5W), per final_dataset_centroid.ipynb in the original repo. A day was labelled
# "fire" if FIRMS detected a hotspot inside 37.1-37.5N, 121.8-121.2W. So the model answers
# "is there fire activity in this Diablo Range box today?" -- scores for the zones below are
# regional-weather estimates, not independent per-site predictions. `vegetation` is a
# static land-cover estimate used for display only (it is not a model feature).
ZONES = [
    {"name": "Mount Hamilton",            "state": "California", "lat": 37.341, "lng": -121.643, "vegetation": 55},
    {"name": "Henry Coe State Park",      "state": "California", "lat": 37.170, "lng": -121.550, "vegetation": 70},
    {"name": "Almaden Quicksilver",       "state": "California", "lat": 37.180, "lng": -121.860, "vegetation": 75},
    {"name": "Mount Umunhum / Sierra Azul","state": "California", "lat": 37.160, "lng": -121.900, "vegetation": 85},
    {"name": "Uvas Canyon",               "state": "California", "lat": 37.070, "lng": -121.800, "vegetation": 90},
    {"name": "Santa Teresa Hills",        "state": "California", "lat": 37.200, "lng": -121.800, "vegetation": 50},
    {"name": "Alum Rock Park",            "state": "California", "lat": 37.395, "lng": -121.790, "vegetation": 60},
    {"name": "Castle Rock (Santa Cruz Mtns)", "state": "California", "lat": 37.230, "lng": -122.100, "vegetation": 95},
    {"name": "Monte Bello Preserve",      "state": "California", "lat": 37.320, "lng": -122.150, "vegetation": 80},
    {"name": "Los Gatos Hills",           "state": "California", "lat": 37.200, "lng": -121.990, "vegetation": 85},
]


@router.get("/zones")
def get_zones():
    """Real-time conditions + model risk for each monitored zone (all 10 real features)."""
    train_pressure = float(get_config()["imputation"]["surface_pressure"]["value"])
    fetched = fetch_many([(z["lat"], z["lng"]) for z in ZONES], train_pressure)

    results = []
    for zone, (live, error) in zip(ZONES, fetched):
        item = {
            "name": zone["name"], "location": zone["name"], "state": zone["state"],
            "lat": zone["lat"], "lng": zone["lng"], "vegetation": zone["vegetation"],
            "live_weather": live is not None,
        }
        if live is None:
            # No fake numbers: if live data is unavailable, say so and give no score.
            item.update(risk_score=None, risk_level=None, error=error)
        else:
            f = live["features"]
            risk = predict_fire_risk(
                f["temperature_2m"], f["relative_humidity_2m"], f["wind_speed_10m"],
                zone["vegetation"], full_features=f,
            )
            item.update(
                temperature=round(f["temperature_2m"], 1),
                humidity=round(f["relative_humidity_2m"], 1),
                wind=round(f["wind_speed_10m"], 1),
                soil_moisture=round(f["soil_moisture_0_to_7cm"], 3),
                features={k: round(v, 3) for k, v in f.items()},
                observed_at=live["observed_at"],
                risk=risk["risk_score"], **risk,
            )
        results.append(item)
    return results

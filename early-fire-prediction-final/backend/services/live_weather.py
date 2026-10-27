"""
Fetch REAL-TIME values for all 10 model features from Open-Meteo (free, no key).

Open-Meteo's forecast API names soil layers differently from the ERA5 data the
model was trained on, so we map them:

  model feature                  <- Open-Meteo variable(s)
  soil_temperature_0_to_7cm      <- soil_temperature_6cm
  soil_temperature_7_to_28cm     <- soil_temperature_18cm
  soil_moisture_0_to_7cm         <- depth-weighted mean of 0_to_1, 1_to_3, 3_to_9 cm layers
  soil_moisture_7_to_28cm        <- depth-weighted mean of 3_to_9, 9_to_27, 27_to_81 cm layers
  surface_pressure               <- pressure_msl, rescaled to the training site's elevation
                                    (see _site_pressure; raw surface pressure depends on
                                    each zone's altitude and would push the model out of range)

These are close approximations, not identical measurements: the forecast model
and ERA5-Land differ slightly, so treat scores as indicative.
"""

import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover",
    "wind_speed_10m", "pressure_msl",
    "soil_temperature_6cm", "soil_temperature_18cm",
    "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm",
    "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm",
]

CACHE_TTL_SECONDS = 600  # 10 min — weather barely changes faster, and it spares the free API
_cache: dict = {}


def _site_pressure(pressure_msl: float, train_site_pressure: float) -> float:
    """
    Convert sea-level pressure to the pressure the model saw at its training site.
    The training site sat at roughly 690 m (median surface pressure ~932 hPa), so
    we scale by (training median / standard sea-level pressure). This keeps the
    weather-driven part of the signal (highs/lows moving through) while removing
    the effect of the zone's own altitude.
    """
    return pressure_msl * (train_site_pressure / 1013.25)


def _pick(hourly: dict, name: str, idx: int):
    v = hourly[name][idx]
    if v is None:
        raise ValueError(f"Open-Meteo returned no value for {name}")
    return float(v)


def parse_open_meteo(payload: dict, train_site_pressure: float, now_utc: datetime | None = None) -> dict:
    """Turn an Open-Meteo hourly payload into the 10 model features (+ metadata)."""
    hourly = payload["hourly"]
    times = hourly["time"]
    now = now_utc or datetime.now(timezone.utc)
    target = now.strftime("%Y-%m-%dT%H:00")
    idx = times.index(target) if target in times else min(now.hour, len(times) - 1)

    m01 = _pick(hourly, "soil_moisture_0_to_1cm", idx)
    m13 = _pick(hourly, "soil_moisture_1_to_3cm", idx)
    m39 = _pick(hourly, "soil_moisture_3_to_9cm", idx)
    m927 = _pick(hourly, "soil_moisture_9_to_27cm", idx)
    m2781 = _pick(hourly, "soil_moisture_27_to_81cm", idx)

    features = {
        "temperature_2m": _pick(hourly, "temperature_2m", idx),
        "relative_humidity_2m": _pick(hourly, "relative_humidity_2m", idx),
        "precipitation": _pick(hourly, "precipitation", idx),
        "surface_pressure": _site_pressure(_pick(hourly, "pressure_msl", idx), train_site_pressure),
        "cloud_cover": _pick(hourly, "cloud_cover", idx),
        "wind_speed_10m": _pick(hourly, "wind_speed_10m", idx),
        "soil_temperature_0_to_7cm": _pick(hourly, "soil_temperature_6cm", idx),
        "soil_temperature_7_to_28cm": _pick(hourly, "soil_temperature_18cm", idx),
        # layer thickness-weighted means over the 0-7 cm and 7-28 cm windows
        "soil_moisture_0_to_7cm": (1 * m01 + 2 * m13 + 4 * m39) / 7.0,
        "soil_moisture_7_to_28cm": (2 * m39 + 18 * m927 + 1 * m2781) / 21.0,
    }
    return {
        "features": features,
        "observed_at": times[idx],
        "elevation_m": payload.get("elevation"),
    }


def fetch_live_features(lat: float, lng: float, train_site_pressure: float) -> dict:
    key = (round(lat, 3), round(lng, 3))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_SECONDS:
        return hit[1]

    resp = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": lat, "longitude": lng,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": "GMT", "forecast_days": 1,
        },
        timeout=8,
    )
    resp.raise_for_status()
    result = parse_open_meteo(resp.json(), train_site_pressure)
    _cache[key] = (time.time(), result)
    return result


def fetch_many(points: list[tuple[float, float]], train_site_pressure: float, workers: int = 6):
    """Fetch several locations in parallel. Returns a list of (result | None, error | None)."""
    def one(pt):
        try:
            return fetch_live_features(pt[0], pt[1], train_site_pressure), None
        except Exception as exc:  # network, bad payload, rate limit...
            return None, str(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, points))

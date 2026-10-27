import { useEffect, useRef, useState, useCallback } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet.heat";

const BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const ZONES_API = `${BASE_URL}/zones`;
const HOTSPOTS_API = `${BASE_URL}/hotspots`;
const REFRESH_MS = 5 * 60 * 1000; // re-run the live analysis every 5 minutes
const MAX_FLAME_MARKERS = 400;

const LEVEL_COLOR = { High: "#cc1111", Medium: "#ff8800", Low: "#22bb66" };
// The backend decides the level from cutoffs learned on held-out data; the browser just displays it.
const zoneLevel = (z) => z.risk_level ?? (z.risk_score > 75 ? "High" : z.risk_score > 40 ? "Medium" : "Low");

function makeZoneIcon(color, isHigh) {
  const pulse = isHigh
    ? `<div style="position:absolute;inset:-8px;border-radius:50%;border:2px solid ${color};opacity:0.5;animation:markerPulse 1.4s ease-out infinite;"></div>`
    : "";
  return L.divIcon({
    className: "",
    html: `<div style="position:relative;width:20px;height:20px;">
      ${pulse}
      <div style="width:20px;height:20px;border-radius:50%;background:${color};
        border:3px solid rgba(255,255,255,0.9);box-shadow:0 0 10px ${color}bb;"></div>
    </div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  });
}

// Satellite detections use a flame so they can never be confused with the round ML risk markers.
const FLAME_ICON = L.divIcon({
  className: "",
  html: `<div style="font-size:18px;line-height:18px;filter:drop-shadow(0 1px 2px rgba(0,0,0,.55));">🔥</div>`,
  iconSize: [18, 18],
  iconAnchor: [9, 16],
});

function topDriver(contributions) {
  if (!contributions) return null;
  const [name, pct] = Object.entries(contributions).sort((a, b) => b[1] - a[1])[0] ?? [];
  return name && pct > 0 ? `${name} (${pct}%)` : null;
}

function zonePopup(z) {
  const score = Math.round(z.risk_score);
  const color = LEVEL_COLOR[zoneLevel(z)];
  const driver = topDriver(z.contributions);
  const cell = (label, value) => `
    <div style="background:#f4f7f5;border-radius:6px;padding:7px 9px">
      <div style="font-size:9px;color:#7a8f88;letter-spacing:.08em;margin-bottom:2px">${label}</div>
      <div style="font-weight:700;font-size:13px;color:#1a2e28">${value}</div>
    </div>`;
  return `
    <div style="font-family:Inter,sans-serif;min-width:210px;padding:4px 2px">
      <div style="font-size:10px;letter-spacing:.12em;color:#888;margin-bottom:6px;font-weight:700">LIVE ML RISK</div>
      <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px">
        <span style="font-size:36px;font-weight:800;color:${color};line-height:1">${score}</span>
        <span style="font-size:11px;font-weight:800;color:${color};letter-spacing:.08em">${zoneLevel(z).toUpperCase()}</span>
      </div>
      <div style="font-size:12px;font-weight:700;color:#1a2e28;margin-bottom:2px">📍 ${z.name}</div>
      <div style="font-size:11px;color:#5a6f68;margin-bottom:10px">Live data ${z.observed_at ? z.observed_at.slice(11, 16) + " UTC" : ""}</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px">
        ${cell("TEMP", `🌡 ${z.temperature}°C`)}
        ${cell("HUMIDITY", `💧 ${z.humidity}%`)}
        ${cell("WIND", `🌬 ${z.wind} km/h`)}
        ${cell("SOIL MOISTURE", `🌱 ${z.soil_moisture}`)}
      </div>
      ${driver ? `<div style="font-size:11px;color:#1a2e28;background:#f0f5f2;border-radius:6px;padding:7px 10px">
        Biggest driver: <b>${driver}</b></div>` : ""}
    </div>`;
}

function hotspotPopup(h) {
  const hh = (h.acq_time || "0000").padStart(4, "0");
  return `
    <div style="font-family:Inter,sans-serif;min-width:170px">
      <div style="font-size:10px;letter-spacing:.12em;color:#888;font-weight:700;margin-bottom:4px">SATELLITE DETECTION</div>
      <div style="font-weight:800;color:#cc1111;font-size:14px;margin-bottom:6px">🔥 Active fire</div>
      <div style="font-size:12px;line-height:1.6;color:#1a2e28">
        Detected: <b>${h.acq_date} ${hh.slice(0, 2)}:${hh.slice(2)} UTC</b><br/>
        Confidence: <b>${h.confidence}</b><br/>
        Fire power: <b>${h.frp} MW</b><br/>
        ${h.daynight === "N" ? "Night pass" : "Day pass"}
      </div>
    </div>`;
}

const bannerBase = {
  borderRadius: 8, padding: "7px 12px", fontSize: 11, fontWeight: 600, lineHeight: 1.4, maxWidth: 320,
};

export default function FireHeatMap({ onAlert, onStats, zoomTarget }) {
  const mapRef = useRef(null);
  const mapInstance = useRef(null);
  const heatLayer = useRef(null);
  const hotspotLayer = useRef(null);
  const zoneMarkers = useRef([]);
  const framed = useRef(false);
  const onAlertRef = useRef(onAlert);
  onAlertRef.current = onAlert;
  const onStatsRef = useRef(onStats);
  onStatsRef.current = onStats;

  const [status, setStatus] = useState("idle");
  const [btnLabel, setBtnLabel] = useState("Run Satellite Analysis");
  const [zones, setZones] = useState([]);
  const [zoneError, setZoneError] = useState("");
  const [hotspotInfo, setHotspotInfo] = useState(null);
  const [updatedAt, setUpdatedAt] = useState(null);
  const [visible, setVisible] = useState(false);

  // init map once
  useEffect(() => {
    if (mapInstance.current) return;
    mapInstance.current = L.map(mapRef.current, {
      center: [37.3, -121.7], zoom: 9, zoomControl: false,
    });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap contributors",
    }).addTo(mapInstance.current);
    L.control.zoom({ position: "topright" }).addTo(mapInstance.current);
    setTimeout(() => setVisible(true), 50);
    runAnalysis(true);
    return () => { mapInstance.current?.remove(); mapInstance.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!zoomTarget || !mapInstance.current) return;
    mapInstance.current.flyTo([zoomTarget.lat, zoomTarget.lng], 10, { duration: 1.4 });
  }, [zoomTarget]);

  const runAnalysis = useCallback(async (reframe = false) => {
    const map = mapInstance.current;
    if (!map) return;
    setStatus("loading");
    setBtnLabel("Analyzing Satellite Data...");

    const getJSON = async (url) => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`${url.replace(BASE_URL, "")} returned HTTP ${res.status}`);
      return res.json();
    };
    const [hsRes, zoneRes] = await Promise.allSettled([getJSON(HOTSPOTS_API), getJSON(ZONES_API)]);

    // ---- 1. Satellite hotspots (real NASA FIRMS data, live or replayed) ----
    const hs = hsRes.status === "fulfilled"
      ? hsRes.value
      : { source: "error", mode: "none", count: 0, hotspots: [],
          message: "Cannot reach the backend. Is uvicorn running on " + BASE_URL + "?" };
    setHotspotInfo(hs);

    if (heatLayer.current) map.removeLayer(heatLayer.current);
    if (hotspotLayer.current) map.removeLayer(hotspotLayer.current);
    heatLayer.current = L.heatLayer(hs.hotspots.map(h => [h.lat, h.lng, h.intensity]), {
      radius: 24, blur: 20, maxZoom: 10, max: 2.0, minOpacity: 0.25,
      gradient: { 0.3: "#ffe066", 0.6: "#ff8800", 1.0: "#cc1111" },
    }).addTo(map);

    const flameLimit = hs.count > 300 ? 120 : MAX_FLAME_MARKERS; // keep dense fires readable
    const strongest = [...hs.hotspots].sort((a, b) => b.frp - a.frp).slice(0, flameLimit);
    hotspotLayer.current = L.layerGroup(
      strongest.map(h => L.marker([h.lat, h.lng], { icon: FLAME_ICON })
        .bindPopup(hotspotPopup(h), { maxWidth: 240 }))
    ).addTo(map);

    // ---- 2. Live ML risk zones (NO fallback data: if it fails we say so) ----
    zoneMarkers.current.forEach(m => map.removeLayer(m));
    zoneMarkers.current = [];

    let scored = [];
    let zErr = "";
    if (zoneRes.status === "rejected") {
      zErr = zoneRes.reason?.message?.includes("Failed to fetch")
        ? "Cannot reach the backend." : (zoneRes.reason?.message ?? "Zone request failed.");
    } else if (!Array.isArray(zoneRes.value) || zoneRes.value.length === 0) {
      zErr = "The backend returned no zones.";
    } else {
      scored = zoneRes.value.filter(z => typeof z.risk_score === "number");
      if (scored.length === 0) {
        zErr = `Live weather unavailable for all zones (${zoneRes.value[0]?.error ?? "unknown error"}).`;
      }
    }
    setZoneError(zErr);
    setZones(scored);

    scored.forEach((z) => {
      const color = LEVEL_COLOR[zoneLevel(z)];
      const marker = L.marker([z.lat, z.lng], { icon: makeZoneIcon(color, zoneLevel(z) === "High") })
        .bindPopup(zonePopup(z), { maxWidth: 270 })
        .addTo(map);
      zoneMarkers.current.push(marker);
    });

    // ---- 3. Frame the map once (or when the button is pressed), not on every auto-refresh ----
    if (reframe === true || !framed.current) {
      const pts = scored.length ? scored.map(z => [z.lat, z.lng])
        : hs.hotspots.map(h => [h.lat, h.lng]);
      if (pts.length) {
        map.fitBounds(L.latLngBounds(pts).pad(0.3), { maxZoom: 10, animate: true });
        framed.current = true;
      }
    }

    // ---- 4. Alert hook ----
    const top = scored.reduce((a, b) => (!a || b.risk_score > a.risk_score ? b : a), null);
    if (top && zoneLevel(top) === "High") {
      onAlertRef.current?.({
        type: "critical",
        message: `HIGH risk at ${top.name} (${Math.round(top.risk_score)}/100)`,
        hotspots: hs.count, maxRisk: Math.round(top.risk_score),
        time: new Date().toLocaleTimeString(),
        zoomTarget: { lat: top.lat, lng: top.lng }, location: top.name,
      });
    }

    onStatsRef.current?.({
      highZones: scored.filter(z => zoneLevel(z) === "High").length,
      mediumZones: scored.filter(z => zoneLevel(z) === "Medium").length,
      zones: scored.length,
      hotspots: hs.count,
    });

    const bothFailed = hsRes.status === "rejected" && zoneRes.status === "rejected";
    setUpdatedAt(new Date());
    setStatus(bothFailed ? "error" : "success");
    setBtnLabel(bothFailed ? "Run Satellite Analysis" : "Updated ✓");
    setTimeout(() => setBtnLabel("Run Satellite Analysis"), 3000);
  }, []);

  // real-time refresh
  useEffect(() => {
    const id = setInterval(() => runAnalysis(false), REFRESH_MS);
    return () => clearInterval(id);
  }, [runAnalysis]);

  const legendRows = [
    [LEVEL_COLOR.High, "High risk", "Model flags fire-prone conditions", null],
    [LEVEL_COLOR.Medium, "Medium risk", "Borderline (rare)", null],
    [LEVEL_COLOR.Low, "Low risk", "Model sees no fire-prone signal", null],
  ];
  const highCount = zones.filter(z => zoneLevel(z) === "High").length;
  const isReplay = hotspotInfo?.mode === "replay";
  const hotspotOk = hotspotInfo && hotspotInfo.source !== "none" && hotspotInfo.source !== "error";

  return (
    <div style={{ position: "relative", width: "100%", height: "100%",
      opacity: visible ? 1 : 0, transition: "opacity 0.6s ease" }}>

      <div style={{
        position: "absolute", top: 12, left: "50%", transform: "translateX(-50%)",
        zIndex: 1000, background: "rgba(10,40,28,0.88)", color: "#fff",
        borderRadius: 10, padding: "8px 18px", fontSize: 11, fontWeight: 700,
        letterSpacing: "0.12em", textAlign: "center", whiteSpace: "nowrap",
        border: "1px solid rgba(100,220,160,0.3)",
      }}>
        🔥 Live ML Fire Risk &nbsp;+&nbsp; NASA Satellite Hotspots
      </div>

      <div ref={mapRef} style={{ width: "100%", height: "100%" }} />

      {/* status chip */}
      {hotspotInfo && (
        <div style={{
          position: "absolute", top: 56, left: 14, zIndex: 1000,
          background: "rgba(10,50,34,0.92)", color: "#7effc8",
          borderRadius: 10, padding: "8px 14px", fontSize: 12, fontWeight: 700,
          border: "1px solid rgba(80,200,130,0.3)", maxWidth: 360,
        }}>
          🔥 Satellite hotspots: {hotspotInfo.count}
          {hotspotOk && (
            <span style={{
              marginLeft: 8, fontSize: 10, letterSpacing: "0.08em", borderRadius: 4, padding: "2px 6px",
              background: isReplay ? "#b45309" : "#15803d", color: "#fff",
            }}>
              {isReplay ? "REPLAY" : "LIVE"}
            </span>
          )}
          {updatedAt && (
            <span style={{ marginLeft: 10, fontWeight: 600, color: "#9fd8bd" }}>
              · {updatedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </span>
          )}
          <div style={{ marginTop: 4, fontSize: 11, fontWeight: 600, color: "#9fd8bd" }}>
            {zones.length} zone{zones.length !== 1 ? "s" : ""} scored live
            {highCount > 0 && <span style={{ marginLeft: 8, color: "#ff8a8a" }}>⚠ {highCount} high risk</span>}
          </div>
        </div>
      )}

      {/* legend */}
      <div style={{
        position: "absolute", bottom: 176, right: 14, zIndex: 1000, maxWidth: 260,
        background: "rgba(250,252,251,0.96)", borderRadius: 14,
        border: "1px solid #d4ddd8", padding: "12px 16px", minWidth: 200,
        boxShadow: "0 4px 16px rgba(0,0,0,0.1)",
      }}>
        <div style={{ fontWeight: 800, fontSize: 10, letterSpacing: "0.16em",
          color: "#4e625a", marginBottom: 10, borderBottom: "1px solid #e4ece8", paddingBottom: 8 }}>
          MAP LEGEND
        </div>
        <div style={{ fontSize: 10, color: "#7a8f88", fontWeight: 700, letterSpacing: "0.1em", marginBottom: 6 }}>
          LIVE RISK ZONES (round dots)
        </div>
        {legendRows
          .map(([c, label, sub, rate]) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 7 }}>
              <div style={{ width: 14, height: 14, borderRadius: "50%", flexShrink: 0, background: c,
                border: "2px solid rgba(255,255,255,0.8)", boxShadow: `0 0 6px ${c}88` }} />
              <div>
                <div style={{ fontWeight: 700, fontSize: 12, color: "#1a2e28" }}>{label}</div>
                <div style={{ fontSize: 10, color: "#8a9e96" }}>
                  {sub}
                </div>
              </div>
            </div>
          ))}
        <div style={{ fontSize: 10, color: "#a0651f", fontWeight: 600, lineHeight: 1.4, marginTop: 2 }}>
          Experimental model: indicative only, not a validated forecast.
        </div>
        <div style={{ fontSize: 10, color: "#7a8f88", fontWeight: 700, letterSpacing: "0.1em",
          margin: "10px 0 6px", borderTop: "1px solid #e4ece8", paddingTop: 8 }}>
          SATELLITE DETECTIONS (NASA FIRMS)
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
          <span style={{ fontSize: 16 }}>🔥</span>
          <div style={{ fontSize: 11, fontWeight: 700, color: "#1a2e28" }}>Detected fire (click for details)</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 48, height: 12, borderRadius: 4, flexShrink: 0,
            background: "linear-gradient(90deg, #ffe066, #ff8800, #cc1111)" }} />
          <div style={{ fontSize: 10, color: "#4e625a" }}>Heat = fire power</div>
        </div>
      </div>

      {/* banners + button */}
      <div style={{
        position: "absolute", bottom: 140, left: 14, zIndex: 1000,
        display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 8,
      }}>
        {status === "error" && (
          <div style={{ ...bannerBase, background: "rgba(180,20,20,0.94)", color: "#fff", fontWeight: 700 }}>
            Backend unreachable. Start it with: uvicorn main:app --reload
          </div>
        )}
        {hotspotInfo && !hotspotOk && status !== "error" && (
          <div style={{ ...bannerBase, background: "rgba(176,96,0,0.94)", color: "#fff" }}>
            ⚠ Hotspot data unavailable: {hotspotInfo.message}
          </div>
        )}
        {zoneError && status !== "error" && (
          <div style={{ ...bannerBase, background: "rgba(176,96,0,0.94)", color: "#fff" }}>
            ⚠ Risk zones hidden: {zoneError} No estimates are shown when live data is missing.
          </div>
        )}
        {isReplay && hotspotOk && (
          <div style={{ ...bannerBase, background: "rgba(120,70,10,0.92)", color: "#ffe9c2" }}>
            🕑 {hotspotInfo.source}. Real past detections, not live. Add FIRMS_MAP_KEY for live data.
          </div>
        )}
        {hotspotOk && hotspotInfo.count === 0 && !isReplay && (
          <div style={{ ...bannerBase, background: "rgba(10,50,34,0.92)", color: "#7effc8" }}>
            ✅ No active fires detected by satellites in this region (recent days)
          </div>
        )}
        <button
          onClick={() => runAnalysis(true)}
          disabled={status === "loading"}
          style={{
            border: 0, borderRadius: 12,
            background: status === "loading" ? "#1a6b4a" : "#0d4e37",
            color: "#fff", fontWeight: 700, padding: "12px 22px",
            cursor: status === "loading" ? "wait" : "pointer",
            display: "flex", alignItems: "center", gap: 8, fontSize: 13,
            boxShadow: "0 4px 14px rgba(0,0,0,0.25)", transition: "background 0.2s ease",
          }}
        >
          {status === "loading" && (
            <span style={{ width: 14, height: 14, border: "2px solid rgba(255,255,255,0.3)",
              borderTopColor: "#fff", borderRadius: "50%", display: "inline-block",
              animation: "spin 0.7s linear infinite" }} />
          )}
          {btnLabel}
        </button>
      </div>

      <style>{`
        @keyframes spin        { to { transform: rotate(360deg); } }
        @keyframes markerPulse { 0%{transform:scale(1);opacity:.6} 100%{transform:scale(2.4);opacity:0} }
      `}</style>
    </div>
  );
}

import { useCallback, useEffect, useMemo, useState } from "react";
import SidebarNav from "../components/SidebarNav";
import SystemStatus from "../components/SystemStatus";

const BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const NEARBY_KM = 25;

function distanceKm(lat1, lng1, lat2, lng2) {
  const rad = (d) => (d * Math.PI) / 180;
  const a = Math.sin(rad(lat2 - lat1) / 2) ** 2
    + Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(rad(lng2 - lng1) / 2) ** 2;
  return 6371 * 2 * Math.asin(Math.sqrt(a));
}

const levelOf = (s) => (s > 75 ? "HIGH RISK" : s >= 40 ? "MEDIUM RISK" : "LOW RISK");
const colorOf = (s) => (s < 40 ? "#22c55e" : s < 75 ? "#f59e0b" : "#ef4444");

function Reading({ label, value, unit }) {
  return (
    <div style={{ background: "#f4f7f5", borderRadius: 10, padding: "10px 12px" }}>
      <div style={{ fontSize: 10, letterSpacing: "0.1em", color: "#7a8f88", fontWeight: 700 }}>{label}</div>
      <div style={{ fontSize: 17, fontWeight: 800, color: "#1a2e28", marginTop: 2 }}>
        {value ?? "—"}<span style={{ fontSize: 11, fontWeight: 600, color: "#6b8078", marginLeft: 3 }}>{unit}</span>
      </div>
    </div>
  );
}

export default function PredictionPanel({ onNavigate }) {
  const [zones, setZones] = useState([]);
  const [info, setInfo] = useState(null);
  const [audit, setAudit] = useState(null);
  const [hotspots, setHotspots] = useState(null);
  const [selected, setSelected] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [displayScore, setDisplayScore] = useState(0);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    const get = async (path) => {
      const res = await fetch(`${BASE_URL}${path}`);
      if (!res.ok) throw new Error(`${path} returned HTTP ${res.status}`);
      return res.json();
    };
    const [zRes, iRes, hRes] = await Promise.allSettled([get("/zones"), get("/status"), get("/hotspots")]);

    if (iRes.status === "fulfilled") { setInfo(iRes.value.model); setAudit(iRes.value.audit?.corrected ?? null); }
    if (hRes.status === "fulfilled") setHotspots(hRes.value);

    if (zRes.status === "rejected") {
      setZones([]);
      setError(zRes.reason?.message?.includes("Failed to fetch")
        ? "Cannot reach the backend. Start it with: uvicorn main:app --reload"
        : zRes.reason.message);
    } else {
      const scored = zRes.value.filter((z) => typeof z.risk_score === "number");
      setZones(scored);
      if (scored.length === 0) {
        setError(`Live weather unavailable (${zRes.value[0]?.error ?? "unknown error"}). No estimate is shown without live data.`);
      } else {
        setSelected((cur) => (scored.some((z) => z.name === cur)
          ? cur
          : scored.reduce((a, b) => (b.risk_score > a.risk_score ? b : a)).name));
      }
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const zone = zones.find((z) => z.name === selected) ?? null;
  const score = zone ? Math.round(zone.risk_score) : null;
  const isExtreme = score !== null && score > 75;

  // count-up animation
  useEffect(() => {
    if (score === null) { setDisplayScore(0); return; }
    let current = 0;
    const inc = score / 50;
    const id = setInterval(() => {
      current += inc;
      if (current >= score) { current = score; clearInterval(id); }
      setDisplayScore(Math.round(current));
    }, 16);
    return () => clearInterval(id);
  }, [score]);

  const drivers = useMemo(() => {
    if (!zone?.contributions) return [];
    return Object.entries(zone.contributions).sort((a, b) => b[1] - a[1]);
  }, [zone]);

  const nearby = useMemo(() => {
    if (!zone || !hotspots?.hotspots) return null;
    return hotspots.hotspots.filter((h) => distanceKm(zone.lat, zone.lng, h.lat, h.lng) <= NEARBY_KM).length;
  }, [zone, hotspots]);

  const sortedZones = useMemo(() => [...zones].sort((a, b) => b.risk_score - a.risk_score), [zones]);
  const replayDate = hotspots?.mode === "replay"
    ? (hotspots.source?.match(/\d{4}-\d{2}-\d{2}/)?.[0] ?? "a past day") : "";

  const f = zone?.features ?? {};
  const color = colorOf(displayScore);
  const insight = zone
    ? `${zone.name} is rated ${levelOf(score).toLowerCase()} (${score}/100). `
      + (drivers[0] && drivers[0][1] > 0 ? `The biggest driver is ${drivers[0][0].toLowerCase()} (${drivers[0][1]}% of the score movement). ` : "")
      + `Live readings: ${zone.temperature}°C, ${zone.humidity}% humidity, ${zone.wind} km/h wind.`
    : "Select a zone to see its live analysis.";

  return (
    <div className={`layout prediction-layout ${isExtreme ? "prediction-extreme" : ""}`}>
      <SidebarNav active="prediction-panel" onNavigate={onNavigate} />

      <main className="content">
        <header className="topbar">
          <div className="left">
            <strong>Vigilant Curator</strong>
            <span className="link-like" onClick={() => onNavigate("fire-risk-map")}>Fire Risk Map</span>
            <span className="active-link">Prediction Panel</span>
          </div>
          <div className="right"><SystemStatus /></div>
        </header>

        <section className="prediction-wrap">
          {isExtreme && (
            <div className="prediction-alert-banner">⚠ HIGH FIRE RISK — {zone.name}</div>
          )}

          <div className="prediction-header">
            <span className="engine-tag">XGBOOST MODEL · LIVE CONDITIONS</span>
            <h2>Prediction Panel</h2>
            <p>
              Scores real, current weather and soil conditions for monitored zones in the Santa Clara
              foothills, using the model's 10 trained inputs.
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 6 }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "#7a6030", background: "rgba(120,90,20,0.08)",
                border: "1px solid rgba(120,90,20,0.18)", borderRadius: 6, padding: "3px 10px" }}>
                ⚠ Experimental model: scores are indicative, not a validated forecast
              </span>
              {audit && (
                <span style={{ fontSize: 11, fontWeight: 700, color: "#4e8a6a", background: "rgba(30,100,60,0.1)",
                  border: "1px solid rgba(30,100,60,0.2)", borderRadius: 6, padding: "3px 10px" }}>
                  Corrected check ({audit.test}): AUC {audit.roc_auc} vs {audit.season_only_auc} calendar-only
                </span>
              )}
            </div>
          </div>

          <div className="prediction-grid">
            {/* ── Left: zone + live readings ── */}
            <section className="params-card">
              <div className="params-head">
                <div>
                  <h3>Live Zone Analysis</h3>
                  <small style={{ display: "block", marginTop: 8 }}>REAL CONDITIONS · OPEN-METEO</small>
                </div>
                <div className="model-status-block">
                  <span className="model-status-label">DATA STATUS</span>
                  <div className="model-status-pill">
                    <span className="model-status-dot" />
                    <span className="model-status-text">{loading ? "Loading…" : error ? "Unavailable" : "Live"}</span>
                  </div>
                </div>
              </div>

              <div style={{ margin: "14px 0 6px", fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", color: "#5b7168" }}>
                MONITORED ZONE
              </div>
              <select
                value={selected}
                onChange={(e) => setSelected(e.target.value)}
                disabled={zones.length === 0}
                style={{ width: "100%", padding: "11px 12px", borderRadius: 10, border: "1px solid #cfdad4",
                  fontSize: 14, fontWeight: 600, background: "#fff", color: "#1a2e28" }}
              >
                {zones.length === 0 && <option value="">No live zones available</option>}
                {sortedZones.map((z) => (
                  <option key={z.name} value={z.name}>{z.name} — {Math.round(z.risk_score)}/100</option>
                ))}
              </select>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 16 }}>
                <Reading label="TEMPERATURE" value={f.temperature_2m} unit="°C" />
                <Reading label="HUMIDITY" value={f.relative_humidity_2m} unit="%" />
                <Reading label="WIND SPEED" value={f.wind_speed_10m} unit="km/h" />
                <Reading label="PRECIPITATION" value={f.precipitation} unit="mm" />
                <Reading label="CLOUD COVER" value={f.cloud_cover} unit="%" />
                <Reading label="PRESSURE (ADJUSTED)" value={f.surface_pressure} unit="hPa" />
                <Reading label="SOIL TEMP (0–7 cm)" value={f.soil_temperature_0_to_7cm} unit="°C" />
                <Reading label="SOIL MOISTURE (0–7 cm)" value={f.soil_moisture_0_to_7cm} unit="" />
              </div>

              {sortedZones.length > 0 && (
                <div style={{ marginTop: 20 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", color: "#5b7168", marginBottom: 8 }}>
                    ALL ZONES · LIVE NOW (click to inspect)
                  </div>
                  <div style={{ display: "grid", gap: 6 }}>
                    {sortedZones.map((z) => (
                      <button key={z.name} onClick={() => setSelected(z.name)} style={{
                        display: "flex", alignItems: "center", gap: 10, width: "100%", textAlign: "left",
                        padding: "8px 12px", borderRadius: 10, cursor: "pointer", fontSize: 13, fontWeight: 600,
                        color: "#1a2e28", background: z.name === selected ? "#eaf4ee" : "#fff",
                        border: z.name === selected ? "2px solid #0d4e37" : "1px solid #e0e8e3",
                      }}>
                        <span style={{ width: 10, height: 10, borderRadius: "50%", background: colorOf(z.risk_score), flexShrink: 0 }} />
                        <span style={{ flex: 1 }}>{z.name}</span>
                        <span style={{ fontWeight: 800 }}>{Math.round(z.risk_score)}</span>
                        <span style={{ fontSize: 10, color: "#6b8078", width: 52, textAlign: "right" }}>{z.risk_level?.toUpperCase()}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="predict-action">
                <div className="predict-action-info">
                  <p>
                    {zone?.observed_at
                      ? `Observed ${zone.observed_at.slice(0, 10)} ${zone.observed_at.slice(11, 16)} UTC. `
                      : ""}
                    Soil layers are approximated from the weather provider's layers and surface pressure is
                    rescaled to the training site, so treat scores as indicative.
                  </p>
                </div>
                <button className={`predict-btn ${isExtreme ? "danger-btn" : ""}`} onClick={load} disabled={loading}>
                  <span>{loading ? "Refreshing…" : "Refresh live data"}</span>
                </button>
              </div>

              {error && <small className="api-error">⚠ {error}</small>}
            </section>

            {/* ── Right: results ── */}
            <aside className="prediction-side">
              <article className={`risk-card ${isExtreme ? "risk-card-extreme" : ""}`}>
                <div className="pf-heading">
                  <span className="pf-heading-line" />
                  <h4>LIVE RISK SCORE</h4>
                  <span className="pf-heading-line" />
                </div>

                <div className={`pf-gauge-wrap${displayScore > 85 ? " risk-pulse" : displayScore > 75 ? " risk-glow-high" : ""}`}>
                  <svg className="pf-gauge-svg" viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="50" fill="none" stroke="#f5e0e0" strokeWidth="8" />
                    <circle cx="60" cy="60" r="50" fill="none" stroke={color} strokeWidth="8" strokeLinecap="round"
                      strokeDasharray={`${2 * Math.PI * 50}`}
                      strokeDashoffset={`${2 * Math.PI * 50 * (1 - displayScore / 100)}`}
                      transform="rotate(-90 60 60)"
                      style={{ transition: "stroke-dashoffset 0.8s ease, stroke 0.5s ease" }} />
                  </svg>
                  <div className="pf-gauge-inner">
                    <span className="pf-gauge-score" style={{ color }}>{score !== null ? displayScore : "--"}</span>
                    <small className="pf-gauge-label">RISK SCORE</small>
                  </div>
                </div>

                <div style={{ display: "flex", justifyContent: "center" }}>
                  <div className={`pf-pill ${displayScore > 75 ? "pf-pill--red" : displayScore >= 40 ? "pf-pill--orange" : "pf-pill--green"}`}>
                    <div className="pf-pill-body">
                      <span className="pf-pill-score">{score !== null ? displayScore : "--"} / 100 →</span>
                      <span className="pf-pill-level">{score !== null ? levelOf(score) : "--"}</span>
                    </div>
                  </div>
                </div>

                <div className="pf-contrib">
                  <div className="pf-contrib-head"><span>WHAT DRIVES THIS SCORE</span></div>
                  {drivers.map(([label, pct]) => (
                    <div key={label} className="pf-bar-row">
                      <div className="pf-bar-meta">
                        <span className="pf-bar-label">{label.toUpperCase()}</span>
                        <span className="pf-bar-pct">{pct}%</span>
                      </div>
                      <div className="pf-bar-track">
                        <div className="pf-bar-fill" style={{ width: `${pct}%`, transition: "width 0.4s ease" }} />
                      </div>
                    </div>
                  ))}
                  <small style={{ display: "block", marginTop: 8, color: "#7a8f88", lineHeight: 1.4 }}>
                    Measured by swapping each input group for a typical value and seeing how far the score moves.
                  </small>
                </div>
              </article>

              <article className={`mi-card ${isExtreme ? "mi-card--critical" : ""}`}>
                <div className="mi-header">
                  <div className="mi-header-left">
                    <span className="mi-title">{isExtreme ? "High Risk Insight" : "Model Insights"}</span>
                  </div>
                </div>
                <p className="mi-body">{insight}</p>
                <div className="mi-sub-grid">
                  <div className="mi-sub-card">
                    <div className="mi-sub-label">SATELLITE FIRES WITHIN {NEARBY_KM} KM</div>
                    <strong className="mi-sub-value">
                      {nearby === null ? "—" : `${nearby} detection${nearby !== 1 ? "s" : ""}`}
                    </strong>
                    {replayDate && (
                      <small style={{ display: "block", marginTop: 4, opacity: 0.85 }}>
                        Replay of {replayDate}, not today's fires
                      </small>
                    )}
                  </div>
                  <div className="mi-sub-card">
                    <div className="mi-sub-label">CORRECTED CHECK {audit ? `(${audit.test})` : ""}</div>
                    <strong className="mi-sub-value">{audit ? `AUC ${audit.roc_auc} (calendar-only ${audit.season_only_auc})` : "run audit_dataset.py"}</strong>
                  </div>
                </div>
              </article>

              <article className="region-card" style={{
                backgroundImage: "linear-gradient(135deg,#0d4e37,#1f7a56)", color: "#fff",
                padding: "18px 20px", display: "flex", alignItems: "flex-end",
                fontSize: 12, fontWeight: 700, letterSpacing: "0.1em", lineHeight: 1.5,
              }}>
                MONITORED REGION · SANTA CLARA FOOTHILLS · {zones.length} ZONE{zones.length !== 1 ? "S" : ""}
              </article>
            </aside>
          </div>
        </section>
      </main>
    </div>
  );
}

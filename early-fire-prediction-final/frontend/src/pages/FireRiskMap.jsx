import { useState } from "react";
import SidebarNav from "../components/SidebarNav";
import FireHeatMap from "../components/FireHeatMap";
import SystemStatus from "../components/SystemStatus";

export default function FireRiskMap({ onNavigate, onAlert, zoomTarget }) {
  const [stats, setStats] = useState(null);
  const pad = (n) => String(n).padStart(2, "0");

  return (
    <div className="layout map-layout">
      <SidebarNav active="fire-risk-map" onNavigate={onNavigate} />

      <main className="content">
        <header className="topbar">
          <div className="left">
            <strong>Vigilant Curator</strong>
            <span className="active-link">Fire Risk Map</span>
            <span className="link-like" onClick={() => onNavigate("prediction-panel")}>
              Prediction Panel
            </span>
          </div>
          <div className="right">
            <SystemStatus />
          </div>
        </header>

        <section className="map-screen" style={{ height: "calc(100vh - 56px - 42px)" }}>
          <FireHeatMap onAlert={onAlert} onStats={setStats} zoomTarget={zoomTarget} />

          <div className="status-dock" style={{ left: "auto" }}>
            <div className="status-group" style={{ borderRight: 0, paddingRight: 0 }}>
              <span className="status-icon">⚠</span>
              <div>
                <small>HIGH-RISK ZONES</small>
                <h3>{stats ? pad(stats.highZones) : "--"}</h3>
                <strong>{stats ? `${stats.mediumZones} MEDIUM` : "LOADING"}</strong>
              </div>
            </div>
          </div>
        </section>

        <footer className="map-footer">
          <span>VIGILANT CURATOR | WILDFIRE EARLY-WARNING PROTOTYPE</span>
          <span style={{ color: "#7aab96", fontWeight: 700 }}>
            DATA: NASA FIRMS (VIIRS) hotspots + Open-Meteo weather · MODEL: XGBoost
          </span>
          <span>REGION: SANTA CLARA FOOTHILLS, CA</span>
        </footer>
      </main>
    </div>
  );
}

import { useState, useEffect } from "react";

const BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

// "online" | "degraded" | "offline" | "checking"
export default function SystemStatus() {
  const [state, setState] = useState("checking");
  const [checkedAt, setCheckedAt] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch(`${BASE_URL}/status`);
        if (!res.ok) throw new Error(String(res.status));
        const data = await res.json();
        if (!cancelled) setState(data.ok ? "online" : "degraded");
      } catch {
        if (!cancelled) setState("offline");
      }
      if (!cancelled) setCheckedAt(new Date());
    };
    check();
    const id = setInterval(check, 30000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const palette = {
    online:   { dot: "#22c55e", ring: "#4ade80", text: "System Active" },
    degraded: { dot: "#f59e0b", ring: "#fbbf24", text: "Degraded: check /status" },
    offline:  { dot: "#ef4444", ring: "#f87171", text: "Backend offline" },
    checking: { dot: "#94a3b8", ring: "#cbd5e1", text: "Checking…" },
  }[state];

  const formatted = checkedAt
    ? checkedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : "--:--";

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 10,
      background: "rgba(255,255,255,0.08)", border: "1px solid rgba(255,255,255,0.15)",
      borderRadius: 999, padding: "5px 12px", fontSize: 11, fontWeight: 600,
      color: "#e8fff4", letterSpacing: "0.04em",
    }}>
      <span style={{ position: "relative", display: "inline-flex", width: 8, height: 8 }}>
        <span style={{
          position: "absolute", inset: 0, borderRadius: "50%",
          background: palette.ring, opacity: 0.6,
          animation: state === "online" ? "statusPing 1.8s ease-out infinite" : "none",
        }} />
        <span style={{
          width: 8, height: 8, borderRadius: "50%", background: palette.dot,
          display: "inline-block", position: "relative",
        }} />
      </span>
      <span>{palette.text}</span>
      <span style={{ opacity: 0.5 }}>|</span>
      <span style={{ opacity: 0.75 }}>🔄 Checked {formatted}</span>
      <style>{`
        @keyframes statusPing {
          0%   { transform: scale(1);   opacity: 0.6; }
          70%  { transform: scale(2.2); opacity: 0;   }
          100% { transform: scale(2.2); opacity: 0;   }
        }
      `}</style>
    </div>
  );
}

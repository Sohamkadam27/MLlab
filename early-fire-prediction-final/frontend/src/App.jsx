import FireRiskMap from "./pages/FireRiskMap";
import PredictionPanel from "./pages/PredictionPanel";
import { useState, Component } from "react";

class AppErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 40, fontFamily: "Inter, sans-serif" }}>
          <h2>Something went wrong</h2>
          <pre style={{ whiteSpace: "pre-wrap", color: "#b00" }}>{String(this.state.error?.message ?? this.state.error)}</pre>
          <button onClick={() => window.location.reload()}>Reload</button>
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  const [activePage, setActivePage] = useState("fire-risk-map");
  const page = activePage === "prediction-panel"
    ? <PredictionPanel onNavigate={setActivePage} />
    : <FireRiskMap onNavigate={setActivePage} />;
  return <AppErrorBoundary>{page}</AppErrorBoundary>;
}

export default App;

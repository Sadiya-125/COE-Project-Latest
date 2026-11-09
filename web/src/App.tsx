import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Hospital,
  Loader2,
  PanelLeftClose,
  PanelLeftOpen,
  ShieldCheck,
  Stethoscope,
} from "lucide-react";
import { api, getApiKey, setApiKey, type Federation as Fed, type Run } from "./api";
import Comparison from "./components/Comparison";
import Diagnose from "./components/Diagnose";
import FederationView from "./components/Federation";
import Privacy from "./components/Privacy";
import Security from "./components/Security";
import "./styles.css";

type Tab = "overview" | "federation" | "diagnose" | "privacy" | "security";

const TABS: {
  id: Tab;
  label: string;
  icon: typeof BarChart3;
  title: string;
}[] = [
  { id: "overview", label: "Performance", icon: BarChart3, title: "Model performance" },
  { id: "federation", label: "Federation", icon: Hospital, title: "Federation" },
  { id: "diagnose", label: "Demo", icon: Stethoscope, title: "Diagnosis demo" },
  { id: "privacy", label: "Privacy", icon: ShieldCheck, title: "Privacy" },
  { id: "security", label: "Security", icon: Activity, title: "Security" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [collapsed, setCollapsed] = useState(false);

  const [runs, setRuns] = useState<Run[]>([]);
  const [fed, setFed] = useState<Fed | null>(null);
  const [key, setKey] = useState(getApiKey());
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void (async () => {
      try {
        await api.health();
        setError(null);
      } catch (e) {
        setError(String((e as Error).message ?? e));
      }
      try {
        setRuns((await api.experiments()).runs);
      } catch {
        setRuns([]);
      }
      try {
        setFed(await api.federation());
      } catch {
        setFed(null);
      }
      setLoading(false);
    })();
  }, []);

  // Fetch the demo key so the UI works out of the box. In a real deployment keys
  // are provisioned per hospital and never served over the wire.
  useEffect(() => {
    if (key) return;
    api
      .security()
      .then((r) => {
        if (r.demo_api_key) {
          setApiKey(r.demo_api_key);
          setKey(r.demo_api_key);
        }
      })
      .catch(() => undefined);
  }, [key]);

  const active = TABS.find((t) => t.id === tab)!;

  return (
    <div className="shell">
      <aside className={`sidebar ${collapsed ? "collapsed" : ""}`}>
        <div className="brand">
          <div className="mark">
            <ShieldCheck size={18} strokeWidth={2.4} />
          </div>
          <div className="hide-collapsed">
            <div className="name">FedDerm</div>
          </div>
        </div>

        <nav>
          {TABS.map((t) => {
            const Icon = t.icon;
            return (
              <button
                key={t.id}
                className={tab === t.id ? "active" : ""}
                onClick={() => setTab(t.id)}
                title={t.label}
                aria-label={t.label}
              >
                <Icon size={18} strokeWidth={2} />
                <span className="hide-collapsed">{t.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="foot">
          <button
            className="collapse-btn"
            onClick={() => setCollapsed((c) => !c)}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
            <span className="hide-collapsed">Collapse</span>
          </button>
        </div>
      </aside>

      <main className="content">
        <header className="page">
          <h1>{active.title}</h1>
        </header>

        {error && (
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="err">
              <AlertTriangle size={16} />
              <span>{error}</span>
            </div>
          </div>
        )}

        {loading && (
          <div className="card">
            <div className="spinner">
              <Loader2 size={16} className="spin" /> loading experiments…
            </div>
          </div>
        )}

        {!loading && tab === "overview" && <Comparison runs={runs} />}
        {!loading && tab === "federation" && <FederationView fed={fed} />}
        {!loading && tab === "diagnose" && <Diagnose />}
        {!loading && tab === "privacy" && <Privacy />}
        {!loading && tab === "security" && <Security />}
      </main>
    </div>
  );
}

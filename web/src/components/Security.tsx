import { useEffect, useState } from "react";
import { AlertTriangle, FileClock, RefreshCw, ShieldCheck } from "lucide-react";
import { api, type Control } from "../api";

type AuditRecord = { ts: number; event: string; detail: Record<string, unknown> };

export default function Security() {
  const [controls, setControls] = useState<Control[]>([]);
  const [audit, setAudit] = useState<AuditRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = () => {
    api
      .security()
      .then((r) => setControls(r.controls))
      .catch((e) => setError(String(e.message)));
    api
      .audit(40)
      .then((r) => setAudit([...r.records].reverse()))
      .catch(() => setAudit([]));
  };

  useEffect(refresh, []);

  return (
    <div className="grid">
      <div className="card">
        <div className="section-title">
          <ShieldCheck size={17} />
          <h2>Enforced controls</h2>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Control</th>
                <th>Implementation</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {controls.map((c) => (
                <tr key={c.id}>
                  <td>{c.control}</td>
                  <td className="note" style={{ textAlign: "left", whiteSpace: "normal" }}>
                    {c.detail}
                  </td>
                  <td>
                    <span className={`pill ${c.active ? "good" : "warn"}`}>
                      {c.active ? "active" : "off"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {error && (
          <div className="err">
            <AlertTriangle size={15} />
            {error}
          </div>
        )}
      </div>

      <div className="card">
        <div className="section-title">
          <FileClock size={17} />
          <h2>Audit trail</h2>
        </div>
        <button className="ghost" onClick={refresh}>
          <RefreshCw size={14} /> Refresh
        </button>
        <div className="table-wrap" style={{ marginTop: 14 }}>
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Event</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {audit.map((r, i) => (
                <tr key={i}>
                  <td className="mono" style={{ fontSize: 11 }}>
                    {new Date(r.ts * 1000).toLocaleTimeString()}
                  </td>
                  <td className="mono" style={{ fontSize: 11 }}>
                    {r.event}
                  </td>
                  <td
                    className="note mono"
                    style={{ textAlign: "left", fontSize: 10.5, whiteSpace: "normal" }}
                  >
                    {JSON.stringify(r.detail).slice(0, 96)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!audit.length && <div className="spinner">no audit records yet</div>}
        </div>
      </div>
    </div>
  );
}

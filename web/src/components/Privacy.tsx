import { useEffect, useState } from "react";
import { AlertTriangle, KeySquare, Loader2, Lock, Play, TrendingDown } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type PrivacyCurve, type PrivacySweep, type SecAggResult } from "../api";

const COLORS = ["#3ddc97", "#6c8cff", "#ffc357", "#ff6b6b", "#b892ff"];

const TOOLTIP = {
  background: "#111729",
  border: "1px solid #232e4f",
  borderRadius: 10,
  fontSize: 12,
  fontFamily: "JetBrains Mono, monospace",
};

export default function Privacy() {
  const [curve, setCurve] = useState<PrivacyCurve | null>(null);
  const [sweep, setSweep] = useState<PrivacySweep | null>(null);
  const [sec, setSec] = useState<SecAggResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clients, setClients] = useState(4);
  const [dim, setDim] = useState(2000);
  const [sigma, setSigma] = useState(0.8);

  useEffect(() => {
    api.privacyCurve(45).then(setCurve).catch((e) => setError(String(e.message)));
    api.privacySweep().then(setSweep).catch(() => setSweep(null));
  }, []);

  const runSecagg = async () => {
    setBusy(true);
    setError(null);
    try {
      setSec(
        await api.secagg({
          num_clients: clients,
          dimension: dim,
          round_no: Math.floor(Math.random() * 1000),
          noise_multiplier: sigma,
          l2_clip: 1.0,
        }),
      );
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  };

  const seriesKeys = curve?.series.map((s) => `σ=${s.noise_multiplier}`) ?? [];
  const merged =
    curve?.series[0]?.points.map((_, i) => {
      const row: Record<string, number> = { round: i + 1 };
      curve.series.forEach((s) => {
        row[`σ=${s.noise_multiplier}`] = Number(s.points[i].epsilon.toFixed(4));
      });
      return row;
    }) ?? [];

  // A log axis needs finite positive bounds; "auto" leaves the chart blank.
  const allEps = merged.flatMap((r) => seriesKeys.map((k) => r[k])).filter((v) => v > 0);
  const domain: [number, number] = allEps.length
    ? [Math.max(Math.min(...allEps) * 0.7, 1e-3), Math.max(...allEps) * 1.4]
    : [0.1, 100];

  return (
    <div className="grid">
      <div className="card chart-card">
        <div className="section-title">
          <TrendingDown size={17} />
          <h2>Privacy budget growth</h2>
        </div>
        {merged.length ? (
          <ResponsiveContainer width="100%" height={330}>
            <LineChart data={merged} margin={{ top: 4, right: 14, left: 2, bottom: 0 }}>
              <CartesianGrid stroke="#1b2340" strokeDasharray="3 3" />
              <XAxis
                dataKey="round"
                stroke="#8f9abd"
                fontSize={11}
                tickLine={false}
                label={{
                  value: "federated round",
                  position: "insideBottom",
                  offset: -2,
                  fill: "#6b7699",
                  fontSize: 11,
                }}
              />
              <YAxis
                stroke="#8f9abd"
                fontSize={11}
                scale="log"
                domain={domain}
                allowDataOverflow
                tickLine={false}
                axisLine={false}
                tickFormatter={(v: number) => (v >= 1 ? v.toFixed(0) : v.toFixed(2))}
              />
              <Tooltip
                contentStyle={TOOLTIP}
                formatter={(v) => (typeof v === "number" ? `ε = ${v.toFixed(2)}` : "-")}
              />
              <Legend wrapperStyle={{ fontSize: 11, paddingTop: 12 }} />
              {seriesKeys.map((k, i) => (
                <Line
                  key={k}
                  type="monotone"
                  dataKey={k}
                  stroke={COLORS[i % COLORS.length]}
                  dot={false}
                  strokeWidth={2}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="spinner">
            <Loader2 size={15} className="spin" /> loading accountant curve…
          </div>
        )}
      </div>

      <div className="card">
        <h2>DP scaling</h2>
        <div className="grid cols-3">
          <div>
            <div className="stat-label">Model dimension d</div>
            <div className="stat small">{sweep?.scaling.num_params.toLocaleString() ?? "—"}</div>
          </div>
          <div>
            <div className="stat-label">Noise : signal — DP-SGD (B=128)</div>
            <div className="stat small">
              {sweep ? `${sweep.scaling.record_level_ratio_at_b128.toFixed(2)} × σ` : "—"}
            </div>
          </div>
          <div>
            <div className="stat-label">Noise : signal — client DP (m=5)</div>
            <div className="stat small">
              {sweep ? `${sweep.scaling.client_level_ratio_at_m5.toFixed(0)} × σ` : "—"}
            </div>
          </div>
        </div>

        {sweep?.sweep && (
          <>
            <h3 style={{ marginTop: 24 }}>Measured DP-SGD frontier</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Noise σ</th>
                    <th>ε (δ=1e-5)</th>
                    <th>Balanced acc.</th>
                    <th>Macro F1</th>
                    <th>Malig. sens.</th>
                    <th>Noise : signal</th>
                  </tr>
                </thead>
                <tbody>
                  {sweep.sweep.points.map((p) => (
                    <tr key={p.noise_multiplier}>
                      <td className="mono">{p.noise_multiplier}</td>
                      <td className="mono">{p.epsilon.toFixed(1)}</td>
                      <td className="mono">{(100 * p.balanced_accuracy).toFixed(2)}%</td>
                      <td className="mono">{(100 * p.macro_f1).toFixed(2)}%</td>
                      <td className="mono">{(100 * p.malignant_sensitivity).toFixed(2)}%</td>
                      <td className="mono">{p.noise_to_signal.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>

      <div className="grid cols-2">
        <div className="card">
          <div className="section-title">
            <Lock size={17} />
            <h2>Live secure aggregation</h2>
          </div>
          <div className="grid cols-3" style={{ marginBottom: 18 }}>
            <label className="field">
              Hospitals
              <input
                type="number"
                min={2}
                max={12}
                value={clients}
                onChange={(e) => setClients(+e.target.value)}
              />
            </label>
            <label className="field">
              Vector size
              <input
                type="number"
                min={8}
                max={200000}
                step={500}
                value={dim}
                onChange={(e) => setDim(+e.target.value)}
              />
            </label>
            <label className="field">
              DP noise σ
              <input
                type="number"
                min={0}
                max={8}
                step={0.1}
                value={sigma}
                onChange={(e) => setSigma(+e.target.value)}
              />
            </label>
          </div>
          <button className="primary" onClick={() => void runSecagg()} disabled={busy}>
            {busy ? <Loader2 size={15} className="spin" /> : <Play size={15} />}
            {busy ? "running…" : "Run a masked round"}
          </button>
          {error && (
            <div className="err">
              <AlertTriangle size={15} />
              {error}
            </div>
          )}
        </div>

        <div className="card">
          <div className="section-title">
            <KeySquare size={17} />
            <h2>What the server observes</h2>
          </div>
          {!sec && (
            <p className="sub" style={{ marginBottom: 0 }}>
              Run a round to populate this panel.
            </p>
          )}
          {sec && (
            <>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Hospital</th>
                      <th>True update mean</th>
                      <th>Masked mean (server view)</th>
                      <th>Masked σ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sec.clients.map((c) => (
                      <tr key={c.client_id}>
                        <td>Hospital-{c.client_id + 1}</td>
                        <td className="mono">{c.true_update_mean.toFixed(5)}</td>
                        <td className="mono">{c.masked_seen_by_server.mean.toExponential(2)}</td>
                        <td className="mono">{c.masked_seen_by_server.std.toExponential(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{ marginTop: 18 }}>
                <div className="kv">
                  <span>Recovered aggregate mean</span>
                  <span className="mono">{sec.aggregate.recovered_mean.toFixed(6)}</span>
                </div>
                <div className="kv">
                  <span>True aggregate mean</span>
                  <span className="mono">{sec.aggregate.true_mean.toFixed(6)}</span>
                </div>
                <div className="kv">
                  <span>Max absolute error (quantization only)</span>
                  <span className="mono">{sec.aggregate.max_abs_error.toExponential(2)}</span>
                </div>
                <div className="kv">
                  <span>Round time · {sec.dimension.toLocaleString()} params</span>
                  <span className="mono">{sec.elapsed_ms.toFixed(1)} ms</span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

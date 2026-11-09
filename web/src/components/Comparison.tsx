import { Info } from "lucide-react";
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
import type { Run } from "../api";

const COLORS: Record<string, string> = {
  centralized: "#6c8cff",
  federated: "#3ddc97",
  example: "#ffc357",
  client: "#ff6b6b",
};

const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(2)}%`);

const PICKS = [
  { key: "centralized", preferred: "centralized_b0_288_ft120_w085_b16_tuned_v5", label: "Centralized" },
  { key: "federated", preferred: "fedavg_ft20_balanced_e3_tuned_v6", label: "Federated" },
  { key: "example", preferred: "federated_dp_example", label: "DP-SGD" },
  { key: "client", preferred: "federated_dp_client_tpu", label: "Client-level DP" },
] as const;

const MATCH: Record<(typeof PICKS)[number]["key"], (r: Run) => boolean> = {
  centralized: (r) => (r.mode ?? "").startsWith("centralized"),
  federated: (r) =>
    !(r.mode ?? "").startsWith("centralized") && r.epsilon == null && r.communication != null,
  example: (r) => r.dp_mode === "example" || r.run.includes("dp_example"),
  client: (r) => r.dp_mode === "client" || r.run.includes("dp_client"),
};

const TOOLTIP = {
  background: "#111729",
  border: "1px solid #232e4f",
  borderRadius: 10,
  fontSize: 12,
  fontFamily: "JetBrains Mono, monospace",
};

export default function Comparison({ runs }: { runs: Run[] }) {
  if (!runs.length)
    return (
      <div className="card">
        <div className="section-title">
          <Info size={17} />
          <h2>No results yet</h2>
        </div>
      </div>
    );

  const seen = new Set<string>();
  const featured = PICKS.map((pick) => {
    let run = runs.find((r) => r.run === pick.preferred);
    if (!run) {
      const cands = runs.filter(MATCH[pick.key]).filter((r) => r.balanced_accuracy != null);
      run = cands.reduce<Run | undefined>(
        (best, r) => (r.balanced_accuracy > (best?.balanced_accuracy ?? -1) ? r : best),
        undefined,
      );
    }
    if (!run || seen.has(run.run)) return null;
    seen.add(run.run);
    return { pick, run };
  }).filter((x): x is { pick: (typeof PICKS)[number]; run: Run } => x !== null);

  const maxLen = Math.max(0, ...featured.map(({ run }) => run.curve.length));
  const merged = Array.from({ length: maxLen }, (_, i) => {
    const row: Record<string, number | null> = { x: i + 1 };
    featured.forEach(({ pick, run }) => {
      row[pick.label] = run.curve[i]?.balanced_accuracy ?? null;
    });
    return row;
  });

  const classSource = featured.find(({ pick }) => pick.key === "centralized")?.run ?? featured[0]?.run;
  const classes = Object.keys(classSource?.per_class_recall ?? {});

  return (
    <div className="grid">
      <div className="metric-grid">
        {featured.map(({ pick, run }) => (
          <div className="card metric-card" key={pick.key}>
            <div className="stat-label">{pick.label}</div>
            <div className="stat">{pct(run.balanced_accuracy)}</div>
            <div className="pill-row">
              <span className={`pill ${pick.key === "centralized" ? "bad" : "good"}`}>
                {pick.key === "centralized" ? "images pooled" : "images local"}
              </span>
              {run.epsilon != null && (
                <span className="pill warn">
                  ε={run.epsilon.toFixed(run.epsilon < 100 ? 1 : 0)}
                </span>
              )}
              {run.secure_aggregation && <span className="pill good">SecAgg</span>}
            </div>
          </div>
        ))}
      </div>

      <div className="card benchmark-table">
        <h2>Verified benchmark results</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Regime</th>
                <th>Accuracy</th>
                <th>Balanced acc.</th>
                <th>Macro F1</th>
                <th>Malignant sensitivity</th>
                <th>Privacy</th>
              </tr>
            </thead>
            <tbody>
              {featured.map(({ pick, run }) => (
                <tr key={pick.key}>
                  <td>{pick.label}</td>
                  <td className="mono">{pct(run.accuracy)}</td>
                  <td className="mono">
                    <b>{pct(run.balanced_accuracy)}</b>
                  </td>
                  <td className="mono">{pct(run.macro_f1)}</td>
                  <td className="mono">{pct(run.malignant_sensitivity)}</td>
                  <td className="mono">
                    {run.epsilon == null ? "No DP" : `ε=${run.epsilon.toFixed(1)}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card chart-card">
        <h2>Convergence</h2>
        <div className="chart-shell">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={merged} margin={{ top: 4, right: 14, left: -6, bottom: 0 }}>
              <CartesianGrid stroke="#1b2340" strokeDasharray="3 3" />
              <XAxis
                dataKey="x"
                interval="preserveStartEnd"
                minTickGap={24}
                stroke="#8f9abd"
                fontSize={11}
                tickLine={false}
                label={{ value: "epoch / round", position: "insideBottomRight", offset: -4, fontSize: 10.5, fill: "#8f9abd" }}
              />
              <YAxis
                stroke="#8f9abd"
                fontSize={11}
                tickLine={false}
                axisLine={false}
                domain={[0, "auto"]}
                tickFormatter={(v: number) => `${(100 * v).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={TOOLTIP}
                formatter={(v) => pct(typeof v === "number" ? v : null)}
              />
              <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
              {featured.map(({ pick }) => (
                <Line
                  key={pick.key}
                  type="monotone"
                  dataKey={pick.label}
                  stroke={COLORS[pick.key]}
                  dot={false}
                  strokeWidth={2}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <h2>Per-class recall</h2>
        {classes.map((cls) => (
          <div className="class-block" key={cls}>
            <div className="cls">{cls}</div>
            {featured.map(({ pick, run }) => (
              <div className="bar-row" key={pick.key}>
                <div className="recall-name">{pick.label}</div>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${100 * (run.per_class_recall?.[cls] ?? 0)}%`,
                      background: COLORS[pick.key],
                    }}
                  />
                </div>
                <div className="val">{pct(run.per_class_recall?.[cls])}</div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

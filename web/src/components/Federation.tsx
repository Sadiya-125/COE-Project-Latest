import { Building2, Info, Layers, Scale } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Federation as Fed } from "../api";

const CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];
const COLORS = ["#ff6b6b", "#ffa15c", "#ffc357", "#b892ff", "#ff8fb1", "#6c8cff", "#3ddc97"];

const TOOLTIP = {
  background: "#111729",
  border: "1px solid #232e4f",
  borderRadius: 10,
  fontSize: 12,
  fontFamily: "JetBrains Mono, monospace",
};

export default function Federation({ fed }: { fed: Fed | null }) {
  if (!fed)
    return (
      <div className="card">
        <div className="section-title">
          <Info size={17} />
          <h2>No federation metadata yet</h2>
        </div>
      </div>
    );

  const data = fed.clients.map((c) => {
    const total = CLASSES.reduce((s, cl) => s + (c.label_counts[cl] ?? 0), 0) || 1;
    const row: Record<string, number | string> = { name: c.name.replace("Hospital-", "H") };
    CLASSES.forEach((cl) => {
      row[cl] = (100 * (c.label_counts[cl] ?? 0)) / total;
    });
    return row;
  });

  return (
    <div className="grid">
      <div className="grid cols-3">
        <div className="card">
          <div className="stat-label">
            <Building2 size={11} style={{ verticalAlign: "-1px", marginRight: 5 }} />
            Participating hospitals
          </div>
          <div className="stat">{fed.num_clients}</div>
        </div>
        <div className="card">
          <div className="stat-label">
            <Layers size={11} style={{ verticalAlign: "-1px", marginRight: 5 }} />
            Site size range
          </div>
          <div className="stat">
            {fed.size_min}–{fed.size_max}
          </div>
        </div>
        <div className="card">
          <div className="stat-label">
            <Scale size={11} style={{ verticalAlign: "-1px", marginRight: 5 }} />
            Label skew
          </div>
          <div className="stat">{fed.mean_total_variation.toFixed(3)}</div>
        </div>
      </div>

      <div className="card chart-card">
        <h2>Diagnosis mix by hospital</h2>
        <div className="chart-shell">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 4, right: 8, left: -8, bottom: 0 }}>
              <CartesianGrid stroke="#1b2340" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" stroke="#8f9abd" fontSize={11} tickLine={false} />
              <YAxis
                stroke="#8f9abd"
                fontSize={11}
                tickLine={false}
                axisLine={false}
                domain={[0, 100]}
                tickFormatter={(v: number) => `${v.toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={TOOLTIP}
                cursor={{ fill: "rgba(108,140,255,0.07)" }}
                formatter={(v) => `${(typeof v === "number" ? v : 0).toFixed(1)}%`}
              />
              <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
              {CLASSES.map((c, i) => (
                <Bar key={c} dataKey={c} stackId="a" fill={COLORS[i]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <h2>Site detail</h2>
        <div className="table-wrap federation-table">
          <table>
            <thead>
              <tr>
                <th>Hospital</th>
                <th>Images</th>
                <th>Lesions</th>
                {CLASSES.map((c) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {fed.clients.map((c) => (
                <tr key={c.client_id}>
                  <td>{c.name}</td>
                  <td className="mono">{c.num_samples}</td>
                  <td className="mono">{c.num_lesions}</td>
                  {CLASSES.map((cl) => (
                    <td
                      key={cl}
                      className="mono"
                      style={{ color: c.label_counts[cl] === 0 ? "#6b7699" : undefined }}
                    >
                      {c.label_counts[cl]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

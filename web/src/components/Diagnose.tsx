import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Upload,
  XCircle,
} from "lucide-react";
import { api, fetchSampleBlob, sampleThumbnail, type Prediction, type Sample } from "../api";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function SampleThumb({
  sample,
  selected,
  onPick,
}: {
  sample: Sample;
  selected: boolean;
  onPick: () => void;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    // `<img src>` cannot send the X-API-Key header, so the thumbnail is fetched
    // with fetch() and handed over as an object URL.
    sampleThumbnail(sample.image_id)
      .then((u) => alive && setUrl(u))
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [sample.image_id]);

  return (
    <div className="sample-wrap">
      {url ? (
        <img
          src={url}
          className={selected ? "sel" : ""}
          onClick={onPick}
          alt={sample.truth}
          title={`${sample.truth_name} · ${sample.sex}, ${sample.age} · ${sample.localization}`}
        />
      ) : (
        <div className="ph">{failed ? "auth?" : "…"}</div>
      )}
      <span className="mono">{sample.truth}</span>
    </div>
  );
}

export default function Diagnose() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [truth, setTruth] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api
      .samples(14)
      .then((r) => setSamples(r.samples))
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  const run = useCallback(async (blob: Blob, name: string, groundTruth: string | null) => {
    setBusy(true);
    setError(null);
    setResult(null);
    setTruth(groundTruth);
    try {
      setPreviewUrl(URL.createObjectURL(blob));
      setResult(await api.predict(blob, name));
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  }, []);

  const onSample = async (s: Sample) => {
    setSelected(s.image_id);
    try {
      await run(await fetchSampleBlob(s.image_id), `${s.image_id}.jpg`, s.truth);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  };

  const onFile = (f: File | undefined) => {
    if (!f) return;
    setSelected(null);
    void run(f, f.name, null);
  };

  const correct = truth && result ? truth === result.predicted_class : null;

  return (
    <div className="grid cols-2">
      <div className="grid" style={{ alignContent: "start" }}>
        <div className="card">
          <h2>Upload an image</h2>
          <div
            className={`dropzone ${drag ? "drag" : ""}`}
            onClick={() => fileInput.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              onFile(e.dataTransfer.files[0]);
            }}
          >
            <Upload size={22} />
            <span>Drop or select an image</span>
            <input
              ref={fileInput}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              hidden
              onChange={(e) => onFile(e.target.files?.[0])}
            />
          </div>
          {error && (
            <div className="err">
              <AlertTriangle size={15} />
              {error}
            </div>
          )}
        </div>

        <div className="card">
          <h2>Or pick a held-out test case</h2>
          {samples.length ? (
            <div className="samples">
              {samples.map((s) => (
                <SampleThumb
                  key={s.image_id}
                  sample={s}
                  selected={selected === s.image_id}
                  onPick={() => void onSample(s)}
                />
              ))}
            </div>
          ) : (
            !error && (
              <div className="spinner">
                <Loader2 size={15} className="spin" /> loading samples…
              </div>
            )
          )}
        </div>
      </div>

      <div className="card" style={{ alignSelf: "start" }}>
        <h2>Assessment</h2>
        {!busy && !result && (
          <p className="sub" style={{ marginBottom: 0 }}>Select an image to run the model.</p>
        )}
        {busy && (
          <div className="spinner">
            <Loader2 size={15} className="spin" /> scoring…
          </div>
        )}

        {previewUrl && result && (
          <div className="grid" style={{ gap: 18 }}>
            <div className={`triage ${result.triage.level}`}>
              {result.triage.level === "routine" ? (
                <CheckCircle2 size={18} />
              ) : (
                <AlertTriangle size={18} />
              )}
              <span>
                <b style={{ textTransform: "uppercase", letterSpacing: "0.04em" }}>
                  {result.triage.level}
                </b>
                {" — "}
                {result.triage.message}
              </span>
            </div>

            <div style={{ display: "flex", gap: 18, alignItems: "flex-start", flexWrap: "wrap" }}>
              <div className="preview" style={{ width: 168, flex: "0 0 168px" }}>
                <img src={previewUrl} alt="lesion" />
              </div>
              <div style={{ flex: 1, minWidth: 180 }}>
                <div className="stat-label">Top prediction</div>
                <div className="stat small">{result.predicted_label}</div>
                <div className="note mono">
                  {result.predicted_class} · {pct(result.confidence)} confidence
                </div>
                {truth && (
                  <div className="pill-row">
                    <span className={`pill ${correct ? "good" : "bad"}`}>
                      {correct ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
                      ground truth {truth}
                      {correct ? " · correct" : " · missed"}
                    </span>
                  </div>
                )}
              </div>
            </div>

            <div>
              <h3 style={{ marginBottom: 10 }}>Class probabilities</h3>
              {result.probabilities.map((p) => (
                <div className="bar-row" key={p.code}>
                  <div className="name mono">
                    {p.code} {p.malignant ? "⚠" : ""}
                  </div>
                  <div className="bar-track">
                    <div
                      className={`bar-fill ${p.malignant ? "mal" : ""}`}
                      style={{ width: `${100 * p.probability}%` }}
                    />
                  </div>
                  <div className="val">{pct(p.probability)}</div>
                </div>
              ))}
            </div>

            <div>
              <div className="kv">
                <span>Malignancy probability</span>
                <span className="mono">{pct(result.malignant_probability)}</span>
              </div>
              <div className="kv">
                <span>Latency</span>
                <span className="mono">{result.latency_ms.toFixed(1)} ms</span>
              </div>
            </div>
            <p className="note" style={{ margin: 0 }}>
              {result.disclaimer}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

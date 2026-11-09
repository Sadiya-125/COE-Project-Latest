export const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";

const KEY_STORAGE = "fedderm.apiKey";

export function getApiKey(): string {
  return localStorage.getItem(KEY_STORAGE) ?? "";
}

export function setApiKey(key: string): void {
  localStorage.setItem(KEY_STORAGE, key);
}

/** Re-fetch the demo key when a cached one has gone stale (e.g. API restarted). */
async function refreshDemoKey(): Promise<string | null> {
  try {
    const res = await fetch(`${API_BASE}/api/security`);
    if (!res.ok) return null;
    const key = (await res.json()).demo_api_key as string | undefined;
    if (key) {
      setApiKey(key);
      return key;
    }
  } catch {
    /* API down */
  }
  return null;
}

export async function authedFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const send = (key: string) => {
    const headers = new Headers(init.headers);
    if (key) headers.set("X-API-Key", key);
    return fetch(`${API_BASE}${path}`, { ...init, headers });
  };
  let res = await send(getApiKey());
  if (res.status === 401 || res.status === 403) {
    const fresh = await refreshDemoKey();
    if (fresh) res = await send(fresh);
  }
  return res;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await authedFetch(path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).error ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------- types ---- //
export type ClassInfo = { code: string; name: string };

export type Health = {
  status: string;
  model_source: string;
  checkpoint: string | null;
  image_size: number;
  classes: ClassInfo[];
  uptime_s: number;
};

export type Control = { id: string; control: string; detail: string; active: boolean };

export type CurvePoint = {
  x: number;
  balanced_accuracy: number | null;
  macro_f1: number | null;
  loss: number | null;
  epsilon: number | null;
};

export type Run = {
  run: string;
  mode: string;
  accuracy: number;
  balanced_accuracy: number;
  macro_f1: number;
  macro_auc: number | null;
  malignant_sensitivity: number;
  malignant_specificity: number | null;
  epsilon: number | null;
  delta: number | null;
  dp_mode: string | null;
  secure_aggregation: boolean | null;
  wall_time_s: number | null;
  num_params: number | null;
  data_left_premises: boolean;
  communication: { total_uplink_mb: number; bytes_per_client_round: number } | null;
  per_class_recall: Record<string, number> | null;
  confusion_matrix: number[][] | null;
  curve: CurvePoint[];
};

export type Client = {
  client_id: number;
  name: string;
  num_samples: number;
  num_lesions: number;
  label_counts: Record<string, number>;
  top_localizations?: Record<string, number>;
};

export type Federation = {
  num_clients: number;
  clients: Client[];
  size_min: number;
  size_max: number;
  mean_total_variation: number;
};

export type Prediction = {
  predicted_class: string;
  predicted_label: string;
  confidence: number;
  malignant_probability: number;
  triage: { level: string; message: string };
  probabilities: { code: string; name: string; probability: number; malignant: boolean }[];
  model_source: string;
  latency_ms: number;
  image_sha256: string;
  privacy: {
    trained_with_dp: boolean;
    epsilon: number | null;
    delta: number | null;
    dp_mode: string | null;
    secure_aggregation: boolean | null;
  };
  disclaimer: string;
};

export type Sample = {
  image_id: string;
  truth: string;
  truth_name: string;
  age: number;
  sex: string;
  localization: string;
};

export type SecAggResult = {
  num_clients: number;
  dimension: number;
  elapsed_ms: number;
  clients: {
    client_id: number;
    true_update_mean: number;
    true_sample: number[];
    masked_seen_by_server: { mean: number; std: number; min: number; max: number; sample: number[] };
  }[];
  aggregate: {
    recovered_mean: number;
    true_mean: number;
    max_abs_error: number;
    recovered_sample: number[];
    with_dp_noise_sample: number[];
  };
  explanation: Record<string, string>;
};

export type SweepPoint = {
  noise_multiplier: number;
  epsilon: number;
  balanced_accuracy: number;
  macro_f1: number;
  malignant_sensitivity: number;
  noise_to_signal: number;
};

export type PrivacySweep = {
  sweep: { rounds: number; mode: string; points: SweepPoint[] } | null;
  scaling: {
    num_params: number;
    sqrt_d: number;
    explanation: string;
    record_level_ratio_at_b128: number;
    client_level_ratio_at_m5: number;
  };
};

export type PrivacyCurve = {
  delta: number;
  sampling_rate: number;
  series: { noise_multiplier: number; points: { round: number; epsilon: number }[] }[];
};

// ------------------------------------------------------------- endpoints --- //
export const api = {
  health: () => request<Health>("/api/health"),
  security: () => request<{ controls: Control[]; demo_api_key: string }>("/api/security"),
  experiments: () =>
    request<{ runs: Run[]; classes: string[]; class_names: Record<string, string> }>(
      "/api/experiments",
    ),
  federation: () => request<Federation>("/api/federation"),
  privacyCurve: (rounds = 40) => request<PrivacyCurve>(`/api/privacy/curve?rounds=${rounds}`),
  privacySweep: () => request<PrivacySweep>("/api/privacy/sweep"),
  samples: (n = 8) => request<{ samples: Sample[] }>(`/api/samples?n=${n}`),
  sampleImageUrl: (id: string) => `${API_BASE}/api/samples/${id}/image`,
  audit: (limit = 30) =>
    request<{ records: { ts: number; event: string; detail: Record<string, unknown> }[] }>(
      `/api/audit?limit=${limit}`,
    ),
  predict: (file: Blob, filename = "lesion.jpg") => {
    const form = new FormData();
    form.append("file", file, filename);
    return request<Prediction>("/api/predict", { method: "POST", body: form });
  },
  secagg: (body: {
    num_clients: number;
    dimension: number;
    round_no: number;
    noise_multiplier: number;
    l2_clip: number;
  }) =>
    request<SecAggResult>("/api/secagg/demo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};

// Fetch a sample image as a Blob so it can be POSTed to /api/predict exactly
// like a user upload (the API key header is required for the GET too).
export async function fetchSampleBlob(id: string): Promise<Blob> {
  const res = await authedFetch(`/api/samples/${id}/image`);
  if (!res.ok) throw new Error(`could not load sample ${id} (${res.status})`);
  return res.blob();
}

// `<img src>` cannot carry the X-API-Key header, so thumbnails have to be
// fetched with fetch() and handed to the browser as object URLs. Cached so
// re-renders and tab switches do not re-download.
const thumbCache = new Map<string, string>();

export async function sampleThumbnail(id: string): Promise<string> {
  const cached = thumbCache.get(id);
  if (cached) return cached;
  const url = URL.createObjectURL(await fetchSampleBlob(id));
  thumbCache.set(id, url);
  return url;
}

import React, { useEffect, useState, useMemo } from "react";
import { getModelStats } from "../api/client.js";

// ── Pure metric computation (no server round-trip) ───────────────────────────

function computeAt(probs, labels, threshold) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < probs.length; i++) {
    const pred = probs[i] >= threshold ? 1 : 0;
    if      (pred === 1 && labels[i] === 1) tp++;
    else if (pred === 1 && labels[i] === 0) fp++;
    else if (pred === 0 && labels[i] === 1) fn++;
    else                                    tn++;
  }
  const n   = probs.length;
  return {
    sensitivity: tp + fn > 0 ? tp / (tp + fn) : 0,
    specificity: tn + fp > 0 ? tn / (tn + fp) : 0,
    accuracy:    (tp + tn) / n,
    ppv:         tp + fp > 0 ? tp / (tp + fp) : 0,
    npv:         tn + fn > 0 ? tn / (tn + fn) : 0,
    tp, fp, tn, fn,
  };
}

/** Build sensitivity & specificity curves at STEPS evenly-spaced thresholds. */
function buildCurves(probs, labels, steps = 100) {
  return Array.from({ length: steps + 1 }, (_, i) => {
    const t = i / steps;
    const m = computeAt(probs, labels, t);
    return { threshold: t, sensitivity: m.sensitivity, specificity: m.specificity };
  });
}

// ── SVG threshold-explorer chart ─────────────────────────────────────────────

const W = 460, H = 200;
const PAD = { left: 38, right: 16, top: 16, bottom: 28 };
const IW  = W - PAD.left - PAD.right;
const IH  = H - PAD.top  - PAD.bottom;

const tx = (t) => PAD.left + t * IW;
const my = (m) => PAD.top  + (1 - m) * IH;

function ThresholdChart({ curves, threshold, optimalThreshold }) {
  const sensPath = curves
    .map((c, i) => `${i === 0 ? "M" : "L"}${tx(c.threshold).toFixed(1)},${my(c.sensitivity).toFixed(1)}`)
    .join(" ");
  const specPath = curves
    .map((c, i) => `${i === 0 ? "M" : "L"}${tx(c.threshold).toFixed(1)},${my(c.specificity).toFixed(1)}`)
    .join(" ");

  const cx   = tx(threshold);
  const optX = tx(optimalThreshold);

  // Find approximate intersection (where |sens - spec| is minimised)
  const intersect = curves.reduce((best, c) =>
    Math.abs(c.sensitivity - c.specificity) < Math.abs(best.sensitivity - best.specificity) ? c : best
  );
  const ix = tx(intersect.threshold);
  const iy = my((intersect.sensitivity + intersect.specificity) / 2);

  return (
    <svg
      width={W} height={H}
      viewBox={`0 0 ${W} ${H}`}
      style={{ maxWidth: "100%", display: "block" }}
      aria-label="Sensitivität und Spezifität vs. Schwellenwert"
    >
      {/* Grid lines */}
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <line key={v}
          x1={PAD.left} y1={my(v)} x2={W - PAD.right} y2={my(v)}
          stroke="#e2e8f0" strokeWidth={1}
        />
      ))}
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <line key={v}
          x1={tx(v)} y1={PAD.top} x2={tx(v)} y2={H - PAD.bottom}
          stroke="#e2e8f0" strokeWidth={1}
        />
      ))}

      {/* Sensitivity (red) */}
      <path d={sensPath} fill="none" stroke="#e53e3e" strokeWidth={2} />
      {/* Specificity (green) */}
      <path d={specPath} fill="none" stroke="#38a169" strokeWidth={2} />

      {/* Intersection marker */}
      <circle cx={ix} cy={iy} r={4} fill="#805ad5" opacity={0.7} />
      <line x1={ix} y1={PAD.top} x2={ix} y2={H - PAD.bottom}
        stroke="#805ad5" strokeWidth={1} strokeDasharray="3,3" opacity={0.5}
      />

      {/* Optimal threshold marker */}
      {Math.abs(optimalThreshold - threshold) > 0.005 && (
        <line x1={optX} y1={PAD.top} x2={optX} y2={H - PAD.bottom}
          stroke="#d69e2e" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.7}
        />
      )}

      {/* Current threshold marker */}
      <line x1={cx} y1={PAD.top} x2={cx} y2={H - PAD.bottom}
        stroke="#3182ce" strokeWidth={2} strokeDasharray="5,3"
      />

      {/* Axes */}
      <line x1={PAD.left} y1={PAD.top}       x2={PAD.left}       y2={H - PAD.bottom} stroke="#a0aec0" strokeWidth={1} />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right}  y2={H - PAD.bottom} stroke="#a0aec0" strokeWidth={1} />

      {/* Y-axis labels */}
      {[0, 0.5, 1].map((v) => (
        <text key={v} x={PAD.left - 4} y={my(v) + 4} textAnchor="end" fontSize={10} fill="#718096">
          {(v * 100).toFixed(0)}%
        </text>
      ))}

      {/* X-axis labels */}
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <text key={v} x={tx(v)} y={H - PAD.bottom + 13} textAnchor="middle" fontSize={10} fill="#718096">
          {v.toFixed(2)}
        </text>
      ))}

      {/* Legend */}
      <rect x={PAD.left + 6}  y={PAD.top + 6}  width={12} height={3} fill="#e53e3e" />
      <text x={PAD.left + 22} y={PAD.top + 11} fontSize={10} fill="#e53e3e">Sensitivität</text>
      <rect x={PAD.left + 95} y={PAD.top + 6}  width={12} height={3} fill="#38a169" />
      <text x={PAD.left + 111} y={PAD.top + 11} fontSize={10} fill="#38a169">Spezifität</text>
      <rect x={PAD.left + 174} y={PAD.top + 6} width={12} height={3} fill="#805ad5" />
      <text x={PAD.left + 190} y={PAD.top + 11} fontSize={10} fill="#805ad5">Schnittpunkt</text>
    </svg>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

const spinnerStyle = {
  display: "inline-block", width: 20, height: 20,
  border: "3px solid #bee3f8", borderTopColor: "#3182ce",
  borderRadius: "50%", animation: "spin 0.7s linear infinite",
  marginRight: 8, verticalAlign: "middle",
};

function MetricCard({ label, value, highlight, changed }) {
  const pct = (value * 100).toFixed(1);
  return (
    <div style={{
      background: highlight ? "#ebf4ff" : "#fff",
      border: `1px solid ${highlight ? "#bee3f8" : "#e2e8f0"}`,
      borderRadius: 10, padding: "0.85rem 1rem", textAlign: "center",
      transition: "background 0.2s",
    }}>
      <div style={{
        fontSize: "1.55rem", fontWeight: 700,
        color: highlight ? "#2b6cb0" : "#2d3748",
        transition: "color 0.2s",
      }}>
        {pct}%
      </div>
      <div style={{ fontSize: "0.78rem", color: "#718096", marginTop: 2 }}>{label}</div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function ModelStats() {
  const [stats, setStats]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);
  const [threshold, setThreshold] = useState(0.5);

  useEffect(() => {
    getModelStats()
      .then((s) => {
        setStats(s);
        setThreshold(s.optimal_threshold ?? 0.5);
      })
      .catch((err) =>
        setError(err?.response?.data?.detail || err.message || "Fehler beim Laden")
      )
      .finally(() => setLoading(false));
  }, []);

  const hasLiveData = stats?.test_probs?.length > 0;

  // Compute live metrics from raw test scores (instant JS, no API call)
  const liveMetrics = useMemo(() => {
    if (!hasLiveData) return null;
    return computeAt(stats.test_probs, stats.test_labels, threshold);
  }, [hasLiveData, stats, threshold]);

  // Build curves for chart (memoised — only re-computed when stats change)
  const curves = useMemo(() => {
    if (!hasLiveData) return null;
    return buildCurves(stats.test_probs, stats.test_labels);
  }, [hasLiveData, stats]);

  // Use live metrics if available, fall back to static JSON values
  const m = liveMetrics ?? (stats ? {
    sensitivity: stats.sensitivity,
    specificity: stats.specificity,
    accuracy:    stats.accuracy,
    ppv:         stats.ppv,
    npv:         stats.npv,
    tp:          stats.tp,
    fp:          stats.fp,
    tn:          stats.tn,
    fn:          stats.fn,
  } : null);

  const optThresh = stats?.optimal_threshold ?? 0.5;
  const isOptimal = Math.abs(threshold - optThresh) < 0.001;

  return (
    <div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <h1 style={{ fontSize: "1.4rem", fontWeight: 700, marginBottom: "1.25rem", color: "#1a365d" }}>
        Modell-Metriken
      </h1>

      {loading && (
        <p style={{ color: "#2c5282" }}><span style={spinnerStyle} />Lade Metriken…</p>
      )}

      {error && (
        <div style={{
          background: "#fff5f5", border: "1px solid #fc8181",
          borderRadius: 8, padding: "1rem 1.25rem", color: "#c53030",
        }}>
          <strong>Fehler:</strong> {error}
          <br />
          <small>Tipp: <code>python evaluate.py --save-json</code> ausführen.</small>
        </div>
      )}

      {stats && m && (
        <>
          {/* ── Threshold Explorer ────────────────────────────────── */}
          <div style={{
            background: "#fff", border: "1px solid #bee3f8",
            borderRadius: 10, padding: "1rem 1.25rem", marginBottom: "1.5rem",
          }}>
            <div style={{
              display: "flex", alignItems: "center", gap: "0.75rem",
              marginBottom: "0.75rem", flexWrap: "wrap",
            }}>
              <span style={{ fontWeight: 600, color: "#1a365d", fontSize: "0.95rem" }}>
                Schwellenwert-Explorer
              </span>
              <span style={{
                fontSize: "1.15rem", fontWeight: 700, color: "#2b6cb0",
                background: "#ebf4ff", borderRadius: 6, padding: "1px 10px",
              }}>
                {threshold.toFixed(3)}
              </span>
              {isOptimal && (
                <span style={{
                  fontSize: "0.75rem", background: "#c6f6d5",
                  color: "#276749", borderRadius: 10, padding: "2px 8px", fontWeight: 600,
                }}>
                  optimal (Youden's J)
                </span>
              )}
              {!isOptimal && (
                <button
                  onClick={() => setThreshold(optThresh)}
                  style={{
                    fontSize: "0.78rem", color: "#2b6cb0", borderColor: "#bee3f8",
                    background: "#ebf4ff", border: "1px solid #bee3f8",
                    borderRadius: 6, padding: "0.25rem 0.6rem", cursor: "pointer",
                  }}
                >
                  ↺ Optimal ({optThresh.toFixed(3)})
                </button>
              )}
              {!hasLiveData && (
                <span style={{ fontSize: "0.75rem", color: "#a0aec0" }}>
                  (Statische Werte — <code>evaluate.py --save-json</code> für Live-Explorer)
                </span>
              )}
            </div>

            {/* Slider */}
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <span style={{ fontSize: "0.78rem", color: "#718096", minWidth: 24 }}>0</span>
              <input
                type="range" min={0} max={1} step={0.001}
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                style={{ flex: 1, accentColor: "#3182ce" }}
                disabled={!hasLiveData}
              />
              <span style={{ fontSize: "0.78rem", color: "#718096", minWidth: 16 }}>1</span>
            </div>

            {/* Balance indicator */}
            {hasLiveData && (
              <div style={{
                marginTop: "0.5rem", fontSize: "0.8rem",
                display: "flex", gap: "1.5rem", color: "#4a5568",
              }}>
                <span>
                  Sensitivität:{" "}
                  <strong style={{ color: "#e53e3e" }}>{(m.sensitivity * 100).toFixed(1)}%</strong>
                </span>
                <span>
                  Spezifität:{" "}
                  <strong style={{ color: "#38a169" }}>{(m.specificity * 100).toFixed(1)}%</strong>
                </span>
                <span style={{ color: Math.abs(m.sensitivity - m.specificity) < 0.03 ? "#276749" : "#a0aec0" }}>
                  {Math.abs(m.sensitivity - m.specificity) < 0.03
                    ? "✓ ausgeglichen"
                    : m.sensitivity > m.specificity
                    ? "↑ sensitiver"
                    : "↑ spezifischer"}
                </span>
              </div>
            )}
          </div>

          {/* ── SVG Chart ─────────────────────────────────────────── */}
          {hasLiveData && curves && (
            <div style={{
              background: "#fff", border: "1px solid #e2e8f0",
              borderRadius: 10, padding: "1rem 1.25rem", marginBottom: "1.5rem",
            }}>
              <div style={{ fontSize: "0.88rem", fontWeight: 600, color: "#2d3748", marginBottom: "0.5rem" }}>
                Sensitivität &amp; Spezifität vs. Schwellenwert
                <span style={{ fontWeight: 400, color: "#a0aec0", marginLeft: 8, fontSize: "0.78rem" }}>
                  — blau: aktuell · lila: Schnittpunkt · gelb: optimal
                </span>
              </div>
              <ThresholdChart
                curves={curves}
                threshold={threshold}
                optimalThreshold={optThresh}
              />
            </div>
          )}

          {/* ── Live metrics grid ─────────────────────────────────── */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
            gap: "1rem", marginBottom: "1.75rem",
          }}>
            <MetricCard label="AUC-ROC"         value={stats.auc_roc}  highlight />
            <MetricCard label="Sensitivität"    value={m.sensitivity} />
            <MetricCard label="Spezifität"      value={m.specificity} />
            <MetricCard label="Accuracy"        value={m.accuracy} />
            <MetricCard label="PPV (Precision)" value={m.ppv} />
            <MetricCard label="NPV"             value={m.npv} />
          </div>

          {/* ── Confusion matrix (live) ───────────────────────────── */}
          <h2 style={{ fontSize: "1rem", fontWeight: 600, color: "#2d3748", marginBottom: "0.6rem" }}>
            Konfusionsmatrix
            {hasLiveData && (
              <span style={{ fontWeight: 400, color: "#a0aec0", fontSize: "0.8rem", marginLeft: 6 }}>
                (Schwellenwert {threshold.toFixed(3)})
              </span>
            )}
          </h2>
          <table style={{ borderCollapse: "collapse", fontSize: "0.88rem", marginTop: "0.5rem" }}>
            <thead>
              <tr>
                <th style={th(false)} />
                <th style={th(false)}>Pred: Normal</th>
                <th style={th(false)}>Pred: Fraktur</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={th(false)}><strong>Tatsächlich: Normal</strong></td>
                <td style={td(true)}>TN = {m.tn}</td>
                <td style={td(false)}>FP = {m.fp}</td>
              </tr>
              <tr>
                <td style={th(false)}><strong>Tatsächlich: Fraktur</strong></td>
                <td style={td(false)}>FN = {m.fn}</td>
                <td style={td(true)}>TP = {m.tp}</td>
              </tr>
            </tbody>
          </table>

          {/* ── Footer info ───────────────────────────────────────── */}
          <div style={{
            marginTop: "1.5rem", padding: "0.85rem 1rem",
            background: "#fffff0", border: "1px solid #f6e05e",
            borderRadius: 8, fontSize: "0.85rem", color: "#744210",
          }}>
            <strong>Optimaler Schwellenwert (Youden's J):</strong> {optThresh}
            &nbsp;—&nbsp; Modell: <code>{stats.model_name}</code>
            &nbsp;·&nbsp;
            Train: {stats.n_train?.toLocaleString()} / Val: {stats.n_val?.toLocaleString()} / Test: {stats.n_test?.toLocaleString()}
          </div>
        </>
      )}
    </div>
  );
}

// ── Table cell helpers ────────────────────────────────────────────────────────

const th = (filled) => ({
  padding: "0.65rem 1.1rem",
  border: "1px solid #e2e8f0",
  background: filled ? "#bee3f8" : "#edf2f7",
  fontWeight: 600,
  color: "#2d3748",
  textAlign: "center",
});

const td = (filled) => ({
  padding: "0.65rem 1.1rem",
  border: "1px solid #e2e8f0",
  fontWeight: filled ? 700 : 400,
  background: filled ? "#bee3f8" : "#fff",
  textAlign: "center",
  color: filled ? "#1a365d" : "#4a5568",
  transition: "background 0.2s",
});
